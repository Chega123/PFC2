import os
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model, 
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType,
    PeftModel
)
from dataset import IEMOCAPDataset, collate_fn
import time
import glob

class TimeCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.start_time
        epoch_num = int(state.epoch)
        print(f"Epoca {epoch_num} Completada")
        print(f"Tiempo: {epoch_time:.2f} segundos ({epoch_time/60:.2f} minutos)")


class LossLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            step = state.global_step
            loss = logs['loss']
            lr = logs.get('learning_rate', 0)
            print(f"Step {step} | Loss: {loss:.4f} | LR: {lr:.2e}")


def find_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest = checkpoints[-1]
    
    print(f"\nCheckpoint encontrado: {latest}")
    return latest


def main():

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    output_dir = "./qwen_iemocap_checkpoints"
    num_epochs = 4
    
    RESUME_FROM_CHECKPOINT = True  
    
    print("Train")
    print(f"Modelo: {model_name}")
    print(f"Epocas: {num_epochs}")
    print(f"Output: {output_dir}")

    print(f"Reanudar: {'Sí' if RESUME_FROM_CHECKPOINT else 'No'}")

    
    resume_checkpoint = None
    if RESUME_FROM_CHECKPOINT:
        resume_checkpoint = find_latest_checkpoint(output_dir)
        if resume_checkpoint:
            print(f"Se reanudará desde: {resume_checkpoint}\n")
        else:
            print("No se encontró checkpoint, comenzando desde cero\n")
    
    print("Cargando modelo y processor...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("Modelo base cargado en 4-bit\n")


    # Config PEFT
    print("Configurando PEFT...")
    
    if resume_checkpoint:
        model = prepare_model_for_kbit_training(model) 
        model = PeftModel.from_pretrained(
            model,
            resume_checkpoint,
            is_trainable=True
        )
        # Verificar parámetros entrenables
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        print("Modelo LoRA cargado desde checkpoint")
    else:
        # Crear nuevo modelo LoRA
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, peft_config)
    
    # Optimizaciones de memoria
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    print("\nParámetros entrenables:")
    model.print_trainable_parameters()
    print()
    

    print("Cargando dataset")
    train_dataset = IEMOCAPDataset(
        "train.json",
        processor,
        max_frames=3
    )
    print(f"Dataset: {len(train_dataset)} samples\n")
    
    training_args = TrainingArguments(
        output_dir=output_dir,

        # Guardado de checkpoints
        save_strategy="steps",              
        save_steps=250,                     
        save_total_limit=3,                 
    
        # Entrenamiento
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_ratio=0.15,
        weight_decay=0.05,
        max_grad_norm=0.5,
        lr_scheduler_type="cosine",
        
        # Optimización
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=True,
        
        # Logging
        logging_strategy="steps",
        logging_steps=10,
        
        # Sin evaluación
        eval_strategy="no",
        
        # Otros
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        
        #Para reanudar correctamente
        load_best_model_at_end=False,
        ignore_data_skip=False, 
    )
    print("Argumentos configurados")
    print(f"Guardando checkpoint cada {training_args.save_steps} steps")
    print(f"Manteniendo últimos {training_args.save_total_limit} checkpoints\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[TimeCallback(), LossLogCallback()]
    )

    print("Comenzando Train")
    
    if resume_checkpoint:
        print(f"Reanudando desde step {resume_checkpoint.split('-')[-1]}\n")
    
    start_time = time.time()
    
    # Entrenar (con o sin checkpoint)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    total_time = time.time() - start_time
    

    print("Guardando modelo final")

    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"Modelo guardado en: {final_model_path}")

    print("Train completo")

    print(f"Tiempo total: {total_time/60:.2f} minutos")
    print(f"Checkpoints: {output_dir}")
    print(f"Modelo final: {final_model_path}")

if __name__ == "__main__":
    main()