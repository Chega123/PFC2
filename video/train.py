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
        print(f"\n{'='*70}")
        print(f"Época {epoch_num} Completada")
        print(f"Tiempo: {epoch_time:.2f} segundos ({epoch_time/60:.2f} minutos)")
        print(f"{'='*70}\n")


class LossLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            step = state.global_step
            loss = logs['loss']
            lr = logs.get('learning_rate', 0)
            print(f"Step {step:>5} | Loss: {loss:.4f} | LR: {lr:.2e}")


def find_latest_checkpoint(output_dir):
    """Encuentra el último checkpoint guardado"""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest = checkpoints[-1]
    
    print(f"\n✓ Checkpoint encontrado: {latest}")
    return latest


def main():
    # ============================================================
    # CONFIGURACIÓN
    # ============================================================
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    output_dir = "./qwen_iemocap_checkpoints"
    num_epochs = 1
    max_frames = 2  # Número de frames a extraer por video
    
    RESUME_FROM_CHECKPOINT = True  # Cambiar a False para empezar desde cero
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO - IEMOCAP EMOTION RECOGNITION")
    print("="*70)
    print(f"Modelo: {model_name}")
    print(f"Épocas: {num_epochs}")
    print(f"Max frames: {max_frames}")
    print(f"Output: {output_dir}")
    print(f"Reanudar: {'Sí' if RESUME_FROM_CHECKPOINT else 'No'}")
    print("="*70 + "\n")

    # Verificar si hay checkpoint para reanudar
    resume_checkpoint = None
    if RESUME_FROM_CHECKPOINT:
        resume_checkpoint = find_latest_checkpoint(output_dir)
        if resume_checkpoint:
            print(f"Se reanudará desde: {resume_checkpoint}\n")
        else:
            print("No se encontró checkpoint, comenzando desde cero\n")
    
    # ============================================================
    # CARGAR MODELO BASE
    # ============================================================
    print("[1/4] Cargando modelo base...")
    
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
    print("✓ Modelo base cargado en 4-bit\n")

    # ============================================================
    # CARGAR PROCESSOR
    # ============================================================
    print("[2/4] Cargando processor...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("✓ Processor cargado\n")

    # ============================================================
    # CONFIGURAR PEFT (LoRA)
    # ============================================================
    print("[3/4] Configurando PEFT...")
    
    if resume_checkpoint:
        # Cargar modelo LoRA existente
        model = prepare_model_for_kbit_training(model) 
        model = PeftModel.from_pretrained(
            model,
            resume_checkpoint,
            is_trainable=True
        )
        # Asegurar que los parámetros LoRA sean entrenables
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        print("✓ Modelo LoRA cargado desde checkpoint")
    else:
        # Crear nuevo modelo LoRA
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=8,                    # Rango de las matrices LoRA
            lora_alpha=16,           # Factor de escalado
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
        print("✓ Nuevo modelo LoRA creado")
    
    # Optimizaciones de memoria
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    print("\nParámetros entrenables:")
    model.print_trainable_parameters()
    print()
    
    # ============================================================
    # CARGAR DATASET
    # ============================================================
    print("[4/4] Cargando dataset...")
    train_dataset = IEMOCAPDataset(
        "train.json",
        processor,
        max_frames=max_frames
    )
    print(f"✓ Dataset: {len(train_dataset)} samples\n")
    
    # ============================================================
    # CONFIGURAR TRAINING ARGUMENTS
    # ============================================================
    print("Configurando argumentos de entrenamiento...")
    
    training_args = TrainingArguments(
        output_dir=output_dir,

        # Guardado de checkpoints
        save_strategy="steps",              
        save_steps=250,                     # Guardar cada 250 steps
        save_total_limit=3,                 # Mantener solo los últimos 3 checkpoints
    
        # Entrenamiento
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,      # Batch size por GPU
        gradient_accumulation_steps=16,     # Acumular 16 batches (batch efectivo = 16)
        learning_rate=2e-5,
        warmup_ratio=0.15,                  # 15% de steps para warmup
        weight_decay=0.05,
        max_grad_norm=0.5,                  # Gradient clipping
        lr_scheduler_type="cosine",
        
        # Optimización
        optim="paged_adamw_8bit",           # Optimizer de 8-bit para ahorrar memoria
        fp16=False,
        bf16=True,                          # Usar bfloat16
        
        # Logging
        logging_strategy="steps",
        logging_steps=10,                   # Log cada 10 steps
        
        # Sin evaluación durante entrenamiento
        eval_strategy="no",
        
        # Otros
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",                   # No usar wandb/tensorboard
        
        # Para reanudar correctamente
        load_best_model_at_end=False,
        ignore_data_skip=False,             # No saltear datos al reanudar
    )
    
    print("✓ Argumentos configurados")
    print(f"  - Guardando checkpoint cada {training_args.save_steps} steps")
    print(f"  - Manteniendo últimos {training_args.save_total_limit} checkpoints")
    print(f"  - Batch efectivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print()

    # ============================================================
    # CREAR TRAINER
    # ============================================================
    print("Creando Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[TimeCallback(), LossLogCallback()]
    )
    
    print("✓ Trainer creado\n")

    # ============================================================
    # ENTRENAR
    # ============================================================
    print("="*70)
    print("COMENZANDO ENTRENAMIENTO")
    print("="*70 + "\n")
    
    if resume_checkpoint:
        step = resume_checkpoint.split('-')[-1]
        print(f"Reanudando desde step {step}\n")
    
    start_time = time.time()
    
    # Entrenar (con o sin checkpoint)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70 + "\n")

    # ============================================================
    # GUARDAR MODELO FINAL
    # ============================================================
    print("Guardando modelo final...")

    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    
    print(f"✓ Modelo guardado en: {final_model_path}\n")

    # ============================================================
    # RESUMEN
    # ============================================================
    print("="*70)
    print("RESUMEN")
    print("="*70)
    print(f"Tiempo total: {total_time/60:.2f} minutos ({total_time/3600:.2f} horas)")
    print(f"Épocas completadas: {num_epochs}")
    print(f"Checkpoints: {output_dir}")
    print(f"Modelo final: {final_model_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
