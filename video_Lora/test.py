"""
Entrenamiento para Qwen2.5-VL con LoRA y sistema de checkpoints
"""
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
        print(f"✓ ÉPOCA {epoch_num} COMPLETADA")
        print(f"⏱  Tiempo: {epoch_time:.2f} segundos ({epoch_time/60:.2f} minutos)")
        print(f"{'='*70}\n")

class LossLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            step = state.global_step
            loss = logs['loss']
            lr = logs.get('learning_rate', 0)
            print(f"Step {step} | Loss: {loss:.4f} | LR: {lr:.2e}")


def find_latest_checkpoint(output_dir):
    """Encuentra el checkpoint más reciente"""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    # Ordenar por número de paso
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest = checkpoints[-1]
    
    print(f"\n[INFO] Checkpoint encontrado: {latest}")
    return latest


def main():
    # ============================================================
    # CONFIGURACIÓN GENERAL
    # ============================================================
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    output_dir = "./qwen_iemocap_checkpoints"
    num_epochs = 5
    
    # NUEVO: Opción para reanudar entrenamiento
    RESUME_FROM_CHECKPOINT = True  # Cambiar a False para empezar desde cero
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO - IEMOCAP EMOTION RECOGNITION")
    print("="*70)
    print(f"Modelo: {model_name}")
    print(f"Épocas: {num_epochs}")
    print(f"Output: {output_dir}")
    print(f"Método: LoRA (r=16, alpha=32)")
    print(f"Reanudar: {'Sí' if RESUME_FROM_CHECKPOINT else 'No'}")
    print("="*70 + "\n")
    
    # Buscar checkpoint existente
    resume_checkpoint = None
    if RESUME_FROM_CHECKPOINT:
        resume_checkpoint = find_latest_checkpoint(output_dir)
        if resume_checkpoint:
            print(f"✓ Se reanudará desde: {resume_checkpoint}\n")
        else:
            print("[INFO] No se encontró checkpoint, comenzando desde cero\n")
    
    # ============================================================
    # CARGAR MODELO Y PROCESSOR
    # ============================================================
    print("[1/5] Cargando modelo y processor...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Cargar modelo base
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
    print("✓ Modelo base cargado en 4-bit\n")

    # ============================================================
    # CONFIGURACIÓN PEFT
    # ============================================================
    print("[2/5] Configurando PEFT...")
    
    if resume_checkpoint:
        # ← AQUÍ está el código que actualicé
        print(f"[INFO] Cargando adaptadores LoRA desde checkpoint...")
        model = prepare_model_for_kbit_training(model)  # ← Agregado aquí
        model = PeftModel.from_pretrained(
            model,
            resume_checkpoint,
            is_trainable=True
        )
        # Verificar parámetros entrenables
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        print("✓ Modelo LoRA cargado desde checkpoint")
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
                "o_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, peft_config)
        print("✓ Nuevo modelo LoRA configurado")
    
    # Optimizaciones de memoria
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    print("\nParámetros entrenables:")
    model.print_trainable_parameters()
    print()
    
    # ============================================================
    # CARGAR DATASET
    # ============================================================
    print("[3/5] Cargando dataset...")
    train_dataset = IEMOCAPDataset(
        "train.json",
        processor,
        max_frames=3
    )
    print(f"✓ Dataset: {len(train_dataset)} samples\n")
    
    # ============================================================
    # ARGUMENTOS DE ENTRENAMIENTO
    # ============================================================
    print("[4/5] Configurando entrenamiento...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Guardado de checkpoints
        save_strategy="steps",              # Guardar cada N steps
        save_steps=250,                     # Cada 200 steps (~40 min)
        save_total_limit=3,                 # Mantener solo últimos 3
        
        # Entrenamiento
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        
        # Optimización
        optim="paged_adamw_8bit",
        fp16=True,
        
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
        
        # NUEVO: Para reanudar correctamente
        load_best_model_at_end=False,
        ignore_data_skip=False,  # Importante: salta datos ya procesados
    )
    print("✓ Argumentos configurados")
    print(f"  - Guardando checkpoint cada {training_args.save_steps} steps")
    print(f"  - Manteniendo últimos {training_args.save_total_limit} checkpoints\n")

    # ============================================================
    # CREAR TRAINER
    # ============================================================
    print("[5/5] Inicializando Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[TimeCallback(), LossLogCallback()]
    )
    print("✓ Trainer listo\n")

    # ============================================================
    # ENTRENAR
    # ============================================================
    print("="*70)
    print("COMENZANDO ENTRENAMIENTO")
    print("="*70 + "\n")
    
    if resume_checkpoint:
        print(f"[INFO] Reanudando desde step {resume_checkpoint.split('-')[-1]}\n")
    
    start_time = time.time()
    
    # Entrenar (con o sin checkpoint)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    total_time = time.time() - start_time
    
    # ============================================================
    # GUARDAR MODELO FINAL
    # ============================================================
    print("\n" + "="*70)
    print("GUARDANDO MODELO FINAL")
    print("="*70)
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"✓ Modelo guardado en: {final_model_path}")
    
    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"Tiempo total: {total_time/60:.2f} minutos")
    print(f"Checkpoints: {output_dir}")
    print(f"Modelo final: {final_model_path}")
    print("="*70 + "\n")
    
    print("Para reanudar entrenamiento:")
    print(f"  1. Deja RESUME_FROM_CHECKPOINT = True")
    print(f"  2. Ejecuta: python train.py")
    print("\nPara evaluar:")
    print(f"  python eval.py")


if __name__ == "__main__":
    main()