
"""
Evaluación optimizada para Qwen2.5-VL con mínimo uso de RAM/VRAM
"""
import os
import torch
import gc
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import PeftModel
from dataset import IEMOCAPDataset
from tqdm import tqdm
import json
from collections import defaultdict

def clear_memory():
    """Limpia memoria agresivamente"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    # ============================================================
    # CONFIGURACIÓN
    # ============================================================
    base_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    peft_model_path = "./qwen_iemocap_checkpoints/final_model"
    val_json = "val.json"
    max_frames = 2 # Reducido de 4 a 3
    
    # CONFIGURACIÓN DE EVALUACIÓN
    batch_size = 1  # Siempre 1 para eval (por simplicidad)
    save_every = 100  # Guardar resultados parciales cada N samples
    
    print("\n" + "="*70)
    print("EVALUACIÓN - IEMOCAP EMOTION RECOGNITION")
    print("="*70)
    print(f"Modelo base: {base_model_name}")
    print(f"Adaptador PEFT: {peft_model_path}")
    print(f"Dataset: {val_json}")
    print(f"Max frames: {max_frames}")
    print(f"Optimización: RAM/VRAM mínima")
    print("="*70 + "\n")
    
    # ============================================================
    # CARGAR MODELO BASE
    # ============================================================
    print("[1/4] Cargando modelo base...")
    
    # Cuantización 4-bit agresiva
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # OPTIMIZACIÓN RAM
    )
    print("✓ Modelo base cargado\n")
    
    # ============================================================
    # CARGAR ADAPTADOR PEFT
    # ============================================================
    print("[2/4] Cargando adaptador PEFT...")
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.eval()
    
    # OPTIMIZACIÓN: Desactivar dropout y gradient checkpointing
    for module in model.modules():
        if hasattr(module, 'dropout'):
            module.dropout = 0.0
    
    print("✓ Adaptador cargado\n")
    clear_memory()
    
    # ============================================================
    # CARGAR PROCESSOR Y DATASET
    # ============================================================
    print("[3/4] Cargando processor y dataset...")
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    # Cargar dataset completo (tienes suficiente RAM)
    val_dataset = IEMOCAPDataset(
        val_json,
        processor,
        max_frames=max_frames
    )
    print(f"✓ Dataset: {len(val_dataset)} samples\n")
    
    # ============================================================
    # EVALUAR
    # ============================================================
    print("[4/4] Evaluando modelo...")
    print("="*70 + "\n")
    
    # Métricas
    total_loss = 0.0
    num_samples = 0
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))
    predictions = []
    
    # Archivo de resultados parciales
    partial_results_path = os.path.join(
        os.path.dirname(peft_model_path), 
        "eval_partial.json"
    )
    
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc="Evaluando"):
            try:
                # Obtener sample directamente del dataset
                sample = val_dataset[i]
                
                # Mover a GPU (batch size 1)
                input_ids = sample['input_ids'].unsqueeze(0).to(model.device)
                attention_mask = sample['attention_mask'].unsqueeze(0).to(model.device)
                pixel_values = sample['pixel_values'].unsqueeze(0).to(model.device)
                image_grid_thw = sample['image_grid_thw'].to(model.device)  # Tensor directo, sin lista
                labels = sample['labels'].unsqueeze(0).to(model.device)
                
                # Forward pass para loss
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_samples += 1
                
                # Generar predicción
                user_length = (labels[0] != -100).nonzero()[0].item()
                gen_input_ids = input_ids[:, :user_length]
                gen_attention_mask = attention_mask[:, :user_length]
                
                generated_ids = model.generate(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,  # Mismo formato que arriba
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
                
                # Decodificar
                prediction = processor.decode(
                    generated_ids[0][user_length:], 
                    skip_special_tokens=True
                ).strip().lower()
                
                # Ground truth
                gt_ids = labels[0][labels[0] != -100]
                ground_truth = processor.decode(
                    gt_ids, 
                    skip_special_tokens=True
                ).strip().lower()
                
                # Calcular accuracy
                is_correct = prediction == ground_truth
                if is_correct:
                    correct += 1
                total += 1
                
                # Matriz de confusión
                confusion[ground_truth][prediction] += 1
                
                # Guardar ejemplos (primeros 5 + todos los errores)
                if i < 5 or not is_correct:
                    predictions.append({
                        'sample': i,
                        'video': val_dataset.data[i]['video'],
                        'prediction': prediction,
                        'ground_truth': ground_truth,
                        'correct': is_correct
                    })
                
                # Liberar memoria de tensores grandes
                del input_ids, attention_mask, pixel_values, labels, generated_ids, outputs
                
                # Limpieza ligera cada 20 samples (no tan agresiva)
                if i % 20 == 0 and i > 0:
                    clear_memory()
                
                # Guardar resultados parciales
                if i % save_every == 0 and i > 0:
                    partial_results = {
                        'samples_processed': num_samples,
                        'current_accuracy': (correct / total * 100) if total > 0 else 0,
                        'avg_loss': total_loss / num_samples if num_samples > 0 else 0
                    }
                    with open(partial_results_path, 'w') as f:
                        json.dump(partial_results, f, indent=2)
                
            except Exception as e:
                print(f"\nError en sample {i}: {e}")
                clear_memory()
                continue
    
    # Limpieza final
    clear_memory()
    
    # ============================================================
    # CALCULAR MÉTRICAS
    # ============================================================
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    print("\n" + "="*70)
    print("RESULTADOS")
    print("="*70)
    print(f"Samples evaluados: {num_samples}/{len(val_dataset)}")
    print(f"Loss promedio: {avg_loss:.4f}")
    print(f"Perplexity: {torch.exp(torch.tensor(avg_loss)):.2f}")
    print(f"\nACCURACY: {accuracy:.2f}% ({correct}/{total})")
    print("="*70 + "\n")
    
    # ============================================================
    # MÉTRICAS POR CLASE
    # ============================================================
    print("MÉTRICAS POR CLASE:")
    print("="*70)
    
    emotions = sorted(set(list(confusion.keys()) + 
                         [pred for preds in confusion.values() for pred in preds.keys()]))
    
    print(f"{'Emoción':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    metrics_per_class = {}
    
    for emotion in emotions:
        tp = confusion[emotion][emotion]
        fp = sum(confusion[other][emotion] for other in emotions if other != emotion)
        fn = sum(confusion[emotion][other] for other in emotions if other != emotion)
        support = tp + fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_per_class[emotion] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
        
        print(f"{emotion:<15} {precision*100:>10.2f}%  {recall*100:>10.2f}%  {f1*100:>10.2f}%  {support:>10}")
    
    # Macro average
    if emotions:
        macro_precision = sum(m['precision'] for m in metrics_per_class.values()) / len(emotions)
        macro_recall = sum(m['recall'] for m in metrics_per_class.values()) / len(emotions)
        macro_f1 = sum(m['f1'] for m in metrics_per_class.values()) / len(emotions)
        
        print("-" * 70)
        print(f"{'MACRO AVG':<15} {macro_precision*100:>10.2f}%  {macro_recall*100:>10.2f}%  {macro_f1*100:>10.2f}%  {total:>10}")
    
    print("="*70 + "\n")
    
    # ============================================================
    # MOSTRAR EJEMPLOS
    # ============================================================
    if predictions:
        print("EJEMPLOS DE PREDICCIONES:")
        print("="*70)
        
        correct_preds = [p for p in predictions if p['correct']]
        wrong_preds = [p for p in predictions if not p['correct']]
        
        print("\nPREDICCIONES CORRECTAS (primeras 3):")
        for pred in correct_preds[:3]:
            print(f"\n  Sample {pred['sample']}:")
            print(f"    Predicción: {pred['prediction']}")
            print(f"    Ground Truth: {pred['ground_truth']}")
        
        if wrong_preds:
            print(f"\n\nPREDICCIONES INCORRECTAS ({len(wrong_preds)} errores):")
            for pred in wrong_preds[:10]:
                print(f"\n  Sample {pred['sample']}:")
                print(f"    Predicción: {pred['prediction']}")
                print(f"    Ground Truth: {pred['ground_truth']}")
        
        print("\n" + "="*70 + "\n")
    
    # ============================================================
    # GUARDAR RESULTADOS FINALES
    # ============================================================
    results = {
        'model': peft_model_path,
        'num_samples': num_samples,
        'avg_loss': avg_loss,
        'perplexity': float(torch.exp(torch.tensor(avg_loss))),
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'metrics_per_class': metrics_per_class,
        'confusion_matrix': {k: dict(v) for k, v in confusion.items()},
        'examples': predictions[:20]
    }
    
    results_path = os.path.join(
        os.path.dirname(peft_model_path), 
        "eval_results.json"
    )
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Eliminar resultados parciales
    if os.path.exists(partial_results_path):
        os.remove(partial_results_path)
    
    print(f"✓ Resultados guardados en: {results_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()