"""
Test de carga de videos con pre-procesamiento manual
"""
import torch
from transformers import AutoProcessor
from dataset import IEMOCAPDataset
import os

def test_video_loading():
    print("="*70)
    print("TEST: Carga y Pre-procesamiento Manual de Videos")
    print("="*70)
    
    # Cargar processor
    print("\n[1/3] Cargando processor...")
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("✓ Processor cargado")
    
    # Cargar dataset
    print("\n[2/3] Cargando dataset...")
    dataset = IEMOCAPDataset(
        "train.json",
        processor,
        max_frames=4
    )
    print(f"✓ Dataset cargado: {len(dataset)} samples")
    
    # Probar primeros 5 samples
    print("\n[3/3] Probando primeros 5 samples...")
    print("="*70)
    
    successful = 0
    failed = 0
    
    for i in range(min(5, len(dataset))):
        print(f"\nSample {i+1}:")
        try:
            sample = dataset[i]
            
            if sample is None:
                print("  ❌ Sample es None")
                failed += 1
                continue
            
            print(f"  - input_ids shape: {sample['input_ids'].shape}")
            print(f"  - attention_mask shape: {sample['attention_mask'].shape}")
            print(f"  - labels shape: {sample['labels'].shape}")
            
            # Verificar pixel_values
            pv = sample['pixel_values']
            print(f"  - pixel_values shape: {pv.shape}")
            print(f"  - pixel_values dtype: {pv.dtype}")
            print(f"  - pixel_values min/max: [{pv.min():.3f}, {pv.max():.3f}]")
            print(f"  - pixel_values mean: {pv.mean():.3f}")
            
            # CRÍTICO: Verificar que NO son todos ceros
            if pv.abs().sum().item() < 0.01:
                print(f"  ❌ pixel_values son todos CEROS!")
                failed += 1
            else:
                print(f"  ✅ pixel_values VÁLIDOS (no ceros)")
                successful += 1
            
            # Verificar labels
            valid_labels = (sample['labels'] != -100).sum().item()
            print(f"  - tokens válidos para entrenamiento: {valid_labels}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    print(f"Exitosos: {successful}/5")
    print(f"Fallidos: {failed}/5")
    
    if successful > 0:
        print(f"\n✅ {successful} videos procesados CORRECTAMENTE con frames válidos")
        print("\n🚀 LISTO PARA ENTRENAR!")
        print("   Ejecuta: python train.py")
    else:
        print("\n❌ NINGÚN video se procesó correctamente")
        print("\nVerifica:")
        print("1. opencv-python está instalado: pip install opencv-python")
        print("2. Los videos son accesibles y legibles")
        print("3. torchvision está instalado: pip install torchvision")
    
    print("="*70)

if __name__ == "__main__":
    test_video_loading()