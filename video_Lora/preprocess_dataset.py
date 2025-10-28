import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from dataset import IEMOCAPDataset  # Importamos tu dataset original
from tqdm import tqdm
import glob
import time

# --- CONFIGURACIÓN ---
JSON_FILE = "train.json"  # El JSON que quieres procesar (train.json, val.json, etc.)
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./preprocessed_data/train/" # Directorio de salida para los tensores
MAX_FRAMES = 32  # ¡Aquí puedes aumentarlo! Prueba con 8 o 16.
NUM_WORKERS = 8 # Número de hilos para procesar videos en paralelo
# ---------------------

def main():
    print(f"Iniciando pre-procesamiento de: {JSON_FILE}")
    print(f"Guardando tensores en: {OUTPUT_DIR}")
    print(f"Frames por video: {MAX_FRAMES}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Cargar Processor (necesario para el dataset)
    print("Cargando processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    # 2. Cargar tu Dataset original
    #    Aquí es donde se validarán los videos
    print("Cargando dataset original (validando videos)...")
    dataset = IEMOCAPDataset(
        JSON_FILE,
        processor,
        max_frames=MAX_FRAMES
    )
    
    print(f"Dataset original cargado. Total de samples a procesar: {len(dataset)}")
    
    # 3. Iterar y guardar
    #    Usamos un DataLoader solo para aprovechar el 'num_workers'
    #    y hacer el __getitem__ (que es lento) en paralelo.
    
    # NOTA: batch_size=1 y collate_fn=None. 
    # Queremos procesar y guardar un sample a la vez.
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=lambda x: x[0] # Devuelve el sample directo, no un batch
    )
    
    print(f"\nProcesando videos con {NUM_WORKERS} workers...")
    start_time = time.time()
    
    for i, sample_data in enumerate(tqdm(loader, desc="Pre-procesando")):
        if sample_data is None:
            print(f"Skipping sample {i} (NoneType)")
            continue
            
        # El nombre del archivo se basará en el índice para simplicidad
        # (Podrías usar el 'id' del JSON si quisieras)
        save_path = os.path.join(OUTPUT_DIR, f"sample_{i:05d}.pt")
        
        # Guardar el diccionario de tensores
        torch.save(sample_data, save_path)

    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("PRE-PROCESAMIENTO COMPLETO")
    print(f"Tiempo total: {total_time/60:.2f} minutos")
    print(f"Tensores guardados en: {OUTPUT_DIR}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()