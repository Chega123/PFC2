import os
import torch
from torch.utils.data import Dataset
import glob

class PreprocessedIEMOCAPDataset(Dataset):
    def __init__(self, data_dir):
        """
        Inicializa el dataset desde un directorio de tensores .pt pre-procesados.
        
        Args:
            data_dir (str): El path al directorio (ej: "./preprocessed_data/train/")
        """
        print(f"Cargando archivos de tensores desde: {data_dir}")
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        
        if len(self.data_files) == 0:
            raise FileNotFoundError(f"No se encontraron archivos .pt en {data_dir}. ¿Ejecutaste preprocess_dataset.py?")
            
        print(f"Se encontraron {len(self.data_files)} samples pre-procesados.")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        Carga un único sample (diccionario de tensores) desde el disco.
        Esto es extremadamente rápido.
        """
        file_path = self.data_files[idx]
        
        try:
            data = torch.load(file_path, map_location='cpu') # Cargar a CPU (collate_fn lo moverá a GPU)
            return data
        except Exception as e:
            print(f"ERROR: No se pudo cargar {file_path}: {e}")
            return None