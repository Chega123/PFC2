import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MultimodalEmotionDatasetQwen(Dataset):
    
    def __init__(
        self, 
        text_dir, 
        audio_dir, 
        qwen_embed_dir,  # NUEVO: carpeta con embeddings de Qwen
        label_dir, 
        sessions=None, 
        max_waveform_length=160000,
        split='train'  # 'train', 'val', o 'test'
    ):
        self.text_dir = text_dir
        self.audio_dir = audio_dir
        self.qwen_embed_dir = qwen_embed_dir
        self.label_dir = label_dir
        self.max_waveform_length = max_waveform_length
        self.sessions = sessions if sessions is not None else []
        self.split = split
        
        self.emotion_to_idx = {
            "neutral": 0, 
            "happy": 1, 
            "sad": 2, 
            "angry": 3, 
            "excited": 1
        }
        
        self.samples = []
        
        # Verificar directorios
        for dir_path in [text_dir, audio_dir, label_dir]:
            if not os.path.exists(dir_path):
                raise ValueError(f"Directory {dir_path} does not exist.")
        
        # Verificar que existan embeddings de Qwen
        qwen_split_dir = os.path.join(qwen_embed_dir, split)
        if not os.path.exists(qwen_split_dir):
            raise ValueError(f"Qwen embeddings directory not found: {qwen_split_dir}")
        
        # Recolectar samples
        self._collect_samples()
        
        print(f"\n[INFO] Dataset cargado ({split}):")
        print(f"  - Total samples: {len(self.samples)}")
        print(f"  - Sessions: {sessions}")
        print(f"  - Qwen embeddings: {qwen_split_dir}")
    
    def _collect_samples(self):

        qwen_split_dir = os.path.join(self.qwen_embed_dir, self.split)
        
        for session in os.listdir(self.text_dir):
            if self.sessions and not any(session.lower() == s.lower() for s in self.sessions):
                continue
            
            session_text_path = os.path.join(self.text_dir, session)
            session_audio_path = os.path.join(self.audio_dir, session)
            session_label_path = os.path.join(self.label_dir, session)
            
            for gender in os.listdir(session_text_path):
                text_gender_path = os.path.join(session_text_path, gender)
                audio_gender_path = os.path.join(session_audio_path, gender)
                label_gender_path = os.path.join(session_label_path, gender)
                
                # Obtener nombres de archivo base
                text_files = {os.path.splitext(f)[0] for f in os.listdir(text_gender_path) if f.endswith(".npy")}
                audio_files = {os.path.splitext(f)[0] for f in os.listdir(audio_gender_path) if f.endswith(".npy")}
                label_files = {os.path.splitext(f)[0] for f in os.listdir(label_gender_path) if f.endswith(".csv")}
                
                # Intersección
                common_files = text_files & audio_files & label_files
                
                for filename in common_files:
                    # Verificar que exista el embedding de Qwen
                    qwen_embed_file = os.path.join(qwen_split_dir, f"{filename}.pt")
                    if not os.path.exists(qwen_embed_file):
                        continue
                    
                    text_file = os.path.join(text_gender_path, f"{filename}.npy")
                    audio_file = os.path.join(audio_gender_path, f"{filename}.npy")
                    label_file = os.path.join(label_gender_path, f"{filename}.csv")
                    
                    self.samples.append((
                        text_file, 
                        audio_file, 
                        qwen_embed_file,  # Embedding pre-extraído
                        label_file, 
                        filename
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text_file, audio_file, qwen_embed_file, label_file, filename = self.samples[idx]
        
        # ============================================================
        # CARGAR TEXTO
        # ============================================================
        text_data = np.load(text_file, allow_pickle=True)
        text_data = text_data.item() if isinstance(text_data, np.ndarray) and text_data.dtype == object else text_data
        
        input_ids = torch.tensor(text_data.get("input_ids"), dtype=torch.long)
        attention_mask = torch.tensor(text_data.get("attention_mask"), dtype=torch.long)
        
        # ============================================================
        # CARGAR AUDIO
        # ============================================================
        audio_data = np.load(audio_file, allow_pickle=True)
        audio_data = audio_data.item() if isinstance(audio_data, np.ndarray) and audio_data.dtype == object else audio_data
        
        # Verificar si tiene waveform o solo path
        if "waveform" in audio_data:
            # Caso 1: Waveform ya procesado
            waveform = audio_data["waveform"]
            if not isinstance(waveform, np.ndarray):
                waveform = np.array(waveform, dtype=np.float32)
            else:
                waveform = waveform.astype(np.float32)
        else:
            # Caso 2: Solo tiene path, cargar audio en tiempo real
            import librosa
            audio_path = audio_data.get("path")
            
            # Arreglar path relativo a absoluto
            if audio_path and not os.path.isabs(audio_path):
                # Asumir que la ruta es relativa a D:/tesis/tesis/
                base_dir = "D:/tesis/tesis"
                audio_path = os.path.join(base_dir, audio_path)
            
            if audio_path and os.path.exists(audio_path):
                waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
            else:
                # Fallback: crear waveform vacío (silencio)
                if audio_path:
                    print(f"⚠️  Audio no encontrado: {audio_path}")
                waveform = np.zeros(self.max_waveform_length, dtype=np.float32)
        
        # Asegurar que es 1D
        if waveform.ndim > 1:
            waveform = waveform.flatten()
        
        # Padding/truncate
        if len(waveform) > self.max_waveform_length:
            waveform = waveform[:self.max_waveform_length]
        else:
            waveform = np.pad(
                waveform, 
                (0, self.max_waveform_length - len(waveform)), 
                mode='constant'
            )
        
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # [1, max_waveform_length]
        
        # ============================================================
        # CARGAR EMBEDDING DE VIDEO (QWEN) <- NUEVO
        # ============================================================
        qwen_data = torch.load(qwen_embed_file, weights_only=False)
        video_embedding = qwen_data['embedding']  # [768]
        
        # ============================================================
        # CARGAR LABEL
        # ============================================================
        label_df = pd.read_csv(label_file, header=None)
        if len(label_df) != 1:
            raise ValueError(f"CSV inválido: {label_file}")
        
        emotion = str(label_df.iloc[0, 2]).strip().lower()
        if emotion not in self.emotion_to_idx:
            raise ValueError(f"Emoción desconocida: {emotion}")
        
        label = self.emotion_to_idx[emotion]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "waveform": waveform,
            "video_embedding": video_embedding,  # [768] <- NUEVO (en lugar de frames)
            "label": torch.tensor(label, dtype=torch.long),
            "filename": filename
        }


def get_multimodal_dataloader_qwen(
    text_dir, 
    audio_dir, 
    qwen_embed_dir, 
    label_dir, 
    sessions=None, 
    batch_size=4, 
    shuffle=True, 
    max_waveform_length=160000,
    split='train',
    generator=None
):

    dataset = MultimodalEmotionDatasetQwen(
        text_dir, 
        audio_dir, 
        qwen_embed_dir, 
        label_dir, 
        sessions, 
        max_waveform_length,
        split=split
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0,
        generator=generator
    )


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    # Configuración para Windows
    # IMPORTANTE: Ajusta estas rutas según tu estructura real
    text_dir = "D:/tesis/tesis/data/text_tokenized"
    audio_dir = "D:/tesis/tesis/data/audio_preprocessed"
    qwen_embed_dir = "D:/tesis/tesis/data/video_embeddings_qwen"  # Carpeta copiada desde WSL
    label_dir = "D:/tesis/tesis/data/labels"
    
    print("="*70)
    print("TEST - DATASET CON QWEN EMBEDDINGS")
    print("="*70)
    
    # Test train
    print("\n[1/2] Cargando train dataloader...")
    train_loader = get_multimodal_dataloader_qwen(
        text_dir, 
        audio_dir, 
        qwen_embed_dir, 
        label_dir, 
        sessions=["Session1", "Session2", "Session3","Session4"],
        batch_size=4,
        split='train',
    )
    
    # Test val
    print("\n[2/2] Cargando val dataloader...")
    val_loader = get_multimodal_dataloader_qwen(
        text_dir, 
        audio_dir, 
        qwen_embed_dir, 
        label_dir, 
        sessions=["Session5"],
        batch_size=4,
        shuffle=False,
        split='val'
    )
    
    print("\n" + "="*70)
    print("TEST DE BATCH")
    print("="*70)
    
    for batch in train_loader:
        print("\nBatch shapes:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  waveform: {batch['waveform'].shape}")
        print(f"  video_embedding: {batch['video_embedding'].shape}")  # [batch, 768] <- NUEVO
        print(f"  labels: {batch['label'].shape}")
        
        print("\nPrimeros 3 samples:")
        idx_to_emotion = {0: "neutral", 1: "happy", 2: "sad", 3: "angry"}
        for filename, label, emb in zip(batch['filename'][:3], batch['label'][:3], batch['video_embedding'][:3]):
            emotion = idx_to_emotion.get(label.item(), 'unknown')
            print(f"  {filename}: {label.item()} ({emotion})")
            print(f"    Video emb shape: {emb.shape}, mean: {emb.mean():.4f}, std: {emb.std():.4f}")
        
        break
    
    print("\n✓ Dataset funciona correctamente con embeddings de Qwen")
