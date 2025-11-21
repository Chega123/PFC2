""" import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast
from transformers import RobertaTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
#por cambiar para q sea general, poner tu propia direccion
sys.path.append('D:/tesis/audio')
sys.path.append('D:/tesis/texto')
sys.path.append('D:/tesis/video')
sys.path.append('D:/tesis/modelo de fusion')

from audio_model import Wav2VecEmotionClassifier
from texto_model import RobertaEmbeddingExtractor, get_tokenizer_and_model
from video_model import VideoEmbeddingExtractor
from hierarchical import HierarchicalFusionModule
from auto_attention import AutoAttentionFusionModule
from final_fusion import FinalFusionMLP
from Fusiondataset import get_multimodal_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


emotion_to_idx = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3, "excited": 1
}
idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}

text_tokenizer, text_model = get_tokenizer_and_model(model_name="roberta-base", return_classifier=False, device=device)
audio_model = Wav2VecEmotionClassifier(pretrained_model="facebook/wav2vec2-base", num_classes=4, dropout=0.3, num_frozen_layers=0).to(device)
video_model = VideoEmbeddingExtractor(hidden_size=768, num_layers=1, dropout=0.0, num_frozen_layers=0).to(device)
hierarchical_fusion = HierarchicalFusionModule(embed_dim=768).to(device)
auto_attention_fusion = AutoAttentionFusionModule(embed_dim=768).to(device)
final_mlp = FinalFusionMLP(embed_dim=768).to(device)

#lo mismo q antes tu misma direccion
checkpoint = torch.load("D:/tesis/texto/checkpoints_final/roberta_best_f1_checkpoint.pth", map_location=device)
text_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
checkpoint = torch.load("D:/tesis/audio/checkpoint_audio/best_model.pth", map_location=device)
audio_model.load_state_dict(checkpoint["model_state_dict"])
checkpoint = torch.load("D:/tesis/video/checkpoints/finetune/best_model.pth", map_location=device)
video_model.load_state_dict(checkpoint, strict=False)
checkpoint = torch.load("D:/tesis/modelo de fusion/fusion_checkpoint.pth", map_location=device)
hierarchical_fusion.load_state_dict(checkpoint["hierarchical_state_dict"])
auto_attention_fusion.load_state_dict(checkpoint["auto_attention_state_dict"])
final_mlp.load_state_dict(checkpoint["final_mlp_state_dict"])

optimizer = torch.optim.AdamW(
    list(hierarchical_fusion.parameters()) + 
    list(auto_attention_fusion.parameters()) + 
    list(final_mlp.parameters()),
    lr=1e-4, weight_decay=1e-2
)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

def train_epoch(dataloader, models, optimizer, criterion, device, epoch):
    text_model, audio_model, video_model, hierarchical_fusion, auto_attention_fusion, final_mlp = models
    for model in models:
        model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        waveform = batch["waveform"].to(device)
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            text_emb = text_model.extract_features(input_ids, attention_mask)
            audio_emb = audio_model.embedding.extract_features(waveform)
            video_emb = video_model.extract_features(frames)
            v1 = hierarchical_fusion(text_emb, audio_emb, video_emb)
            v2 = auto_attention_fusion(text_emb, audio_emb, video_emb)
            logits = final_mlp(v1, v2)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy())
        
        torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    return epoch_loss, epoch_acc, epoch_f1

def evaluate(dataloader, models, criterion, device, epoch):
    text_model, audio_model, video_model, hierarchical_fusion, auto_attention_fusion, final_mlp = models
    for model in models:
        model.eval()
    
    all_preds = []
    all_labels = []
    running_loss = 0.0
    first_batch = True
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            waveform = batch["waveform"].to(device)
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            filenames = batch["filename"]
            
            with autocast():
                text_emb = text_model.extract_features(input_ids, attention_mask)
                audio_emb = audio_model.embedding.extract_features(waveform)
                video_emb = video_model.extract_features(frames)
                v1 = hierarchical_fusion(text_emb, audio_emb, video_emb)
                v2 = auto_attention_fusion(text_emb, audio_emb, video_emb)
                logits = final_mlp(v1, v2)
                loss = criterion(logits, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy())
            
            if first_batch:
                print("\nLabels per sample in validation (first batch):")
                for filename, label, pred in zip(filenames, labels.cpu().numpy(), preds):
                    print(f"{filename}: True Label {label} ({idx_to_emotion.get(label, 'unknown')}), Predicted {pred} ({idx_to_emotion.get(pred, 'unknown')})")
                first_batch = False
            
            torch.cuda.empty_cache()
    
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[idx_to_emotion[i] for i in [0, 1, 2, 3]],
                yticklabels=[idx_to_emotion[i] for i in [0, 1, 2, 3]])
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"D:/tesis/modelo de fusion/confusion_matrix_epoch_{epoch+1}.png")
    plt.close()
    
    return val_loss, val_acc, val_f1, cm

def main():
    train_dirs = {
        "text_dir": "D:/tesis/data/text_tokenized",
        "audio_dir": "D:/tesis/data/audio_preprocessed",
        "video_dir": "D:/tesis/data/video_preprocessed",
        "label_dir": "D:/tesis/data/labels"
    }
    val_dirs = {
        "text_dir": "D:/tesis/data/text_tokenized",
        "audio_dir": "D:/tesis/data/audio_preprocessed",
        "video_dir": "D:/tesis/data/video_preprocessed",
        "label_dir": "D:/tesis/data/labels"
    }
    
    train_loader = get_multimodal_dataloader(**train_dirs, sessions=["Session1", "Session2", "Session3"], batch_size=4, max_waveform_length=160000)
    val_loader = get_multimodal_dataloader(**val_dirs, sessions=["Session4"], batch_size=4, max_waveform_length=160000, shuffle=False)
    
    models = (text_model, audio_model, video_model, hierarchical_fusion, auto_attention_fusion, final_mlp)
    num_epochs = 10
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        train_loss, train_acc, train_f1 = train_epoch(train_loader, models, optimizer, criterion, device, epoch)
        val_loss, val_acc, val_f1, cm = evaluate(val_loader, models, criterion, device, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "hierarchical_state_dict": hierarchical_fusion.state_dict(),
                "auto_attention_state_dict": auto_attention_fusion.state_dict(),
                "final_mlp_state_dict": final_mlp.state_dict(),
                "epoch": epoch + 1,
                "val_f1": val_f1
            }, "D:/tesis/modelo de fusion/fusion_checkpoint_best.pth")
            print("Mejor modelo  checkpoint.")

if __name__ == "__main__":
    main() """



"""
Dataset de fusión multimodal con embeddings de Qwen pre-extraídos
Compatible con estructura de Windows
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MultimodalEmotionDatasetQwen(Dataset):
    """
    Dataset que carga:
    - Texto tokenizado (.npy)
    - Audio preprocesado (.npy)
    - Embeddings de video pre-extraídos de Qwen (.pt) <- NUEVO
    - Labels (.csv)
    """
    
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
        """Recolecta samples válidos"""
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
        
        waveform = np.array(audio_data.get("waveform"), dtype=np.float32)
        
        # Padding/truncate
        if len(waveform) > self.max_waveform_length:
            waveform = waveform[:self.max_waveform_length]
        else:
            waveform = np.pad(
                waveform, 
                (0, self.max_waveform_length - len(waveform)), 
                mode='constant'
            )
        
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        
        # ============================================================
        # CARGAR EMBEDDING DE VIDEO (QWEN) <- NUEVO
        # ============================================================
        qwen_data = torch.load(qwen_embed_file)
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
    split='train'
):
    """
    Crea DataLoader para fusión con embeddings de Qwen
    
    Args:
        split: 'train', 'val', o 'test' (para buscar en qwen_embed_dir/train o /val)
    """
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
        num_workers=0
    )


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    # Configuración para Windows
    text_dir = "D:/tesis/tesis/data/text_tokenized"
    audio_dir = "D:/tesis/tesis/data/audio_preprocessed"
    qwen_embed_dir = "D:/tesis/tesis/data/video_embeddings_qwen_session5"  # Carpeta copiada desde WSL
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
        batch_size=16,
        split='train'
    )
    
    # Test val
    print("\n[2/2] Cargando val dataloader...")
    val_loader = get_multimodal_dataloader_qwen(
        text_dir, 
        audio_dir, 
        qwen_embed_dir, 
        label_dir, 
        sessions=["Session5"],
        batch_size=16,
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