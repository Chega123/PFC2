import sys
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
    main()