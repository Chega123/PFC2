
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
g = torch.Generator()
g.manual_seed(42)
sys.path.append('D:/tesis/tesis/audio')
sys.path.append('D:/tesis/tesis/texto')
sys.path.append('D:/tesis/tesis/modelo de fusion')

from audio_model import Wav2VecEmotionClassifier
from texto_model import get_tokenizer_and_model
from hierarchical import HierarchicalFusionModule
from auto_attention import AutoAttentionFusionModule
from final_fusion import FinalFusionMLP
from Fusiondataset import get_multimodal_dataloader_qwen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

emotion_to_idx = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "excited": 1}
idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}


def print_emotion_metrics(all_labels, all_preds, phase="Val", epoch=None):

    # Calcular métricas por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2, 3], zero_division=0
    )
    
    # Header
    if epoch is not None:
        print(f"\n{'='*80}")
        print(f"  {phase} Metrics - Epoch {epoch}")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"  {phase} Metrics")
        print(f"{'='*80}")
    
    # Tabla de métricas por emoción
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print(f"{'-'*80}")
    
    for i in [0, 1, 2, 3]:
        emotion_name = idx_to_emotion.get(i, f"Class{i}")
        print(f"{emotion_name:<12} {precision[i]:>11.4f} {recall[i]:>11.4f} {f1[i]:>11.4f} {int(support[i]):>11d}")
    
    # Métricas globales
    print(f"{'-'*80}")
    overall_acc = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"{'Overall':<12} {'Accuracy':<12} {'Weighted F1':<12} {'Macro F1':<12}")
    print(f"{'':<12} {overall_acc:>11.4f} {overall_f1:>11.4f} {macro_f1:>11.4f}")
    print(f"{'='*80}\n")
    
    return precision, recall, f1, support


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.002, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:
            if score > self.best_score - self.min_delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
                
        return self.early_stop



print("\nCargando modelos de texto y audio...")

# Texto (RoBERTa)
text_tokenizer, text_model = get_tokenizer_and_model(
    model_name="roberta-base", 
    return_classifier=False, 
    device=device
)
checkpoint = torch.load("D:/tesis/tesis/texto/best_models/roberta_best_f1_checkpoint_Session2.pth", map_location=device)
text_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
text_model.eval()
for param in text_model.parameters():
    param.requires_grad = False
print("Modelo de texto cargado y congelado")

# Audio (Wav2Vec2)
audio_model = Wav2VecEmotionClassifier(
    pretrained_model="facebook/wav2vec2-base", 
    num_classes=4, 
    dropout=0.3, 
    num_frozen_layers=0
).to(device)
checkpoint = torch.load("D:/tesis/tesis/audio/best_models/best_model_Session2.pth", map_location=device)
audio_model.load_state_dict(checkpoint["model_state_dict"])
audio_model.eval()
for param in audio_model.parameters():
    param.requires_grad = False
print(" Modelo de audio cargado y congelado")

# Módulos de fusión con DROPOUT AGRESIVO
print("\n🔧 Inicializando módulos de fusión...")
hierarchical_fusion = HierarchicalFusionModule(embed_dim=768, dropout=0.6).to(device)
auto_attention_fusion = AutoAttentionFusionModule(embed_dim=768, dropout=0.6).to(device)
final_mlp = FinalFusionMLP(embed_dim=768, dropout=0.6).to(device)

# Intentar cargar checkpoint previo (opcional)
fusion_checkpoint_path = "D:/tesis/tesis/modelo_de_fusion/fusion_checkpoint_qwen_best.pth"
if os.path.exists(fusion_checkpoint_path):
    print(f" Cargando checkpoint previo: {fusion_checkpoint_path}")
    checkpoint = torch.load(fusion_checkpoint_path, map_location=device)
    hierarchical_fusion.load_state_dict(checkpoint["hierarchical_state_dict"])
    auto_attention_fusion.load_state_dict(checkpoint["auto_attention_state_dict"])
    final_mlp.load_state_dict(checkpoint["final_mlp_state_dict"])
    print("Checkpoint de fusión cargado")
else:
    print("Iniciando desde cero (sin checkpoint previo)")

print("Módulos de fusión listos")

# Optimizador con LR MUY BAJO y weight decay ALTO
optimizer = torch.optim.AdamW(
    list(hierarchical_fusion.parameters()) + 
    list(auto_attention_fusion.parameters()) + 
    list(final_mlp.parameters()),
    lr=1e-5,  # MUY BAJO - aprendizaje lento
    weight_decay=2e-4,  # ALTO - más regularización
    betas=(0.9, 0.999),
    eps=1e-8
)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='max',
    factor=0.5, 
    patience=2, 
    verbose=True,
    min_lr=1e-7
)

# Loss con label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()


def train_epoch(dataloader, models, optimizer, criterion, device, epoch, accumulation_steps=4):
    text_model, audio_model, hierarchical_fusion, auto_attention_fusion, final_mlp = models
    
    text_model.eval()
    audio_model.eval()
    hierarchical_fusion.train()
    auto_attention_fusion.train()
    final_mlp.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        waveform = batch["waveform"].to(device)
        video_emb = batch["video_embedding"].to(device)
        labels = batch["label"].to(device)
        
        with autocast():
            # Extraer features sin gradientes
            with torch.no_grad():
                text_emb = text_model.extract_features(input_ids, attention_mask)
                audio_emb = audio_model.embedding.extract_features(waveform)
            
            # Fusión con gradientes
            v1 = hierarchical_fusion(text_emb, audio_emb, video_emb)
            v2 = auto_attention_fusion(text_emb, audio_emb, video_emb)
            logits = final_mlp(v1, v2)
            
            # Normalizar loss por accumulation steps
            loss = criterion(logits, labels) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        # Actualizar cada accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(hierarchical_fusion.parameters()) + 
                list(auto_attention_fusion.parameters()) + 
                list(final_mlp.parameters()), 
                max_norm=0.5  # Más restrictivo
            )
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy())
        
        torch.cuda.empty_cache()
    
    # Último step si no es múltiplo exacto
    if len(dataloader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(hierarchical_fusion.parameters()) + 
            list(auto_attention_fusion.parameters()) + 
            list(final_mlp.parameters()), 
            max_norm=0.5
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    # Imprimir métricas por emoción
    print_emotion_metrics(all_labels, all_preds, phase="Train", epoch=epoch+1)
    
    return epoch_loss, epoch_acc, epoch_f1


def evaluate(dataloader, models, criterion, device, epoch):
    text_model, audio_model, hierarchical_fusion, auto_attention_fusion, final_mlp = models
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
            video_emb = batch["video_embedding"].to(device)
            labels = batch["label"].to(device)
            filenames = batch["filename"]
            
            with autocast():
                text_emb = text_model.extract_features(input_ids, attention_mask)
                audio_emb = audio_model.embedding.extract_features(waveform)
                
                v1 = hierarchical_fusion(text_emb, audio_emb, video_emb)
                v2 = auto_attention_fusion(text_emb, audio_emb, video_emb)
                logits = final_mlp(v1, v2)
                loss = criterion(logits, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy())
            
            if first_batch:
                print("\nPrimeros samples de validación:")
                for filename, label, pred in zip(filenames[:5], labels.cpu().numpy()[:5], preds[:5]):
                    correct = "✓" if label == pred else "✗"
                    print(f"  {correct} {filename}: True {label} ({idx_to_emotion.get(label, 'unknown')}), "
                          f"Pred {pred} ({idx_to_emotion.get(pred, 'unknown')})")
                first_batch = False
            
            torch.cuda.empty_cache()
    
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    # Imprimir métricas por emoción
    print_emotion_metrics(all_labels, all_preds, phase="Val", epoch=epoch+1)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[idx_to_emotion[i] for i in [0, 1, 2, 3]],
                yticklabels=[idx_to_emotion[i] for i in [0, 1, 2, 3]])
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"D:/tesis/tesis/modelo_de_fusion/confusion_matrix_qwen_epoch_{epoch+1}.png")
    plt.close()
    
    return val_loss, val_acc, val_f1, cm


# ============================================================
# MAIN
# ============================================================
def main():

    # Configuración
    print("Cargando datasets...")
    data_dirs = {
        "text_dir": "D:/tesis/tesis/data/text_tokenized",
        "audio_dir": "D:/tesis/tesis/data/audio_preprocessed",
        "qwen_embed_dir": "D:/tesis/tesis/data/video_embeddings_qwen_session2",
        "label_dir": "D:/tesis/tesis/data/labels"
    }
    

    train_loader = get_multimodal_dataloader_qwen(
        **data_dirs, 
        sessions=["Session1", "Session3", "Session4", "Session5"], 
        batch_size=32,  # Efectivo: 8*4=32
        max_waveform_length=160000,
        split='train',
        generator=g
    )
    
    val_loader = get_multimodal_dataloader_qwen(
        **data_dirs, 
        sessions=["Session2"], 
        batch_size=32, 
        max_waveform_length=160000, 
        shuffle=False,
        split='val',
        generator=g
    )
    
    print(" Datasets cargados")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}\n")
    
    # Modelos
    models = (text_model, audio_model, hierarchical_fusion, auto_attention_fusion, final_mlp)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=4, min_delta=0.001, mode='max')
    
    # Entrenamiento
    print(" Comenzando entrenamiento")
    
    num_epochs = 30
    best_f1 = 0.0
    accumulation_steps = 4
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        
        train_loss, train_acc, train_f1 = train_epoch(
            train_loader, models, optimizer, criterion, device, epoch, accumulation_steps
        )
        val_loss, val_acc, val_f1, cm = evaluate(val_loader, models, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        

        print(f"Epoch {epoch+1}/{num_epochs} Summary | LR: {current_lr:.2e}")
        print(f"{'─'*60}")
        print(f"Train → Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   → Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"Gap   → Acc: {train_acc-val_acc:+.4f}, F1: {train_f1-val_f1:+.4f}")

        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "hierarchical_state_dict": hierarchical_fusion.state_dict(),
                "auto_attention_state_dict": auto_attention_fusion.state_dict(),
                "final_mlp_state_dict": final_mlp.state_dict(),
                "epoch": epoch + 1,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "train_f1": train_f1,
                "train_acc": train_acc
            }, "D:/tesis/tesis/modelo_de_fusion/fusion_checkpoint_qwen_best.pth")
            print("💾 Mejor modelo guardado")
        
        # Guardar checkpoint regular
        torch.save({
            "hierarchical_state_dict": hierarchical_fusion.state_dict(),
            "auto_attention_state_dict": auto_attention_fusion.state_dict(),
            "final_mlp_state_dict": final_mlp.state_dict(),
            "epoch": epoch + 1,
            "val_f1": val_f1
        }, "D:/tesis/tesis/modelo_de_fusion/fusion_checkpoint_qwen.pth")
        
        # Check early stopping
        if early_stopping(val_f1):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"No mejora en {early_stopping.patience} épocas")
            break

    print(f" Entrenamiento Completado")
    print(f" Mejor Val F1: {best_f1:.4f}")
    print(f" Modelo guardado en: fusion_checkpoint_qwen_best.pth")


if __name__ == "__main__":
    main() 


""" import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)

sys.path.append('D:/tesis/tesis/audio')
sys.path.append('D:/tesis/tesis/texto')
sys.path.append('D:/tesis/tesis/modelo de fusion')

from audio_model import Wav2VecEmotionClassifier
from texto_model import get_tokenizer_and_model
from hierarchical import HierarchicalFusionModule
from auto_attention import AutoAttentionFusionModule
from final_fusion import FinalFusionMLP
from Fusiondataset import get_bimodal_dataloader  # ✅ Dataset BIMODAL (sin video)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

emotion_to_idx = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "excited": 1}
idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}


def print_emotion_metrics(all_labels, all_preds, phase="Val", epoch=None):

    # Calcular métricas por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2, 3], zero_division=0
    )
    
    # Header
    if epoch is not None:
        print(f"\n{'='*80}")
        print(f"  {phase} Metrics - Epoch {epoch}")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"  {phase} Metrics")
        print(f"{'='*80}")
    
    # Tabla de métricas por emoción
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print(f"{'-'*80}")
    
    for i in [0, 1, 2, 3]:
        emotion_name = idx_to_emotion.get(i, f"Class{i}")
        print(f"{emotion_name:<12} {precision[i]:>11.4f} {recall[i]:>11.4f} {f1[i]:>11.4f} {int(support[i]):>11d}")
    
    # Métricas globales
    print(f"{'-'*80}")
    overall_acc = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"{'Overall':<12} {'Accuracy':<12} {'Weighted F1':<12} {'Macro F1':<12}")
    print(f"{'':<12} {overall_acc:>11.4f} {overall_f1:>11.4f} {macro_f1:>11.4f}")
    print(f"{'='*80}\n")
    
    return precision, recall, f1, support


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.002, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:
            if score > self.best_score - self.min_delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
                
        return self.early_stop


print("\nCargando modelos de texto y audio...")

# Texto (RoBERTa)
text_tokenizer, text_model = get_tokenizer_and_model(
    model_name="roberta-base", 
    return_classifier=False, 
    device=device
)
checkpoint = torch.load("D:/tesis/tesis/texto/best_models/roberta_best_f1_checkpoint_Session4.pth", map_location=device)
text_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
text_model.eval()
for param in text_model.parameters():
    param.requires_grad = False
print("✅ Modelo de texto cargado y congelado")

# Audio (Wav2Vec2)
audio_model = Wav2VecEmotionClassifier(
    pretrained_model="facebook/wav2vec2-base", 
    num_classes=4, 
    dropout=0.3, 
    num_frozen_layers=0
).to(device)
checkpoint = torch.load("D:/tesis/tesis/audio/Best_models/best_model_Session4.pth", map_location=device)
audio_model.load_state_dict(checkpoint["model_state_dict"])
audio_model.eval()
for param in audio_model.parameters():
    param.requires_grad = False
print("✅ Modelo de audio cargado y congelado")

# Módulos de fusión BIMODAL (sin video)
print("\n🔧 Inicializando módulos de fusión BIMODAL (solo texto + audio)...")
hierarchical_fusion = HierarchicalFusionModule(embed_dim=768, dropout=0.55).to(device)
auto_attention_fusion = AutoAttentionFusionModule(embed_dim=768, dropout=0.55).to(device)
final_mlp = FinalFusionMLP(embed_dim=768, dropout=0.6).to(device)

# Intentar cargar checkpoint previo (opcional)
fusion_checkpoint_path = "D:/tesis/tesis/modelo de fusion/fusion_checkpoint_bimodal.pth"
if os.path.exists(fusion_checkpoint_path):
    print(f"📂 Cargando checkpoint previo: {fusion_checkpoint_path}")
    checkpoint = torch.load(fusion_checkpoint_path, map_location=device)
    hierarchical_fusion.load_state_dict(checkpoint["hierarchical_state_dict"])
    auto_attention_fusion.load_state_dict(checkpoint["auto_attention_state_dict"])
    final_mlp.load_state_dict(checkpoint["final_mlp_state_dict"])
    print("✅ Checkpoint de fusión cargado")
else:
    print("🆕 Iniciando desde cero (sin checkpoint previo)")

print("✅ Módulos de fusión listos")

# Optimizador con LR MUY BAJO y weight decay ALTO
optimizer = torch.optim.AdamW(
    list(hierarchical_fusion.parameters()) + 
    list(auto_attention_fusion.parameters()) + 
    list(final_mlp.parameters()),
    lr=1e-5,  # MUY BAJO - aprendizaje lento
    weight_decay=2e-4,  # ALTO - más regularización
    betas=(0.9, 0.999),
    eps=1e-8
)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='max',
    factor=0.5, 
    patience=2, 
    verbose=True,
    min_lr=1e-7
)

# Loss con label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler()


def train_epoch(dataloader, models, optimizer, criterion, device, epoch, accumulation_steps=4):
    text_model, audio_model, hierarchical_fusion, auto_attention_fusion, final_mlp = models
    
    text_model.eval()
    audio_model.eval()
    hierarchical_fusion.train()
    auto_attention_fusion.train()
    final_mlp.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)
        
        with autocast():
            # Extraer features sin gradientes
            with torch.no_grad():
                text_emb = text_model.extract_features(input_ids, attention_mask)
                audio_emb = audio_model.embedding.extract_features(waveform)
            
            # Fusión BIMODAL (solo texto + audio)
            v1 = hierarchical_fusion(text_emb, audio_emb)
            v2 = auto_attention_fusion(text_emb, audio_emb)
            logits = final_mlp(v1, v2)
            
            # Normalizar loss por accumulation steps
            loss = criterion(logits, labels) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        # Actualizar cada accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(hierarchical_fusion.parameters()) + 
                list(auto_attention_fusion.parameters()) + 
                list(final_mlp.parameters()), 
                max_norm=0.5
            )
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy())
        
        torch.cuda.empty_cache()
    
    # Último step si no es múltiplo exacto
    if len(dataloader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(hierarchical_fusion.parameters()) + 
            list(auto_attention_fusion.parameters()) + 
            list(final_mlp.parameters()), 
            max_norm=0.5
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    # Imprimir métricas por emoción
    print_emotion_metrics(all_labels, all_preds, phase="Train", epoch=epoch+1)
    
    return epoch_loss, epoch_acc, epoch_f1


def evaluate(dataloader, models, criterion, device, epoch):
    text_model, audio_model, hierarchical_fusion, auto_attention_fusion, final_mlp = models
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
            labels = batch["label"].to(device)
            filenames = batch["filename"]
            
            with autocast():
                text_emb = text_model.extract_features(input_ids, attention_mask)
                audio_emb = audio_model.embedding.extract_features(waveform)
                
                # Fusión BIMODAL (solo texto + audio)
                v1 = hierarchical_fusion(text_emb, audio_emb)
                v2 = auto_attention_fusion(text_emb, audio_emb)
                logits = final_mlp(v1, v2)
                loss = criterion(logits, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy())
            
            if first_batch:
                print("\nPrimeros samples de validación:")
                for filename, label, pred in zip(filenames[:5], labels.cpu().numpy()[:5], preds[:5]):
                    correct = "✓" if label == pred else "✗"
                    print(f"  {correct} {filename}: True {label} ({idx_to_emotion.get(label, 'unknown')}), "
                          f"Pred {pred} ({idx_to_emotion.get(pred, 'unknown')})")
                first_batch = False
            
            torch.cuda.empty_cache()
    
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    # Imprimir métricas por emoción
    print_emotion_metrics(all_labels, all_preds, phase="Val", epoch=epoch+1)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[idx_to_emotion[i] for i in [0, 1, 2, 3]],
                yticklabels=[idx_to_emotion[i] for i in [0, 1, 2, 3]])
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"D:/tesis/tesis/modelo de fusion/confusion_matrix_bimodal_epoch_{epoch+1}.png")
    plt.close()
    
    return val_loss, val_acc, val_f1, cm


# ============================================================
# MAIN
# ============================================================
def main():

    # ============================================================
    # K-FOLD VALIDATION - Cambiar aquí qué sesión usar como validación
    # ============================================================
    VAL_SESSION = "Session4"  # 🔄 Cambiar esto para K-Fold (Session1-5)
    ALL_SESSIONS = ["Session1", "Session2", "Session3", "Session4", "Session5"]
    TRAIN_SESSIONS = [s for s in ALL_SESSIONS if s != VAL_SESSION]
    
    print(f"\n📊 K-Fold Configuration (BIMODAL - Sin Video):")
    print(f"  Validación: {VAL_SESSION}")
    print(f"  Train: {TRAIN_SESSIONS}\n")
    
    # Configuración de directorios (SIN video)
    print("Cargando datasets bimodales...")
    
    train_loader = get_bimodal_dataloader(
        text_dir="D:/tesis/tesis/data/text_tokenized",
        audio_dir="D:/tesis/tesis/data/audio_preprocessed",
        label_dir="D:/tesis/tesis/data/labels",
        sessions=TRAIN_SESSIONS,
        batch_size=10,  # Efectivo: 10*4=40
        max_waveform_length=160000,
        shuffle=True
    )
    
    val_loader = get_bimodal_dataloader(
        text_dir="D:/tesis/tesis/data/text_tokenized",
        audio_dir="D:/tesis/tesis/data/audio_preprocessed",
        label_dir="D:/tesis/tesis/data/labels",
        sessions=[VAL_SESSION],
        batch_size=10,
        max_waveform_length=160000,
        shuffle=False
    )
    
    print("✅ Datasets cargados")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}\n")
    
    # Modelos
    models = (text_model, audio_model, hierarchical_fusion, auto_attention_fusion, final_mlp)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=4, min_delta=0.001, mode='max')
    
    # Entrenamiento
    print("🚀 Comenzando entrenamiento (SOLO TEXTO + AUDIO)")
    
    num_epochs = 20
    best_f1 = 0.0
    accumulation_steps = 4
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        
        train_loss, train_acc, train_f1 = train_epoch(
            train_loader, models, optimizer, criterion, device, epoch, accumulation_steps
        )
        val_loss, val_acc, val_f1, cm = evaluate(val_loader, models, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        

        print(f"Epoch {epoch+1}/{num_epochs} Summary | LR: {current_lr:.2e}")
        print(f"{'─'*60}")
        print(f"Train → Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   → Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"Gap   → Acc: {train_acc-val_acc:+.4f}, F1: {train_f1-val_f1:+.4f}")

        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "hierarchical_state_dict": hierarchical_fusion.state_dict(),
                "auto_attention_state_dict": auto_attention_fusion.state_dict(),
                "final_mlp_state_dict": final_mlp.state_dict(),
                "epoch": epoch + 1,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "train_f1": train_f1,
                "train_acc": train_acc,
                "val_session": VAL_SESSION
            }, f"D:/tesis/tesis/modelo de fusion/fusion_checkpoint_bimodal_best_{VAL_SESSION}.pth")
            print("💾 Mejor modelo guardado")
        
        # Guardar checkpoint regular
        torch.save({
            "hierarchical_state_dict": hierarchical_fusion.state_dict(),
            "auto_attention_state_dict": auto_attention_fusion.state_dict(),
            "final_mlp_state_dict": final_mlp.state_dict(),
            "epoch": epoch + 1,
            "val_f1": val_f1,
            "val_session": VAL_SESSION
        }, "D:/tesis/tesis/modelo de fusion/fusion_checkpoint_bimodal.pth")
        
        # Check early stopping
        if early_stopping(val_f1):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"No mejora en {early_stopping.patience} épocas")
            break

    print(f"🎉 Entrenamiento Completado")
    print(f"📊 Mejor Val F1: {best_f1:.4f}")
    print(f"💾 Modelo guardado en: fusion_checkpoint_bimodal_best_{VAL_SESSION}.pth")


if __name__ == "__main__":
    main() """