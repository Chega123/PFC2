import os
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dataset import EmotionDataset
from torch.optim import AdamW
import torch.nn.functional as F
from texto_model import get_tokenizer_and_model 
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")

def print_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total}")
    print(f"Parámetros entrenables: {trainable}")
    print(f"Porcentaje entrenable: {100 * trainable / total:.2f}%")


def train(
    data_dir: str,
    validation_session: str = "Session5",
    batch_size: int =  64,
    num_epochs: int = 15,
    lr: float = 2.7676263131199924e-05,
    weight_decay: float = 0.01,
    dropout: float = 0.15,
    num_frozen_layers: int = 0,
    checkpoint_dir: str = "texto/checkpoints",
    output_dir: str = "texto/fine_tuned_model",
    warmup_ratio: float = 0.1,  
    grad_clip: float = 1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    torch.backends.cudnn.benchmark = True  
    torch.backends.cuda.matmul.allow_tf32 = True  #usar tensor cores
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer, model = get_tokenizer_and_model(device=device, dropout=dropout, num_frozen_layers=num_frozen_layers)
    print_trainable_parameters(model)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Datasets
    sessions = [f"Session{i}" for i in range(1, 6)]
    train_sessions = [s for s in sessions if s != validation_session]
    print(f"Sesiones de entrenamiento: {train_sessions}")
    print(f"Sesión de validación: {validation_session}")

    train_dataset = EmotionDataset(data_dir, session_filter=train_sessions)
    val_dataset = EmotionDataset(data_dir, session_filter=[validation_session])

    print(f"Tamaño del dataset de entrenamiento: {len(train_dataset)}")
    print(f"Tamaño del dataset de validación: {len(val_dataset)}")

    sample = train_dataset[0]
    print("Ejemplo de train_dataset[0]:", sample)
    print("Tipo de dato devuelto:", type(sample))


    # Obtener solo las etiquetas de entrenamiento
    train_labels = [sample["labels"].item() for sample in train_dataset]

    # Y lo mismo para validación
    val_labels = [sample["labels"].item() for sample in val_dataset]
    print("Primeros labels del train_dataset:", train_labels[:10])
    num_classes = len(set(train_labels))
    print(f"Número de clases detectadas: {num_classes}")

    # Calcular pesos para CrossEntropyLoss
    label_counts = Counter(train_labels)
    total_count = sum(label_counts.values())

    class_weights = torch.tensor(
        [total_count / label_counts.get(i, 1) for i in range(num_classes)], 
        dtype=torch.float,
        device=device
    )
    class_weights = class_weights / class_weights.sum()

    print("Pesos para CrossEntropyLoss:", class_weights)

    # DataLoader normal, sin sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),  # Usa el parámetro warmup_ratio
        num_training_steps=total_steps
    )
    # Crear directorio para checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, "roberta_best_f1_checkpoint.pth")
    
    best_val_f1_macro = 0.0  # Cambiado a F1 macro para guardar el mejor modelo
    start_epoch = 0

    # Entrenamiento
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        all_train_losses = []
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs} [Entrenamiento]")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                per_sample_loss = loss_fn(logits, labels)
                loss = per_sample_loss.mean()

            all_train_losses.extend(per_sample_loss.detach().cpu().numpy())
            preds = torch.argmax(logits, dim=1)
            all_train_preds.extend(preds.detach().cpu().numpy())
            all_train_labels.extend(labels.detach().cpu().numpy())
            if not torch.isfinite(loss):
                print(f"Advertencia: Pérdida no finita (loss={loss.item()})")
                continue

            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip) 
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        avg_train = total_loss / len(train_loader)
        print(f"Época {epoch+1}:")
        print(f"  Pérdida entrenamiento: {avg_train:.4f}")
        print(f"  Accuracy entrenamiento: {train_accuracy:.4f}")

        # Validación
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []  

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Época {epoch+1}/{num_epochs} [Validación]")):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                logits = outputs.logits
                # Calcular probabilidades
                probs = F.softmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())  # Guardar probabilidades

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val = val_loss / len(val_loader)
        print(f"Pérdida promedio de validación: {avg_val:.4f}")

        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        current_lr = scheduler.get_last_lr()[0]
        print(f"Precisión de validación: {val_accuracy:.4f}")
        print(f"F1 Macro de validación: {val_f1_macro:.4f}, LR: {current_lr:.6f}")

        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1_macro": best_val_f1_macro,
                "best_val_loss": avg_val
            }, best_ckpt)
            print(f"Guardado checkpoint con F1 macro de validación: {best_val_f1_macro:.4f}")

    # Guardar modelo final
    os.makedirs(output_dir, exist_ok=True)
    final_model_path = os.path.join(output_dir, "roberta_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Modelo final guardado en: {final_model_path}")

    return avg_val, val_accuracy, val_f1_macro