import os
import torch
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

def print_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total}")
    print(f"Parámetros entrenables: {trainable}")
    print(f"Porcentaje entrenable: {100 * trainable / total:.2f}%")

def train(
    data_dir: str,
    validation_session: str = "Session4",
    batch_size: int = 4,
    num_epochs: int = 2,
    lr: float = 2e-5,
    weight_decay: float = 1e-2,
    dropout: float = 0.1,
    num_frozen_layers: int = 0,
    checkpoint_dir: str = "texto/checkpoints",
    output_dir: str = "texto/fine_tuned_model"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    tokenizer, model = get_tokenizer_and_model(device=device, dropout=dropout, num_frozen_layers=num_frozen_layers)
    print_trainable_parameters(model)

    encoder_layers = list(model.roberta.encoder.layer)
    group1 = encoder_layers[:4]   # Capas más profundas
    group2 = encoder_layers[4:8]  # Capas intermedias
    group3 = encoder_layers[8:]   # Capas más cercanas a la salida

    optimizer = AdamW([
        {'params': [p for l in group1 for p in l.parameters() if p.requires_grad], 'lr': lr * 0.25},
        {'params': [p for l in group2 for p in l.parameters() if p.requires_grad], 'lr': lr * 0.5},
        {'params': [p for l in group3 for p in l.parameters() if p.requires_grad], 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=weight_decay)

    # Nuevo scheduler: CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2,  # Período inicial (número de épocas antes del primer reinicio)
        T_mult=2,  # Factor para aumentar el período tras cada reinicio
        eta_min=1e-6  # Tasa de aprendizaje mínima
    )

    # Datasets
    sessions = [f"Session{i}" for i in range(1, 4)]
    train_sessions = [s for s in sessions if s != validation_session]
    print(f"Sesiones de entrenamiento: {train_sessions}")
    print(f"Sesión de validación: {validation_session}")
    
    train_dataset = EmotionDataset(data_dir, session_filter=train_sessions)
    val_dataset = EmotionDataset(data_dir, session_filter=[validation_session])

    print(f"Tamaño del dataset de entrenamiento: {len(train_dataset)}")
    print(f"Tamaño del dataset de validación: {len(val_dataset)}")

    # Calcular distribución de etiquetas para WeightedRandomSampler
    train_labels = [train_dataset[i]["labels"].item() for i in range(len(train_dataset))]
    label_counts = Counter(train_labels)
    print("Distribución de etiquetas en entrenamiento:", label_counts)

    # Calcular pesos inversos para cada muestra
    num_classes = 4  # neutral, happy, sad, angry
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # DataLoaders con WeightedRandomSampler para entrenamiento
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Crear directorio para checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, "roberta_best_f1_checkpoint.pth")
    
    best_val_f1_macro = 0.0  # Cambiado a F1 macro para guardar el mejor modelo
    start_epoch = 0
    
    # Cargar checkpoint si existe
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1_macro = ckpt.get("best_val_f1_macro", 0.0)
        print(f"Reanudando desde época {start_epoch}, mejor F1 macro: {best_val_f1_macro:.4f}")
    
    # Entrenamiento
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        all_train_losses = []
        for batch in tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs} [Entrenamiento]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            per_sample_loss = loss_fn(logits, labels)
            loss = per_sample_loss.mean()

            all_train_losses.extend(per_sample_loss.detach().cpu().numpy())

            if not torch.isfinite(loss):
                print(f"Advertencia: Pérdida no finita (loss={loss.item()})")
                continue

            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Actualizar el scheduler después de cada época
        scheduler.step()

        avg_train = total_loss / len(train_loader)
        print(f"Pérdida promedio de entrenamiento: {avg_train:.4f}")

        # Validación
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []  # Lista para almacenar las probabilidades

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