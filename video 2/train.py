import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from model_wrapper import VideoQwenWrapper
from dataset import VideoDataset, EMOTION_MAP
import numpy as np

# ====== FUNCIONES AUXILIARES ======
def compute_class_weights(dataset: VideoDataset):
    """Calcula pesos para WeightedRandomSampler usando etiquetas en .npy"""
    class_counts = torch.zeros(len(EMOTION_MAP), dtype=torch.long)
    for fp in dataset.files:
        try:
            data = np.load(fp, allow_pickle=True).item()
            emotion = data.get("emotion", None)
            label = EMOTION_MAP.get(emotion, -1)
            if label >= 0:
                class_counts[label] += 1
        except Exception:
            continue

    counts = class_counts.float() + 1e-6
    class_weights = 1.0 / counts

    sample_weights = []
    for fp in dataset.files:
        try:
            data = np.load(fp, allow_pickle=True).item()
            emotion = data.get("emotion", None)
            label = EMOTION_MAP.get(emotion, -1)
            if label >= 0:
                sample_weights.append(class_weights[label].item())
            else:
                sample_weights.append(0.0)
        except Exception:
            sample_weights.append(0.0)
    return sample_weights

def collate_fn(batch):
    frames_batch, labels_batch = [], []
    for item in batch:
        frames_batch.append(item["frames"])   # list of PIL.Image
        labels_batch.append(item["label_idx"])
    return {
        "frames": frames_batch,
        "labels": torch.stack(labels_batch)
    }

def evaluate(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()
    val_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Época {epoch+1}/{num_epochs} [Validación]", leave=False):
            frames = batch["frames"]
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(frames=frames, labels=labels)
                loss = outputs["loss"]

            val_loss += loss.item()
            preds_batch = torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy()
            preds.extend(preds_batch)
            targets.extend(labels.cpu().numpy())

    val_loss /= len(dataloader)
    # Ahora calculamos F1 macro
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    acc = accuracy_score(targets, preds)
    return f1, acc, val_loss

# ====== ENTRENAMIENTO ======
def train(
    data_root,
    batch_size,
    num_epochs,
    lr,
    weight_decay,
    sessions,
    validation_session,
    genders,
    checkpoint_dir,
    output_dir,
    accumulation_steps=1,
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    init_prompt="The emotion in the video is:"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando device:", device)

    # Split train / val
    train_sessions = [s for s in sessions if s != validation_session]
    train_dataset = VideoDataset(data_root, train_sessions, genders)
    val_dataset = VideoDataset(data_root, [validation_session], genders)

    sample_weights = compute_class_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = VideoQwenWrapper(
        model_name=model_name,
        device=device,
        init_prompt=init_prompt,
        num_classes=4
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=True
    )
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    best_f1 = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        preds, targets = [], []

        optimizer.zero_grad()

        # ======= ENTRENAMIENTO =======
        for step, batch in enumerate(tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs} [Entrenamiento]")):
            frames = batch["frames"]
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(frames=frames, labels=labels)
                loss = outputs["loss"]

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            preds_batch = torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy()
            preds.extend(preds_batch)
            targets.extend(labels.cpu().numpy())

        # ======= MÉTRICAS ENTRENAMIENTO =======
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(targets, preds)

        print(
            f"Época {epoch+1}: Pérdida entrenamiento: {train_loss:.4f} "
            f"Accuracy entrenamiento: {train_acc:.4f}"
        )

        # ======= VALIDACIÓN =======
        val_f1, val_acc, val_loss = evaluate(model, val_loader, criterion, device, epoch, num_epochs)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Pérdida promedio de validación: {val_loss:.4f} "
            f"Precisión de validación: {val_acc:.4f} "
            f"F1 Macro de validación: {val_f1:.4f}, LR: {current_lr:.6f}"
        )

        # ======= CHECKPOINT =======
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            model.model.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
            print(f"Guardado checkpoint con F1 macro de validación: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No mejora en F1. Paciencia: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping activado.")
                break

    print(f"Modelo final guardado en {checkpoint_dir}")
    return best_f1, val_acc, val_loss
