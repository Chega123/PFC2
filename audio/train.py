import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from load_dataset import AudioDataset
from collate import collate_fn


def train(
    model,
    root_dir,
    train_sessions,
    test_sessions,
    batch_size,
    device,
    epochs=10,
    lr=1e-5,
    weight_decay=1e-2,
    checkpoint_path="wav2vec_emotion_model.pth",
    resume_training=True,
    collate_fn=collate_fn
):

    train_dataset = AudioDataset(root_dir=root_dir, include_sessions=train_sessions)
    test_dataset = AudioDataset(root_dir=root_dir, include_sessions=test_sessions)

    num_classes = 4  # neutral, happy/excited, sad, angry
    train_labels = []
    for npy_file in train_dataset.files:
        data = np.load(npy_file, allow_pickle=True).item()
        label = train_dataset.emotion_to_id(data["emotion"])
        train_labels.append(label)

    label_counts = {}
    for label in range(num_classes):
        label_counts[label] = train_labels.count(label) or 1  # evitar división por cero

    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model.to(device)


    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        print(f"Modelo cargado desde epoca {start_epoch}")
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        start_epoch = 0
        best_val_f1 = 0.0

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,       
        T_mult=2,    
        eta_min=1e-7 
    )

    criterion = CrossEntropyLoss()
    scaler = GradScaler()  

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []

        for batch in tqdm(train_dataloader, desc=f"Epoca {epoch+1}/{epochs} [Entrenamiento]"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast():
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_train_preds.extend(preds.detach().cpu().numpy())
            all_train_labels.extend(y.detach().cpu().numpy())

        scheduler.step()
        avg_loss = total_loss / len(train_dataloader)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        print(f"Época {epoch+1}:")
        print(f"  Pérdida entrenamiento: {avg_loss:.4f}")
        print(f"  Accuracy entrenamiento: {train_accuracy:.4f}")

        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Época {epoch+1}/{epochs} [Validación]"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                with autocast():
                    logits = model(x)
                    val_loss = criterion(logits, y)

                total_val_loss += val_loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_dataloader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')

        current_lr = scheduler.get_last_lr()[0]
        print(f"Pérdida promedio de validación: {avg_val_loss:.4f}")
        print(f"Precisión de validación: {val_accuracy:.4f}")
        print(f"F1 Macro de validación: {val_f1_macro:.4f}, LR: {current_lr:.6f}")

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1
            }, checkpoint_path)
            print(f"Guardado checkpoint con F1 macro de validación: {best_val_f1:.4f}")

        else:
            print(f"Checkpoint no actualizado, mejor F1 macro sigue siendo {best_val_f1:.4f}")

    return avg_val_loss, val_accuracy, val_f1_macro
