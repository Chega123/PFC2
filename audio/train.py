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
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    # Dataset setup
    train_dataset = AudioDataset(root_dir=root_dir, include_sessions=train_sessions)
    test_dataset = AudioDataset(root_dir=root_dir, include_sessions=test_sessions)

    # Calculate class weights for WeightedRandomSampler
    num_classes = 4  # neutral, happy/excited, sad, angry
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]  # Assuming dataset returns (data, label)
    label_counts = {}
    for label in range(num_classes):
        label_counts[label] = train_labels.count(label) or 1  # Avoid division by zero
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [class_weights[label.item()] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # DataLoader setup
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
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)  # Changed to track F1
        print(f"Modelo cargado desde epoca {start_epoch}")
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        start_epoch = 0
        best_val_f1 = 0.0  # Initialize to 0 for F1 score

    # Scheduler: CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Initial period (number of epochs before first restart)
        T_mult=2,  # Factor to increase period after each restart
        eta_min=1e-7  # Minimum learning rate
    )

    criterion = CrossEntropyLoss()

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Update scheduler after each epoch
        scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoca {epoch+1} - Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Validating"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss = criterion(logits, y)
                total_val_loss += val_loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_dataloader)
        print(f"Epoca {epoch+1} - Validation Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        report = classification_report(
            all_labels,
            all_preds,
            target_names=["neutral", "happy/excited", "sad", "angry"],
            output_dict=True
        )
        print("Reporte de clasificación:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=["neutral", "happy/excited", "sad", "angry"]
        ))

        val_f1 = report["weighted avg"]["f1-score"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1  # Save best F1 score
            }, checkpoint_path)
            print(f"Checkpoint guardado en {checkpoint_path} (mejor F1: {best_val_f1:.4f})")
        else:
            print(f"Checkpoint no actualizado, best_val_f1 sigue siendo {best_val_f1:.4f}")

    val_accuracy = report["accuracy"]
    val_f1 = report["weighted avg"]["f1-score"]

    return avg_val_loss, val_accuracy, val_f1