import os
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from video_model import VideoEmotionClassifier
from dataset import VideoDataset


def compute_class_weights(dataset):
    class_counts = np.zeros(4)
    for _, label, _ in dataset.data:
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for _, label, _ in dataset.data]
    return sample_weights


def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["frames"].to(device)
            y = batch["labels"].to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    val_loss /= len(dataloader)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return f1, acc, val_loss


def train(
    data_root,
    batch_size,
    num_epochs,
    lr,
    weight_decay,
    dropout,
    num_frozen_layers,
    hidden_size,
    num_layers_gru,
    sessions,
    validation_session,
    genders,
    checkpoint_dir,
    output_dir
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_sessions = [s for s in sessions if s != validation_session]
    train_dataset = VideoDataset(data_root, train_sessions, genders)
    val_dataset = VideoDataset(data_root, [validation_session], genders)

    sample_weights = compute_class_weights(train_dataset)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    print("entre")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("termine los dataloader")

    model = VideoEmotionClassifier(
        num_classes=4,
        hidden_size=hidden_size,
        num_layers=num_layers_gru,
        dropout=dropout,
        num_frozen_layers=num_frozen_layers
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    last_f1 = 0.0
    best_f1 = 0.0
    model_path = os.path.join(checkpoint_dir, "last_model.pth")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    print("pase scheduler")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        preds, targets = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = batch["frames"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(x)
            
            if torch.rand(1).item() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                batch_size = x.size(0)
                index = torch.randperm(batch_size).to(device)
                mixed_x = lam * x + (1 - lam) * x[index]
                y_a, y_b = y, y[index]
                outputs = model(mixed_x)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(x)
                loss = criterion(outputs, y)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(targets, preds, average="weighted")
        train_acc = accuracy_score(targets, preds)

        val_f1, val_acc, val_loss = evaluate(model, val_loader, criterion, device)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoca {epoch+1} | Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}, LR: {current_lr:.6f}")

        last_f1 = val_f1
        torch.save(model.state_dict(), model_path)
        print(f"Model guardado en: {model_path}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
        print(f"Nuevo mejor model guardado en: {best_model_path}")

    torch.save(model.state_dict(), os.path.join(output_dir, "last_model_final.pth"))

    return best_f1, val_acc, val_loss