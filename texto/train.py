import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from dataset import EmotionDataset
from torch.optim import AdamW
from model import get_tokenizer_and_model 
import torch.nn as nn
from tqdm import tqdm


def train(data_dir: str,validation_session: str = "Session4",batch_size: int = 4,num_epochs: int = 3,lr: float = 2e-5,checkpoint_dir: str = "texto/checkpoints",output_dir: str = "texto/fine_tuned_model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    tokenizer, model = get_tokenizer_and_model(device=device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)  

    # Datasets
    sessions = [f"Session{i}" for i in range(1, 4)]
    train_sessions = [s for s in sessions if s != validation_session]
    print(f"Train: {train_sessions}")
    print(f"Val: {validation_session}")
    train_dataset = EmotionDataset(data_dir, session_filter=train_sessions)
    val_dataset = EmotionDataset(data_dir, session_filter=[validation_session])

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, "roberta_best_checkpoint.pth")

    best_val_loss = float('inf')
    start_epoch = 0

    # Reanudar si hay checkpoint
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"reanudar epoca {start_epoch}, best val loss {best_val_loss:.4f}")

    # entrenamiento
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train = total_loss / len(train_loader)
        print(f"Avg train loss: {avg_train:.4f}")

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"Avg val loss: {avg_val:.4f}")

        #guardar la mejor
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, best_ckpt)
            print(f"Checkpoint guardado {epoch+1}, val loss {best_val_loss:.4f}")
