import torch
import os
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from sklearn.metrics import classification_report

def train(model, train_dataloader, test_dataloader, device, epochs=10, lr=1e-5, checkpoint_path="wav2vec_emotion_model.pth", resume_training=True):
    model.to(device)

    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Modelo cargado desde epoch {start_epoch}")
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2) 
        start_epoch = 0
        best_val_loss = float('inf')

    # scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-7)

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

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

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
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=["neutral", "happy/excited", "sad", "angry"]))

        # Actualizar scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)  # Actualizar scheduler
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"Lr reducido de {current_lr:.2e} a {new_lr:.2e}")


        # Salvar checkpoint si es mejor
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"Checkpoint guardado en {checkpoint_path} (mejor validación)")
        else:
            print(f"Checkpoint no actualizado, best_val_loss sigue siendo {best_val_loss:.4f}")