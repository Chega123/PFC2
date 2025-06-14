import torch
from torch.utils.data import DataLoader
from audio.audio_model import Wav2VecEmotion
from load_dataset import AudioDataset
from train import train
from collate import collate_fn
import optuna
import time
import os

def objective(trial):
    # Hiperparámetros sugeridos por Optuna
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
    dropout = trial.suggest_categorical("dropout", [0.1, 0.3, 0.5])
    batch_size = trial.suggest_categorical("batch_size", [4, 8])

    # Modelo
    model = Wav2VecEmotion(pretrained_model="facebook/wav2vec2-base", num_classes=4, dropout=dropout)

    # Dataset y DataLoader
    train_sessions = ["Session1", "Session2", "Session3"]
    test_sessions = ["Session4"]

    train_dataset = AudioDataset(root_dir="data/audio_preprocessed", include_sessions=train_sessions)
    test_dataset = AudioDataset(root_dir="data/audio_preprocessed", include_sessions=test_sessions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Entrenamiento
    checkpoint_path = f"audio/checkpoint_audio/trial_{trial.number}.pth"
    val_loss, _ = train(model, train_loader, test_loader, device,
                        epochs=10,
                        lr=lr,
                        weight_decay=weight_decay,
                        checkpoint_path=checkpoint_path,
                        resume_training=False)

    return val_loss  # Queremos minimizar el validation loss

if __name__ == "__main__":
    start_time = time.time()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Cambia esto si quieres más/menos pruebas

    best_trial = study.best_trial
    print("✅ Mejor combinación encontrada:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"Mejor validation loss: {best_trial.value:.4f}")

    print("\nModelo guardado como audio/checkpoint_audio/trial_<id>.pth")

    elapsed = time.time() - start_time
    print(f"\n⏱️ Tiempo total: {elapsed:.2f} segundos.")
