# optuna_audio.py
import torch
from audio_model import Wav2VecEmotionClassifier
import optuna
import time
import os
from train import train
from collate import collate_fn

# Elegir la métrica a optimizar
metric_to_optimize = "f1"  # Opciones: "val_loss", "accuracy", "f1"

def objective(trial):
    import torch
    import numpy as np
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42)
    lr = trial.suggest_loguniform("lr", 1e-6, 5e-5)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-3, 1e-1)
    dropout = trial.suggest_float("dropout", 0.2, 0.3)
    num_frozen_layers = trial.suggest_int("num_frozen_layers", 0, 12)
    batch_size = trial.suggest_categorical("batch_size", [4,5])
    num_epochs = trial.suggest_int("num_epochs", 2, 3)

    # Crear el modelo
    model = Wav2VecEmotionClassifier(
        pretrained_model="facebook/wav2vec2-base",
        num_classes=4,
        dropout=dropout,
        num_frozen_layers=num_frozen_layers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Configuración del dataset
    train_sessions = ["Session1", "Session2", "Session3"]
    test_sessions = ["Session4"]
    root_dir = "data/audio_preprocessed"

    checkpoint_path = f"audio/checkpoint_audio/trial_{trial.number}.pth"

    # Llamar a train con los parámetros del dataset
    val_loss, val_acc, val_f1 = train(
        model=model,
        root_dir=root_dir,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
        batch_size=batch_size,
        device=device,
        epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_path=checkpoint_path,
        resume_training=False,
        collate_fn=collate_fn
    )

    # Return metric
    if metric_to_optimize == "val_loss":
        return val_loss
    elif metric_to_optimize == "accuracy":
        return -val_acc
    elif metric_to_optimize == "f1":
        return -val_f1
    else:
        raise ValueError("Métrica no reconocida. Usa 'val_loss', 'accuracy' o 'f1'.")

if __name__ == "__main__":
    start_time = time.time()

    print(f"Optimizando métrica: {metric_to_optimize}")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=15)  # Puedes cambiar n_trials

    best_trial = study.best_trial
    print("Mejor combinación encontrada:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"Métrica ({metric_to_optimize}) óptima: {study.best_value:.4f}")

    # Guardar los mejores parámetros
    os.makedirs("audio/optuna_results", exist_ok=True)
    with open("audio/optuna_results/best_params_audio.txt", "w") as f:
        for key, value in best_trial.params.items():
            f.write(f"{key} = {value}\n")
        f.write(f"{metric_to_optimize} = {study.best_value:.4f}\n")

    print("Guardado en audio/optuna_results/best_params_audio.txt")

    elapsed = time.time() - start_time
    print(f"\nTiempo total: {elapsed:.2f} segundos.")