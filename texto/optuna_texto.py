# main.py
import torch
import optuna
import time
from train import train

metric_to_optimize = "f1"  # "val_loss", "accuracy" o "f1"

def objective(trial):
    """
    Función objetivo para Optuna. Ejecuta entrenamiento con hiperparámetros sugeridos
    y retorna la métrica elegida.
    """
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
    num_frozen_layers = trial.suggest_int("num_frozen_layers", 0, 11)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32,48])
    num_epochs = trial.suggest_int("num_epochs", 5, 6)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.3, step=0.05)  # Valores entre 5% y 30%
    grad_clip = trial.suggest_float("grad_clip", 0.5, 2.0, step=0.1)

    best_val_loss, val_accuracy, val_f1_macro = train(
        data_dir="data/text_tokenized",
        validation_session="Session4",
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        num_frozen_layers=num_frozen_layers,
        checkpoint_dir="texto/checkpoints_optuna",
        output_dir="texto/fine_tuned_model_optuna",
        warmup_ratio=warmup_ratio,  
        grad_clip=grad_clip
    )

    if metric_to_optimize == "val_loss":
        return best_val_loss  # Minimizar
    elif metric_to_optimize == "accuracy":
        return -val_accuracy  # Maximizar (pero Optuna minimiza, así que negamos)
    elif metric_to_optimize == "f1":
        return -val_f1_macro  # Maximizar F1 macro (lo mismo q arriba)
    else:
        raise ValueError("Métrica no reconocida. Usa 'val_loss', 'accuracy' o 'f1'.")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo disponible: {device}")

    # Crear el estudio
    direction = "minimize"
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))

    # Optimizar
    print(f"Optimizando hiperparámetros con Optuna ({metric_to_optimize}) ")
    start_time = time.time()
    study.optimize(objective, n_trials=20)
    elapsed = time.time() - start_time
    print(f"Finalizado en {elapsed:.2f} segundos.")

    # Mostrar mejores parámetros
    print("Mejores hiperparámetros encontrados:")
    for key, val in study.best_params.items():
        print(f"  • {key}: {val}")
    print(f"Metrica optima ({metric_to_optimize}) = {study.best_value:.4f}")

    # Guardar
    with open("best_params_optuna.txt", "w") as f:
        for key, val in study.best_params.items():
            f.write(f"{key} = {val}\n")
        f.write(f"{metric_to_optimize} = {study.best_value:.4f}\n")

    print("Hiperparametros optimos guardados en 'best_params_optuna.txt'.")
