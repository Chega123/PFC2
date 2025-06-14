# main.py

import optuna
import torch
import time
from train import train

metric_to_optimize = "f1"  # "val_loss", "accuracy" o "f1"

def objective(trial):
    """
    Función objetivo para Optuna. Ejecuta entrenamiento con hiperparámetros sugeridos
    y retorna la métrica elegida.
    """

    lr = trial.suggest_loguniform("lr", 1e-6, 5e-5)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-3, 1e-1)
    dropout = trial.suggest_float("dropout", 0.2, 0.3)
    num_frozen_layers = trial.suggest_int("num_frozen_layers", 0, 11)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32,48])
    num_epochs = trial.suggest_int("num_epochs", 2, 6)

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
        output_dir="texto/fine_tuned_model_optuna"
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
    # Mostrar el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo disponible: {device}")

    # Crear el estudio
    direction = "minimize"
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))

    # Optimizar
    print(f"▶️ Optimizando hiperparámetros con Optuna ({metric_to_optimize})...")
    start_time = time.time()
    study.optimize(objective, n_trials=30)
    elapsed = time.time() - start_time
    print(f"⏱️ Finalizado en {elapsed:.2f} segundos.")

    # Mostrar mejores parámetros
    print("✅ Mejores hiperparámetros encontrados:")
    for key, val in study.best_params.items():
        print(f"  • {key}: {val}")
    print(f"Métrica óptima ({metric_to_optimize}) = {study.best_value:.4f}")

    # Guardar
    with open("best_params_optuna.txt", "w") as f:
        for key, val in study.best_params.items():
            f.write(f"{key} = {val}\n")
        f.write(f"{metric_to_optimize} = {study.best_value:.4f}\n")

    print("Hiperparámetros óptimos guardados en `best_params_optuna.txt`.")
