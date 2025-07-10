import optuna
import time
import torch
from train import train 

metric_to_optimize = "f1"  # Options: "f1", "accuracy", "val_loss"


def objective(trial):
    try:
        import torch
        import numpy as np
        import random
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(42)
        batch_size = trial.suggest_categorical("batch_size", [4,5])  #RAM D:
        num_epochs = trial.suggest_int("num_epochs", 4,5)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-3, 0.3, log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.4)  
        num_frozen_layers = trial.suggest_int("num_frozen_layers", 0, 9)  # ViT 12 layers
        hidden_size = trial.suggest_categorical("hidden_size", [768])
        num_layers_gru = trial.suggest_int("num_layers_gru", 1, 3)  # Reducir max layers

        best_val_f1, val_acc, avg_val_loss = train(
            data_root="data/video_preprocessed",
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            num_frozen_layers=num_frozen_layers,
            hidden_size=hidden_size,
            num_layers_gru=num_layers_gru,
            sessions=["Session1", "Session2", "Session3", "Session4"],
            validation_session="Session4",
            genders=["Male", "Female"],
            checkpoint_dir=f"video/checkpoints/trial_{trial.number}",
            output_dir=f"video/fine_tuned_model/trial_{trial.number}"
        )

        # Optuna minimiza
        if metric_to_optimize == "val_loss":
            return avg_val_loss
        elif metric_to_optimize == "accuracy":
            return -val_acc
        elif metric_to_optimize == "f1":
            return -best_val_f1
        else:
            raise ValueError("Metrica no reconocida.")

    except Exception as e:
        print(f"prueba {trial.number} fallo: {str(e)}")
        return float("inf")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo disponible: {device}")


    study_name = "vit_gru_optimization"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1) 
    )

    print(f"Optimizando hiperparámetros con Optuna para ViT ({metric_to_optimize})...")

    start_time = time.time()
    study.optimize(objective, n_trials=20)  #cantidad de pruebas
    elapsed = time.time() - start_time

    print(f"Finalizado en {elapsed:.2f} segundos.")
    print("Mejores hiperparámetros encontrados:")
    for key, val in study.best_params.items():
        print(f"  • {key}: {val}")
    best_metric = study.best_value
    if metric_to_optimize in ["f1", "accuracy"]:
        best_metric = -best_metric
    print(f"Métrica optima ({metric_to_optimize}) = {best_metric:.4f}")

    with open("best_params_optuna_vit.txt", "w") as f:
        for key, val in study.best_params.items():
            f.write(f"{key} = {val}\n")
        f.write(f"{metric_to_optimize} = {best_metric:.4f}\n")

    print(f"Hiperparametros optimos guardados en 'best_params_optuna_vit.txt'.")