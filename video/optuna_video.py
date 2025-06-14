import optuna
import torch
import time
from train import train 

metric_to_optimize = "f1"  # Options: "f1", "accuracy", "val_loss"


def objective(trial):
    try:
        # Define hyperparameter search space
        batch_size = trial.suggest_categorical("batch_size", [4,6,8])  # Further reduced for RAM efficiency
        num_epochs = trial.suggest_int("num_epochs", 3, 5)  # Reduced for faster trials
        lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.3)  # Narrowed range
        num_frozen_layers = trial.suggest_int("num_frozen_layers", 0, 12)  # ViT has 12 layers
        hidden_size = trial.suggest_categorical("hidden_size", [768])
        num_layers_gru = trial.suggest_int("num_layers_gru", 1, 2)  # Reduced max layers

        # Call the train function with the suggested hyperparameters
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

        # Return the metric to optimize (Optuna always minimizes)
        if metric_to_optimize == "val_loss":
            return avg_val_loss
        elif metric_to_optimize == "accuracy":
            return -val_acc
        elif metric_to_optimize == "f1":
            return -best_val_f1
        else:
            raise ValueError("Metric not recognized. Use 'val_loss', 'accuracy', or 'f1'.")

    except Exception as e:
        print(f"⚠️ Trial {trial.number} failed: {str(e)}")
        return float("inf")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Dispositivo disponible: {device}")

    # Create Optuna study without storage
    study_name = "vit_gru_optimization"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)  # Added pruning
    )

    print(f"🚀 Optimizando hiperparámetros con Optuna para ViT ({metric_to_optimize})...")

    start_time = time.time()
    study.optimize(objective, n_trials=15)  # Reduced trials for faster execution
    elapsed = time.time() - start_time

    print(f"⏱️ Finalizado en {elapsed:.2f} segundos.")
    print("✅ Mejores hiperparámetros encontrados:")
    for key, val in study.best_params.items():
        print(f"  • {key}: {val}")
    best_metric = study.best_value
    if metric_to_optimize in ["f1", "accuracy"]:
        best_metric = -best_metric
    print(f"🏆 Métrica óptima ({metric_to_optimize}) = {best_metric:.4f}")

    with open("best_params_optuna_vit.txt", "w") as f:
        for key, val in study.best_params.items():
            f.write(f"{key} = {val}\n")
        f.write(f"{metric_to_optimize} = {best_metric:.4f}\n")

    print(f"📁 Hiperparámetros óptimos guardados en `best_params_optuna_vit.txt`.")