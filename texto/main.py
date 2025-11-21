# main.py
import torch
import numpy as np
import random
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
from train import train

def load_best_params(filepath="texto\\best_params_optuna.txt"):
    params = {}
    with open(filepath, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=")
                key = key.strip()
                val = val.strip()
                if key in ["batch_size", "num_epochs", "num_frozen_layers"]:
                    val = int(val)
                elif key in ["lr", "weight_decay", "dropout","warmup_ratio","grad_clip"]:
                    val = float(val)
                params[key] = val
    return params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo disponible: {device}")

    print("Cargando hiperparámetros óptimos desde Optuna...")
    best_params = load_best_params()

    print("Entrenando modelo con hiperparámetros óptimos:")
    for key, val in best_params.items():
        print(f"  • {key}: {val}")

    best_val_loss, val_accuracy, val_f1_macro = train(
        data_dir="data/text_tokenized",
        validation_session="Session1",
        batch_size=best_params["batch_size"],
        num_epochs=25,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        dropout=best_params["dropout"],
        num_frozen_layers=best_params["num_frozen_layers"],
        checkpoint_dir="texto/checkpoints_final",
        output_dir="texto/fine_tuned_model_final",
        warmup_ratio=best_params["warmup_ratio"],
        grad_clip=best_params["grad_clip"]
    )

    print(" Entrenamiento finalizado con métricas:")
    print(f"  • Val Loss: {best_val_loss:.4f}")
    print(f"  • Val Accuracy: {val_accuracy:.4f}")
    print(f"  • Val F1 Macro: {val_f1_macro:.4f}")
