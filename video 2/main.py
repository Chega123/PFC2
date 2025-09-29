import torch
import numpy as np
import random
import time
import os
from train import train

# ------------------------------
# Configurar semillas
# ------------------------------
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)

# ------------------------------
# Leer parámetros desde archivo
# ------------------------------
def read_best_params(file_path):
    params = {}
    with open(file_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = [x.strip() for x in line.split("=")]
                if key in ["batch_size", "num_epochs", "num_frozen_layers", "num_layers_gru", "accumulation_steps", "num_virtual_tokens"]:
                    params[key] = int(value)
                elif key in ["lr", "weight_decay", "dropout"]:
                    params[key] = float(value)
                elif key == "f1":
                    params["best_metric"] = float(value)
                elif key == "model_name":
                    params[key] = str(value)
                elif key == "init_prompt":
                    params[key] = str(value)
    # Valores por defecto si no están en el archivo
    if "num_virtual_tokens" not in params:
        params["num_virtual_tokens"] = 20
    if "model_name" not in params:
        params["model_name"] = "Qwen/Qwen2-VL-2B-Instruct"
    if "init_prompt" not in params:
        params["init_prompt"] = "This image represents the emotion of (happy, sad, angry or neutral):"
    return params

# ------------------------------
# Main
# ------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo disponible: {device}")
    #, "Session2", "Session3", "Session4"
    # Configuración de datos
    data_root = "D:/tesis/tesis/data/video_preprocessed"
    sessions = ["Session1","Session5"]
    validation_session = "Session5"
    genders = ["Male", "Female"]
    checkpoint_dir = "video/checkpoints/finetune"
    output_dir = "video/fine_tuned_model/final"

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"No existe {data_root}. Asegúrate de haber preprocesado los videos.")

    # Leer hiperparámetros del archivo
    params_file = "D:/tesis/tesis/video/best_params_optuna_vit.txt"
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"No se encontró {params_file}. Ejecuta primero la optimización con Optuna.")
    
    print(f"Leyendo hiperparámetros desde {params_file}...")
    params = read_best_params(params_file)

    print("Hiperparámetros usados:")
    for k, v in params.items():
        if k != "best_metric":
            print(f"  • {k}: {v}")

    # Entrenamiento
    print(f"Iniciando fine-tuning con Qwen2-VL-2B en {device}...")
    start_time = time.time()

    best_val_f1, val_acc, val_loss = train(
        data_root=data_root,
        batch_size=params.get("batch_size", 2),
        num_epochs=params.get("num_epochs", 3),
        lr=params.get("lr", 5e-4),
        weight_decay=params.get("weight_decay", 0.01),
        sessions=sessions,
        validation_session=validation_session,
        genders=genders,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        accumulation_steps=params.get("accumulation_steps", 1),
        model_name=params.get("model_name", "Qwen/Qwen2-VL-2B-Instruct"),
        init_prompt=params.get("init_prompt",  "This image represents the emotion of (happy, sad, angry or neutral):")
    )

    elapsed = time.time() - start_time
    print(f"\nFine-tuning finalizado en {elapsed:.2f} segundos.")
    print(f"Resultados finales:")
    print(f"Mejor F1 en validación: {best_val_f1:.4f}")
    print(f"Precisión en validación: {val_acc:.4f}")
    print(f"Pérdida en validación: {val_loss:.4f}")
    print(f"Modelo guardado en {output_dir}/best_model")

if __name__ == "__main__":
    main()