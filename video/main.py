import torch
import time
from train import train
import os

def read_best_params(file_path):
    """Lee los mejores hiperparámetros desde un archivo de texto."""
    params = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = [x.strip() for x in line.split("=")]
                    # Convertir valores a los tipos adecuados
                    if key in ["batch_size", "num_epochs", "num_frozen_layers", "num_layers_gru"]:
                        params[key] = int(value)
                    elif key in ["lr", "weight_decay", "dropout"]:
                        params[key] = float(value)
                    elif key == "hidden_size":
                        params[key] = int(value)  # Aunque es categórico, se guarda como int
                    elif key == "f1":
                        params["best_metric"] = float(value)
        return params
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo {file_path}. Ejecute primero la optimización con Optuna.")
    except Exception as e:
        raise ValueError(f"Error al leer {file_path}: {str(e)}")

def main():
    # Configuración inicial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Dispositivo disponible: {device}")

    # Leer los mejores hiperparámetros
    params_file = "best_params_optuna_vit.txt"
    print(f"📖 Leyendo mejores hiperparámetros desde {params_file}...")
    params = read_best_params(params_file)
    
    # Mostrar parámetros leídos
    print("✅ Hiperparámetros cargados:")
    for key, value in params.items():
        print(f"  • {key}: {value}")

    # Configuración de directorios y sesiones
    data_root = "data/video_preprocessed"
    sessions = ["Session1", "Session2", "Session3", "Session4"]
    validation_session = "Session4"
    genders = ["Male", "Female"]
    checkpoint_dir = "video/checkpoints/finetune"
    output_dir = "video/fine_tuned_model/final"

    # Verificar que los directorios de datos existen
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"El directorio de datos {data_root} no existe. Asegúrese de haber preprocesado los videos.")

    # Iniciar fine-tuning
    print(f"🚀 Iniciando fine-tuning con ViT en {device}...")
    start_time = time.time()
    
    try:
        best_val_f1, val_acc, val_loss = train(
            data_root=data_root,
            batch_size=params["batch_size"],
            num_epochs=80,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            dropout=params["dropout"],
            num_frozen_layers=params["num_frozen_layers"],
            hidden_size=params["hidden_size"],
            num_layers_gru=params["num_layers_gru"],
            sessions=sessions,
            validation_session=validation_session,
            genders=genders,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️ Fine-tuning finalizado en {elapsed:.2f} segundos.")
        print(f"✅ Resultados finales:")
        print(f"  • Mejor F1 en validación: {best_val_f1:.4f}")
        print(f"  • Precisión en validación: {val_acc:.4f}")
        print(f"  • Pérdida en validación: {val_loss:.4f}")
        print(f"📁 Modelo guardado en {output_dir}/best_model_final.pth")

    except Exception as e:
        print(f"⚠️ Error durante el fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()