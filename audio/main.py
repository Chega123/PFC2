import torch
import numpy as np
import random
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
import os
from audio_model import Wav2VecEmotionClassifier
from train import train
from collate import collate_fn

def load_best_params(filepath):
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            key, value = line.strip().split(' = ')
            try:
                if key in ['lr', 'weight_decay', 'dropout']:
                    params[key] = float(value)
                elif key in ['num_frozen_layers', 'num_epochs', 'batch_size']:
                    params[key] = int(value)
                elif key == 'f1':  
                    params[key] = float(value)
            except ValueError:
                continue
    return params

def main():
    best_params_file = "audio/optuna_results/best_params_audio.txt"
    if not os.path.exists(best_params_file):
        raise FileNotFoundError(f"Mejores hiperparametros encontrados:  {best_params_file}")

    params = load_best_params(best_params_file)
    print("Mejores hiperparametros cargados :")
    for key, value in params.items():
        print(f"  {key}: {value}")

    lr = params.get('lr', 1e-5) 
    weight_decay = params.get('weight_decay', 1e-2)
    dropout = params.get('dropout', 0.2)
    num_frozen_layers = params.get('num_frozen_layers', 0)
    batch_size = params.get('batch_size', 4)
    num_epochs = params.get('num_epochs', 15)

    model = Wav2VecEmotionClassifier(
        pretrained_model="facebook/wav2vec2-base",
        num_classes=4,
        dropout=dropout,
        num_frozen_layers=num_frozen_layers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Dispositivo usado: {device}")

    train_sessions = ["Session2", "Session3", "Session4", "Session5"]
    test_sessions = ["Session1"]
    root_dir = "data/audio_preprocessed"

    checkpoint_path = "audio/checkpoint_audio/best_model.pth"

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
        resume_training=True,
        collate_fn=collate_fn
    )

    print(f"Entrenamiento completo, metricas finales:")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation F1 Score: {val_f1:.4f}")

if __name__ == "__main__":
    main()