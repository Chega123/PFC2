# main.py

import torch
import os
from audio_model import Wav2VecEmotionClassifier
from train import train
from collate import collate_fn

def load_best_params(filepath):
    """Load the best hyperparameters from the Optuna results file."""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            key, value = line.strip().split(' = ')
            try:
                # Convert string values to appropriate types
                if key in ['lr', 'weight_decay', 'dropout']:
                    params[key] = float(value)
                elif key in ['num_frozen_layers', 'num_epochs', 'batch_size']:
                    params[key] = int(value)
                elif key == 'f1':  # Metric stored in file
                    params[key] = float(value)
            except ValueError:
                continue
    return params

def main():
    # Load best parameters from Optuna results
    best_params_file = "audio/optuna_results/best_params_audio.txt"
    if not os.path.exists(best_params_file):
        raise FileNotFoundError(f"Best parameters file not found at {best_params_file}")

    params = load_best_params(best_params_file)
    print("Loaded best parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Extract hyperparameters
    lr = params.get('lr', 1e-5)  # Default if not found
    weight_decay = params.get('weight_decay', 1e-2)
    dropout = params.get('dropout', 0.2)
    num_frozen_layers = params.get('num_frozen_layers', 0)
    batch_size = params.get('batch_size', 4)
    num_epochs = 100

    # Create the model
    model = Wav2VecEmotionClassifier(
        pretrained_model="facebook/wav2vec2-base",
        num_classes=4,
        dropout=dropout,
        num_frozen_layers=num_frozen_layers
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Dataset configuration
    train_sessions = ["Session1", "Session2", "Session3"]
    test_sessions = ["Session4"]
    root_dir = "data/audio_preprocessed"

    # Checkpoint path for saving the best model
    checkpoint_path = "audio/checkpoint_audio/best_model.pth"

    # Train the model with the best parameters
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

    print(f"Training completed. Final metrics:")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation F1 Score: {val_f1:.4f}")

if __name__ == "__main__":
    main()