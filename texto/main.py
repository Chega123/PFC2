import torch
import time
from train import train

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    data_dir           = "data/text_tokenized"
    validation_session = "Session4"
    batch_size         = 16
    num_epochs         = 8
    learning_rate      = 1e-5
    checkpoint_dir     = "texto/checkpoints"
    output_dir         = "texto/fine_tuned_model"
    print("Empezando entrenamiento de texto")
    start_time = time.time()
    train(
        data_dir=data_dir,
        validation_session=validation_session,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=learning_rate,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir
    )

    elapsed = time.time() - start_time
    print(f"Entrenamiento completado en {elapsed:.2f} segundos.")

