import torch
from torch.utils.data import DataLoader
import time
from model import Wav2VecEmotion  # Usa tu clase real, no el classifier wrapper
from load_dataset import AudioDataset
from train import train
from collate import collate_fn


if __name__ == "__main__":
    root_dir = "data/audio_preprocessed"
    batch_size = 4
    num_epochs = 15
    learning_rate = 2e-5
    num_classes = 4
    checkpoint_path = "audio/checkpoint_audio/wav2vec_emotion_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    train_sessions = ["Session1", "Session2","Session3"]
    dataset = AudioDataset(root_dir="data/audio_preprocessed", include_sessions=train_sessions)
    test_sessions = ["Session4"]
    test_dataset = AudioDataset(root_dir="data/audio_preprocessed", include_sessions=test_sessions)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = Wav2VecEmotion(pretrained_model="facebook/wav2vec2-base", num_classes=num_classes)
    model.to(device)
    print("Empezando entrenamiento")
    start_time = time.time()
    train(model, dataloader, test_dataloader, device, epochs=num_epochs, lr=learning_rate, checkpoint_path=checkpoint_path, resume_training=True)
    end_time = time.time()

    elapsed = end_time - start_time
    total_iters = num_epochs * len(dataloader)
    print(f"Entrenamiento completado en {elapsed:.2f} segundos.")
    print(f"Iteraciones por segundo: {total_iters / elapsed:.2f}")
