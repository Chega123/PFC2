import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, root_dir, include_sessions=None, sample_rate=16000):

        self.files = []
        self.labels = []
        self.sample_rate = sample_rate
        include_sessions = include_sessions or []

        for root, _, filenames in os.walk(root_dir):
            if include_sessions:
                if not any(sess in root for sess in include_sessions):
                    continue

            for file in filenames:
                if file.endswith(".npy"):
                    self.files.append(os.path.join(root, file))
                    self.labels.append(None)  # placeholder (las labels se toman del .npy)

        if len(self.files) == 0:
            print(f"No se encontraron archivos .npy en {root_dir} con sesiones {include_sessions}")

    def emotion_to_id(self, emotion):
        mapping = {"angry": 0, "sad": 1, "happy": 2, "neutral": 3}
        return mapping.get(emotion.lower(), -1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        path = data["path"]          # ruta al wav
        emotion = data["emotion"]     # etiqueta

        # Carga el audio 
        waveform, sr = torchaudio.load(path)  # waveform = (channels, samples)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Asegurarse que sea (1, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        label = torch.tensor(self.emotion_to_id(emotion), dtype=torch.long)
        if label == -1:
            raise ValueError(f"Etiqueta invalida en archivo {path}: {emotion}")

        return waveform, label
