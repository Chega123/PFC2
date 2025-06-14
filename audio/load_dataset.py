import os
import numpy as np
import torch
from torch.utils.data import dataset


class AudioDataset(dataset.Dataset):
    def __init__(self, root_dir,include_sessions=None):
        self.files =[]
        self.labels =[]
        include_sessions = include_sessions or []
        for root, _, filenames in os.walk(root_dir):
            if include_sessions:
                if not any(sess in root for sess in include_sessions):
                    continue

            for file in filenames:
                if file.endswith(".npy"):
                    self.files.append(os.path.join(root, file))
                    self.labels.append(None)  


    def emotion_to_id(self, emotion):
            mapping = {"angry": 0, "sad": 1, "happy": 2, "neutral": 3}
            return mapping.get(emotion.lower(), -1)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        waveform = torch.tensor(data["waveform"], dtype=torch.float32).unsqueeze(0)

        emotion = data.get("emotion", "").lower()
        label = torch.tensor(self.emotion_to_id(emotion), dtype=torch.long)
        if label == -1:
            print(f"Etiqueta inválida en archivo: {self.files[idx]}, emoción: {emotion}")
            raise ValueError(f"Etiqueta inválida: {emotion}")

        return waveform, label
