import os
import numpy as np
from torch.utils.data import Dataset
import torch

class EmotionDataset(Dataset):
    def __init__(self, data_dir, session_filter=None):
        self.data = []
        self.emotion_to_idx = {
            "Neutral": 0,
            "Happy": 1,
            "Sad": 2,
            "Angry": 3,
            "Excited": 1
        }

        if not os.path.exists(data_dir):
            print(f"Error: Data directory {data_dir} no hay :c")
            return

        if session_filter is not None and not isinstance(session_filter, (list, tuple)):
            session_filter = [session_filter]

        for session in os.listdir(data_dir):
            # filtrar la de evaluacion
            if session_filter is not None and session not in session_filter:
                continue

            session_path = os.path.join(data_dir, session)
            for gender in os.listdir(session_path):
                gender_path = os.path.join(session_path, gender)
                for file in os.listdir(gender_path):
                    file_path = os.path.join(gender_path, file)
                    try:
                        record = np.load(file_path, allow_pickle=True).item()
                        input_ids = record.get("input_ids")
                        attention_mask = record.get("attention_mask")
                        emotion = record.get("emotion")
                        if emotion not in self.emotion_to_idx:
                            print(f"Skipping file {file_path}: Invalid emotion {emotion}")
                            continue
                        if input_ids is None or attention_mask is None:
                            print(f"Skipping file {file_path}: Missing input_ids or attention_mask")
                            continue
                        idx = self.emotion_to_idx[emotion]
                        self.data.append((input_ids, attention_mask, idx, session))
                    except Exception as e:
                        print(f"Error loading file {file_path}: {str(e)}")
        print(f"Total samples loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, attention_mask, label, session = self.data[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
            "session": session
        }
