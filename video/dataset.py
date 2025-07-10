import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        sessions: list,
        genders: list,
        max_frames: int = 8
    ):

        self.max_frames = max_frames

        self.emotion_to_idx = {
            "Neutral": 0,
            "Happy": 1,
            "Sad": 2,
            "Angry": 3,
            "Excited": 1,  
        }
        self.data = []
        for sess in sessions:
            for gen in genders:
                pattern = os.path.join(data_root, sess, gen, "*.npy")
                for fp in glob.glob(pattern):
                    try:
                        data = np.load(fp, allow_pickle=True).item()
                        video_frames = data.get("frames")
                        emotion = data.get("emotion")
                        if video_frames is None or emotion is None:
                            print(f"Skipping {fp}: missing 'frames' or 'emotion'")
                            continue
                        if video_frames.shape[1:] != (3, 224, 224):
                            print(f"Skipping {fp}: invalid shape {video_frames.shape}")
                            continue
                        label = self.emotion_to_idx.get(emotion, self.emotion_to_idx.get(emotion))
                        if label is None:
                            print(f"Skipping {fp}: unknown emotion {emotion}")
                            continue
                        self.data.append((video_frames, label, sess)) 
                    except Exception as e:
                        print(f"Error loading {fp}: {e}")
                        continue
        print(f"Total de muestras de video cargadas: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_frames, label, session = self.data[idx]

        arr = np.array(video_frames) 
        num_frames = arr.shape[0]

        if num_frames >= self.max_frames:
            indices = np.linspace(0, num_frames - 1, self.max_frames).astype(int)
            frames = arr[indices]
        else:
            pad_len = self.max_frames - num_frames
            pads = np.repeat(arr[-1:], pad_len, axis=0)
            frames = np.concatenate([arr, pads], axis=0)

        frames = torch.from_numpy(frames).float()
        label = torch.tensor(label, dtype=torch.long)

        return {
            "frames": frames,
            "labels": label,
        }