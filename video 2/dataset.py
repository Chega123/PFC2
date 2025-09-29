"""  import os
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
    ):
        self.max_frames = 32  # fijo en 32
        self.emotion_to_idx = {
            "Neutral": 0,
            "Happy": 1,
            "Sad": 2,
            "Angry": 3,
            "Excited": 1,  # mapeado a Happy
        }

        self.samples = []
        for sess in sessions:
            for gen in genders:
                pattern = os.path.join(data_root, sess, gen, "*.npy")
                for fp in glob.glob(pattern):
                    self.samples.append(fp)

        print(f"Total de muestras de video: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp = self.samples[idx]

        # lazy loading: cargamos solo cuando se pide
        data = np.load(fp, allow_pickle=True).item()

        video_frames = data.get("frames")   # [32, 3, 224, 224]
        emotion = data.get("emotion")

        if video_frames is None or emotion is None:
            raise RuntimeError(f"Archivo corrupto: {fp}")

        label = self.emotion_to_idx.get(emotion)
        if label is None:
            raise RuntimeError(f"Emoción desconocida {emotion} en {fp}")

        frames = torch.from_numpy(video_frames).float()
        label = torch.tensor(label, dtype=torch.long)

        return {
            "frames": frames,   # [32, 3, 224, 224]
            "labels": label,    # int
        }
 
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Mapeo de emociones a índices
EMOTION_MAP = {
    "Neutral": 0,
    "Happy": 1,
    "Excited": 1,  # Excited se fusiona con Happy
    "Sad": 2,
    "Angry": 3
}

class VideoDataset(Dataset):
    
    Dataset que carga .npy con estructura:
      {
        "frames": np.ndarray (N, H, W, 3) uint8,
        "emotion": str
      }
    Devuelve:
      {
        "frames": list of PIL.Image,
        "label_idx": torch.LongTensor,
        "label_text": str
      }
    

    def __init__(self, root_dir, sessions=None, genders=("Female", "Male"), transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []

        all_sessions = sorted([
            s for s in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, s)) and s.lower().startswith("session")
        ])
        target_sessions = sessions if sessions else all_sessions

        for session in target_sessions:
            for gender in genders:
                gender_path = os.path.join(root_dir, session, gender)
                if not os.path.isdir(gender_path):
                    continue
                self.files.extend(sorted(glob.glob(os.path.join(gender_path, "*.npy"))))

        print(f"[INFO] Total de archivos de video encontrados ({target_sessions}): {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        data = np.load(fp, allow_pickle=True).item()

        frames = data["frames"]   # (N, H, W, 3)
        emotion = data["emotion"]

        frames_list = []
        for frame in frames:
            img = Image.fromarray(frame.astype(np.uint8))
            if self.transform:
                img = self.transform(img)
            frames_list.append(img)

        label_idx = EMOTION_MAP.get(emotion, -1)
        if label_idx == -1:
            raise ValueError(f"Emoción desconocida '{emotion}' en {fp}")

        return {
            "frames": frames_list,
            "label_idx": torch.tensor(label_idx, dtype=torch.long),
            "label_text": emotion
        }

def get_video_dataloader(root_dir, batch_size=2, shuffle=True, num_workers=2, sessions=None, genders=None):
    dataset = VideoDataset(root_dir, sessions, genders)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
 """

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Mapeo de emociones a índices
EMOTION_MAP = {
    "Neutral": 0,
    "Happy": 1,
    "Excited": 1,  # Excited se fusiona con Happy
    "Sad": 2,
    "Angry": 3
}

class VideoDataset(Dataset):
    """
    Dataset que carga .npy con estructura:
      {
        "frames": np.ndarray (N, H, W, 3) uint8,
        "emotion": str
      }
    Devuelve:
      {
        "frames": list of PIL.Image,
        "label_idx": torch.LongTensor,
        "label_text": str
      }
    """

    def __init__(self, root_dir, sessions=None, genders=("Female", "Male"), transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []

        all_sessions = sorted([
            s for s in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, s)) and s.lower().startswith("session")
        ])
        target_sessions = sessions if sessions else all_sessions

        for session in target_sessions:
            for gender in genders:
                gender_path = os.path.join(root_dir, session, gender)
                if not os.path.isdir(gender_path):
                    continue
                self.files.extend(sorted(glob.glob(os.path.join(gender_path, "*.npy"))))

        print(f"[INFO] Total de archivos de video encontrados ({target_sessions}): {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        data = np.load(fp, allow_pickle=True).item()

        frames = data["frames"]   # (N, H, W, 3)
        emotion = data["emotion"]

        frames_list = []
        for frame in frames:
            img = Image.fromarray(frame.astype(np.uint8))
            if self.transform:
                img = self.transform(img)
            frames_list.append(img)

        label_idx = EMOTION_MAP.get(emotion, -1)
        if label_idx == -1:
            raise ValueError(f"Emoción desconocida '{emotion}' en {fp}")

        return {
            "frames": frames_list,
            "label_idx": torch.tensor(label_idx, dtype=torch.long),
            "label_text": emotion
        }
