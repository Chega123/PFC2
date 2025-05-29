import os
import csv
import cv2
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np

IMG_SIZE = 224
FRAME_RATE = 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def extract_and_save_npy(video_path: Path, save_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] No se pudo abrir {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps // FRAME_RATE))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            tensor = preprocess(img)
            frames.append(tensor.numpy())
        frame_idx += 1

    cap.release()

    if not frames:
        print(f"[Warning] No se extrajeron frames de {video_path}")
        return

    video_tensor = np.stack(frames)  # [T, 3, 224, 224]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, video_tensor)


def process_videos_like_audio(video_root, output_root):
    video_root = Path(video_root)
    output_root = Path(output_root)

    sessions = os.listdir(video_root)
    for session in sessions:
        session_path = video_root / session
        if not session_path.is_dir():
            continue
        for gender in os.listdir(session_path):
            gender_path = session_path / gender
            if not gender_path.is_dir():
                continue
            for file in os.listdir(gender_path):
                if file.endswith(".mp4"):
                    video_path = gender_path / file
                    name = file[:-4]
                    save_path = output_root / session / gender / f"{name}.npy"
                    extract_and_save_npy(video_path, save_path)


if __name__ == "__main__":
    video_root = "data/video"              # Estructura: /SesXX/gender/*.mp4
    output_root = "data/video_preprocessed"         # Salida .npy por video

    process_videos_like_audio(video_root, output_root)
