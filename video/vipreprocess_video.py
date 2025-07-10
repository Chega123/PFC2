import os
import csv
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import gc

IMG_SIZE = 224
FRAME_RATE = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def extract_and_save_npy(video_path: Path, save_path: Path, emotion: str):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] No se pudo abrir {video_path}")
        return False

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
        return False

    video_tensor = np.stack(frames)  # [T, 3, 224, 224]
    data_to_save = {
        "frames": video_tensor,
        "emotion": emotion
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, data_to_save, allow_pickle=True)

    del frames, video_tensor, data_to_save
    gc.collect()
    return True

def process_videos(video_root, output_root, labels_dir):
    video_root = Path(video_root)
    output_root = Path(output_root)
    labels_dir = Path(labels_dir)

    sessions = os.listdir(video_root)
    for session in sessions:
        session_path = video_root / session
        if not session_path.is_dir():
            continue
        label_session_path = labels_dir / session
        if not label_session_path.is_dir():
            print(f"[Warning] Directorio de label no encontrado {session}")
            continue
        for gender in os.listdir(session_path):
            gender_path = session_path / gender
            label_gender_path = label_session_path / gender
            if not gender_path.is_dir() or not label_gender_path.is_dir():
                continue
            video_to_emotion = {}
            for label_file in label_gender_path.glob("*.csv"):
                with open(label_file, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:
                            continue
                        start_end, name, emotion = row
                        video_to_emotion[name] = emotion.strip()

            for file in os.listdir(gender_path):
                if file.endswith(".mp4"):
                    video_path = gender_path / file
                    name = file[:-4]
                    save_path = output_root / session / gender / f"{name}.npy"
                    emotion = video_to_emotion.get(name)
                    if emotion is None:
                        print(f"[Warning] No se encontro emocion {name} en {session}/{gender}")
                        continue
                    extract_and_save_npy(video_path, save_path, emotion)

if __name__ == "__main__":
    video_root = "D:/tesis/data/video"
    output_root = "D:/tesis/data/video_preprocessed"
    labels_dir = "D:/tesis/data/labels"
    process_videos(video_root, output_root, labels_dir)