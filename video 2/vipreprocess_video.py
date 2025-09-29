import os
import csv
from pathlib import Path
import cv2
import numpy as np
import gc

IMG_SIZE = 224
FRAME_RATE = 2
MAX_FRAMES = 8

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
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB crudo
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))   # resize aquí
            frames.append(img.astype(np.uint8))
        frame_idx += 1

    cap.release()

    if not frames:
        print(f"[Warning] No se extrajeron frames de {video_path}")
        return False

    # Ajuste a MAX_FRAMES
    if len(frames) > MAX_FRAMES:
        indices = np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)
        frames = [frames[i] for i in indices]
    elif len(frames) < MAX_FRAMES:
        last_frame = frames[-1]
        while len(frames) < MAX_FRAMES:
            frames.append(last_frame)

    video_array = np.stack(frames)  # [MAX_FRAMES, H, W, 3] en uint8
    data_to_save = {"frames": video_array, "emotion": emotion}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, data_to_save, allow_pickle=True)

    del frames, video_array, data_to_save
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
                    for row in csv.reader(f):
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
    video_root = "D:/tesis/tesis/data/video"
    output_root = "D:/tesis/tesis/data/video_preprocessed"
    labels_dir = "D:/tesis/tesis/data/labels"
    process_videos(video_root, output_root, labels_dir)
