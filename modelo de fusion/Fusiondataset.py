import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def check_directories(text_dir, audio_dir, video_dir, label_dir, sessions):
    print(f"Checkeando todas las sesiones: {sessions or 'All'}")
    for dir_path, dir_name in [(text_dir, "Text"), (audio_dir, "Audio"), (video_dir, "Video"), (label_dir, "Label")]:
        if not os.path.exists(dir_path):
            print(f"{dir_name} directorio {dir_path} no existe.")
            continue
        print(f"\n{dir_name} directorio: {dir_path}")
        session_count = 0
        file_count = 0
        for session in os.listdir(dir_path):
            session_path = os.path.join(dir_path, session)
            if not os.path.isdir(session_path):
                continue
            if sessions and not any(session.lower() == s.lower() for s in sessions):
                continue
            session_count += 1
            gender_count = sum(1 for g in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, g)))
            for gender in os.listdir(session_path):
                gender_path = os.path.join(session_path, gender)
                if not os.path.isdir(gender_path):
                    continue
                files = [f for f in os.listdir(gender_path) if f.endswith(".npy" if dir_name != "Label" else ".csv")]
                file_count += len(files)
                print(f"  Session: {session}, Gender: {gender}, Files: {len(files)}")
            print(f"  Total sessions: {session_count}, Total genders: {gender_count}, Total files: {file_count}")
        if session_count == 0:
            print(f"Sesion no valida encontrada en el  directorio {dir_name}.")

class MultimodalEmotionDataset(Dataset):
    def __init__(self, text_dir, audio_dir, video_dir, label_dir, sessions=None, max_frames=8, max_waveform_length=160000):
        self.text_dir = text_dir
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.max_frames = max_frames
        self.max_waveform_length = max_waveform_length
        self.sessions = sessions if sessions is not None else []
        self.emotion_to_idx = {
            "neutral": 0, "happy": 1, "sad": 2, "angry": 3, "excited": 1
        }
        self.samples = []  #(text_file, audio_file, video_file, label_file, filename)

        for dir_path in [text_dir, audio_dir, video_dir, label_dir]:
            if not os.path.exists(dir_path):
                raise ValueError(f"Directory {dir_path} does not exist.")

        # Recolectar nombres de archivo base presentes en todos los directorios
        for session in os.listdir(text_dir):
            if self.sessions and not any(session.lower() == s.lower() for s in self.sessions):
                continue
            session_text_path = os.path.join(text_dir, session)
            session_audio_path = os.path.join(audio_dir, session)
            session_video_path = os.path.join(video_dir, session)
            session_label_path = os.path.join(label_dir, session)

            for gender in os.listdir(session_text_path):
                text_gender_path = os.path.join(session_text_path, gender)
                audio_gender_path = os.path.join(session_audio_path, gender)
                video_gender_path = os.path.join(session_video_path, gender)
                label_gender_path = os.path.join(session_label_path, gender)

                # Obtener nombres de archivo base
                text_files = {os.path.splitext(f)[0] for f in os.listdir(text_gender_path) if f.endswith(".npy")}
                audio_files = {os.path.splitext(f)[0] for f in os.listdir(audio_gender_path) if f.endswith(".npy")}
                video_files = {os.path.splitext(f)[0] for f in os.listdir(video_gender_path) if f.endswith(".npy")}
                label_files = {os.path.splitext(f)[0] for f in os.listdir(label_gender_path) if f.endswith(".csv")}

                # Encontrar intersección de nombres de archivo
                common_files = text_files & audio_files & video_files & label_files

                for filename in common_files:
                    text_file = os.path.join(text_gender_path, f"{filename}.npy")
                    audio_file = os.path.join(audio_gender_path, f"{filename}.npy")
                    video_file = os.path.join(video_gender_path, f"{filename}.npy")
                    label_file = os.path.join(label_gender_path, f"{filename}.csv")
                    self.samples.append((text_file, audio_file, video_file, label_file, filename))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_file, audio_file, video_file, label_file, filename = self.samples[idx]

        text_data = np.load(text_file, allow_pickle=True)
        audio_data = np.load(audio_file, allow_pickle=True)
        video_data = np.load(video_file, allow_pickle=True)

        text_data = text_data.item() if isinstance(text_data, np.ndarray) and text_data.dtype == object else text_data
        audio_data = audio_data.item() if isinstance(audio_data, np.ndarray) and audio_data.dtype == object else audio_data
        video_data = video_data.item() if isinstance(video_data, np.ndarray) and video_data.dtype == object else video_data

        # Extraer componentes
        input_ids = text_data.get("input_ids")
        attention_mask = text_data.get("attention_mask")
        waveform = audio_data.get("waveform")
        frames = video_data.get("frames")

        # Cargar etiqueta desde CSV
        label_df = pd.read_csv(label_file, header=None)
        if len(label_df) != 1:
            raise ValueError(f"Columna invalida en CSV{label_file}")
        emotion = str(label_df.iloc[0, 2]).strip().lower()
        if emotion not in self.emotion_to_idx:
            raise ValueError(f"Emocion invalida {emotion}")
        label = self.emotion_to_idx[emotion]

        # Procesar texto
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Procesar audio
        waveform = np.array(waveform, dtype=np.float32)
        if len(waveform) > self.max_waveform_length:
            waveform = waveform[:self.max_waveform_length]
        else:
            waveform = np.pad(waveform, (0, self.max_waveform_length - len(waveform)), mode='constant')
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # [1, max_waveform_length]

        # Procesar video
        frames = np.array(frames)
        num_frames = frames.shape[0]
        if num_frames >= self.max_frames:
            indices = np.linspace(0, num_frames - 1, self.max_frames).astype(int)
            frames = frames[indices]
        else:
            pad_len = self.max_frames - num_frames
            pads = np.repeat(frames[-1:], pad_len, axis=0)
            frames = np.concatenate([frames, pads], axis=0)
        frames = torch.tensor(frames, dtype=torch.float32) 

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "waveform": waveform,
            "frames": frames,
            "label": torch.tensor(label, dtype=torch.long),
            "filename": filename
        }


def get_multimodal_dataloader(text_dir, audio_dir, video_dir, label_dir, sessions=None, batch_size=4, shuffle=True, max_waveform_length=160000):
    dataset = MultimodalEmotionDataset(text_dir, audio_dir, video_dir, label_dir, sessions, max_frames=8, max_waveform_length=max_waveform_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

if __name__ == "__main__":
    text_dir = "D:/tesis/data/text_tokenized"
    audio_dir = "D:/tesis/data/audio_preprocessed"
    video_dir = "D:/tesis/data/video_preprocessed"
    label_dir = "D:/tesis/data/labels"
    sessions = ["Session1", "Session2", "Session3"]

    check_directories(text_dir, audio_dir, video_dir, label_dir, sessions)
    dataloader = get_multimodal_dataloader(text_dir, audio_dir, video_dir, label_dir, sessions, batch_size=4)
    for batch in dataloader:
        print("Batch shapes:")
        print(f"input_ids: {batch['input_ids'].shape}")
        print(f"attention_mask: {batch['attention_mask'].shape}")
        print(f"waveform: {batch['waveform'].shape}")
        print(f"frames: {batch['frames'].shape}")
        print(f"labels: {batch['label'].shape}")
        print(f"filenames: {batch['filename']}")
        print("\nLabels per sample:")
        idx_to_emotion = {0: "neutral", 1: "happy", 2: "sad", 3: "angry"}  # Para mapear índices a emociones
        for filename, label in zip(batch['filename'], batch['label']):
            print(f"{filename}: Label {label.item()} ({idx_to_emotion.get(label.item(), 'unknown')})")
        break