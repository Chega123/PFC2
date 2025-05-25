import os
import re
import csv
from pathlib import Path
from pydub import AudioSegment
import cv2
import numpy as np

base_dir            = Path("dataset")
output_speech_dir   = Path("data/speech")
output_text_dir     = Path("data/text")
output_labels_dir   = Path("data/labels")
output_video_dir    = Path("data/video")

for d in (output_speech_dir, output_text_dir, output_labels_dir, output_video_dir):
    d.mkdir(parents=True, exist_ok=True)

# Mapeo de emociones
target_emotions = {"ang": "Angry", "hap": "Happy", "exc": "Happy", "neu": "Neutral", "sad": "Sad"}

def load_valid_emotions(session_num):
    from collections import defaultdict
    counts = defaultdict(int)
    data   = {}
    emo_dir = base_dir / f"Session{session_num}" / "dialog" / "EmoEvaluation"
    for emo_file in emo_dir.glob("*.txt"):
        for line in emo_file.open():
            m = re.match(r"^\[(\d+\.\d+)\s-\s(\d+\.\d+)\]\s(\S+)\s(\w+)", line)
            if not m: continue
            st, en, utt, emo = m.groups()
            if emo in target_emotions:
                label = target_emotions[emo]
                counts[(utt, label)] += 1
                data[(utt, label)] = (st, en)
    valid = {}
    for (utt, label), cnt in counts.items():
        if cnt >= 2:
            st, en = data[(utt, label)]
            valid[utt] = (float(st), float(en), label)
    return valid

def convert_audio_to_mono_16kHz(audio_segment):
    return audio_segment.set_channels(1).set_frame_rate(16000)

def split_and_save(wav_file, video_file, trans_file, session_num, valid_utterances):
    audio = AudioSegment.from_wav(wav_file)
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps == 0 or width == 0 or height == 0:
        print(f"Propiedades de video no válidas: {video_file}")
        cap.release()
        return

    for line in open(trans_file):
        m = re.match(r"^(\S+)\s\[(\d+\.\d+)-(\d+\.\d+)\]:\s(.+)", line)
        if not m: continue
        utt, _, _, text = m.groups()
        if utt not in valid_utterances: 
            continue
        st, en, emo = valid_utterances[utt]
        start_ms = int(st * 1000)
        end_ms = int(en * 1000)
        gender = "Male" if utt[5] == "M" else "Female"

        # AUDIO
        seg = convert_audio_to_mono_16kHz(audio[start_ms:end_ms])
        d = output_speech_dir / f"Session{session_num}" / gender
        d.mkdir(parents=True, exist_ok=True)
        seg.export(d / f"{utt}.wav", format="wav")

        # TEXTO 
        txt = re.sub(r"\[.*?\]", "", text).upper().strip()
        d = output_text_dir / f"Session{session_num}" / gender
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{utt}.txt").write_text(txt, encoding="utf-8")

        # LABEL
        d = output_labels_dir / f"Session{session_num}" / gender
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{utt}.csv", "w", newline="") as f:
            csv.writer(f).writerow([f"[{st} - {en}]", utt, emo])

        # VIDEO 
        # Calculate frame range
        start_frame = int(st * fps)
        end_frame = int(en * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if gender == "Male":
            x1, x2 = width // 2, width
        else:
            x1, x2 = 0, width // 2
        y1, y2 = 0, height
        crop_width = x2 - x1
        crop_height = y2 - y1

        d = output_video_dir / f"Session{session_num}" / gender
        d.mkdir(parents=True, exist_ok=True)
        out_path = str(d / f"{utt}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (crop_width, crop_height))

        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"Error leyendo frame {current_frame} en {video_file}")
                break
            # Crop frame
            cropped_frame = frame[y1:y2, x1:x2]
            out.write(cropped_frame)
            current_frame += 1
        out.release()
    cap.release()


for session_num in range(1, 4):
    valid = load_valid_emotions(session_num)
    dialog = base_dir / f"Session{session_num}" / "dialog"
    wav_dir   = dialog / "wav"
    trans_dir = dialog / "transcriptions"
    video_dir = dialog / "avi" / "DivX"  

    for trans in trans_dir.glob("*.txt"):
        wav = wav_dir / trans.name.replace(".txt", ".wav")
        video = video_dir / trans.name.replace(".txt", ".avi")
        if wav.exists() and video.exists():
            split_and_save(wav, video, trans, session_num, valid)
        else:
            print(f"Faltante: {wav if not wav.exists() else video}")
