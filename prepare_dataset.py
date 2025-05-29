from pydub import AudioSegment
import os
import re
import csv
import cv2

base_dir = "dataset"
output_speech_dir = "data/speech"
output_text_dir = "data/text"
output_labels_dir = "data/labels"
output_video_dir = "data/video"
for d in (output_speech_dir, output_text_dir, output_labels_dir, output_video_dir):
    os.makedirs(d, exist_ok=True)

# Emociones de interés con etiquetas
target_emotions = {"ang": "Angry", "hap": "Happy", "exc": "Happy", "neu": "Neutral", "sad": "Sad"}

def load_valid_emotions(session_num):
    valid_utterances = {}
    emo_dir = os.path.join(base_dir, f"Session{session_num}", "dialog", "EmoEvaluation")
    for fname in os.listdir(emo_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(emo_dir, fname)) as f:
                for line in f:
                    m = re.match(r"^\[(\d+\.\d+) - (\d+\.\d+)\]\s(\S+)\s(\w+)", line)
                    if not m: continue
                    start, end, utt_id, emo = m.groups()
                    if emo in target_emotions:
                        valid_utterances[utt_id] = (float(start), float(end), target_emotions[emo])
    return valid_utterances

def convert_audio(audio_seg):
    return audio_seg.set_channels(1).set_frame_rate(16000)

def crop_video_cv2(avi_path, out_path, start, end, x1, y1, x2, y2):
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el video {avi_path}")
        return

    # FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 29.98

    # Coordenadas dentro de límites
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1 = max(0, min(x1, frame_w-1))
    x2 = max(x1+1, min(x2, frame_w))
    y1 = max(0, min(y1, frame_h-1))
    y2 = max(y1+1, min(y2, frame_h))
    width, height = x2-x1, y2-y1

    # Frames a recortar
    start_frame = int(start * fps)
    end_frame   = int(end   * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    current = 0
    while True:
        ret, frame = cap.read()
        if not ret or current > end_frame:
            break
        if current >= start_frame:
            cropped = frame[y1:y2, x1:x2]
            # validar dimensión antes de escribir
            if cropped.shape[0] != height or cropped.shape[1] != width:
                print(f"[{avi_path}] Saltando frame {current}: cropped {cropped.shape}")
            else:
                try:
                    out.write(cropped)
                except Exception as e:
                    print(f"[{avi_path}] Error al escribir frame {current}: {e}")
                    break
        current += 1

    cap.release()
    out.release()


def split_and_save(wav_path, avi_path, trans_path, session_num, valid_utts):
    # --- Audio ---
    audio = AudioSegment.from_wav(wav_path)

    # --- Vídeo: sólo para obtener w,h ---
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        print(f"Error abriendo {avi_path}")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    half_w = w // 2

    with open(trans_path) as f:
        for line in f:
            m = re.match(r"^(\S+) \[(\d+\.\d+)-(\d+\.\d+)\]: (.+)", line)
            if not m: continue
            utt_id, _, _, text = m.groups()
            if utt_id not in valid_utts: continue
            start, end, emo = valid_utts[utt_id]
            start_ms, end_ms = int(start*1000), int(end*1000)

            session_letter = utt_id.split('_')[0][-1]
            speaker_suffix = utt_id.split('_')[-1]    
            is_female_turn = (speaker_suffix[0] == 'F')

            if is_female_turn:
                # Queremos la MITAD donde está la mujer:
                if session_letter == 'F':
                    # mujer a la IZQUIERDA (corregido)
                    x1, x2 = 0, half_w
                else:
                    # sesión con 'M' → mujer a la DERECHA (corregido)
                    x1, x2 = half_w, w
                gender = 'Female'
            else:
                # es turno del hombre → recortamos la MITAD donde está el hombre:
                if session_letter == 'F':
                    # sesión F → hombre a la DERECHA (corregido)
                    x1, x2 = half_w, w
                else:
                    # sesión M → hombre a la IZQUIERDA (corregido)
                    x1, x2 = 0, half_w
                gender = 'Male'

            # Crear directorios por modalidad y género
            dir_speech = os.path.join(output_speech_dir, f"Session{session_num}", gender)
            dir_text   = os.path.join(output_text_dir,   f"Session{session_num}", gender)
            dir_labels = os.path.join(output_labels_dir, f"Session{session_num}", gender)
            dir_video  = os.path.join(output_video_dir,  f"Session{session_num}", gender)
            for d in (dir_speech, dir_text, dir_labels, dir_video):
                os.makedirs(d, exist_ok=True)

            # Exportar audio
            seg = convert_audio(audio[start_ms:end_ms])
            seg.export(os.path.join(dir_speech, f"{utt_id}.wav"), format='wav')

            # Guardar texto limpio
            txt = text.replace("[LAUGHTER]", "").replace("[BREATHING]", "").upper().strip()
            with open(os.path.join(dir_text, f"{utt_id}.txt"), 'w') as out:
                out.write(txt)

            # Guardar etiqueta CSV
            with open(os.path.join(dir_labels, f"{utt_id}.csv"), 'w', newline='') as out:
                csv.writer(out).writerow([f"[{start} - {end}]", utt_id, emo])

            # Recortar y exportar vídeo
            out_video = os.path.join(dir_video, f"{utt_id}.mp4")
            crop_video_cv2(avi_path, out_video, start, end, x1, 0, x2, h)

# Ejecutar para sesiones 1–3
for session_num in range(4, 6):
    valid = load_valid_emotions(session_num)
    session_path = os.path.join(base_dir, f"Session{session_num}", "dialog")
    wav_dir  = os.path.join(session_path, 'wav')
    txt_dir  = os.path.join(session_path, 'transcriptions')
    avi_dir  = os.path.join(session_path, 'avi', 'DivX')
    for fname in os.listdir(txt_dir):
        if not fname.endswith('.txt'):
            continue
        wav_file = os.path.join(wav_dir, fname.replace('.txt', '.wav'))
        avi_file = os.path.join(avi_dir, fname.replace('.txt', '.avi'))
        if os.path.exists(wav_file) and os.path.exists(avi_file):
            split_and_save(wav_file, avi_file, os.path.join(txt_dir, fname), session_num, valid)
