import os
import json
import csv
# import re  <-- Ya no lo necesitamos

# Mapeo de emociones (el tuyo estÃ¡ perfecto)
emotion_map = {
    "Neutral": "neutral",
    "Happy": "happy",
    "Excited": "happy",   # se fusiona con Happy
    "Sad": "sad",
    "Angry": "angry"
}



def prepare_iemocap_split(videos_root, labels_root, transcriptions_root, sessions, output_file):
    """
    Genera un JSON con los samples (video + transcripciÃ³n + etiqueta)
    de las sesiones especificadas.
    
    Esta versiÃ³n lee la transcripciÃ³n desde un archivo .txt individual
    que coincide con el nombre del video.
    """
    data = []

    for session in sessions:
        session_label_path = os.path.join(labels_root, session)
        if not os.path.exists(session_label_path):
            print(f"Warning: session labels not found {session_label_path}")
            continue

        for gender in os.listdir(session_label_path):
            label_dir = os.path.join(session_label_path, gender)

            for csv_file in os.listdir(label_dir):
                if not csv_file.endswith(".csv"):
                    continue

                # 1. Procesar las etiquetas (como antes)
                csv_path = os.path.join(label_dir, csv_file)
                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) < 3:
                            continue

                        segment_id = row[1].strip()
                        emotion = row[2].strip()

                        if emotion not in emotion_map:
                            continue 

                        mapped_emotion = emotion_map[emotion]

                        # 2. Construir la ruta al archivo .txt de la transcripciÃ³n
                        trans_path = os.path.join(
                            transcriptions_root, session, gender, f"{segment_id}.txt"
                        )

                        # 3. Leer la transcripciÃ³n
                        if not os.path.exists(trans_path):
                            print(f"Warning: No transcription file found at: {trans_path}")
                            continue
                        
                        try:
                            with open(trans_path, 'r', encoding='utf-8') as f_trans:
                                transcription_text = f_trans.read().strip()
                        except Exception as e:
                            print(f"Error reading {trans_path}: {e}")
                            continue
                            
                        if not transcription_text:
                            print(f"Warning: Empty transcription for {segment_id}")
                            continue
                            
                        # 4. Ruta al video (como antes)
                        video_path = os.path.join(
                            videos_root, session, gender, f"{segment_id}.mp4"
                        )

                        if not os.path.exists(video_path):
                            print(f"Warning: Video not found: {video_path}")
                            continue

                        # 5. Construir el sample con el NUEVO PROMPT
                        
                        # Escapar comillas dobles en la transcripciÃ³n
                        transcription_text = transcription_text.replace('"', '\\"')

                        user_prompt = f'<video>\nTranscription: "{transcription_text}"\nBased on the video and text, what emotion is being expressed?'
                        
                        system_prompt = "You are an expert in multimodal emotion recognition. Your task is to analyze the video and the accompanying text transcription to determine the speaker's emotion. The possible emotions are: neutral, happy, sad, or angry. Respond only with the emotion label."

                        sample = {
                            "id": segment_id,
                            "video": video_path,
                            "conversations": [
                                {
                                    "from": "system",
                                    "value": system_prompt
                                },
                                {
                                    "from": "user",
                                    "value": user_prompt
                                },
                                {
                                    "from": "assistant",
                                    "value": mapped_emotion
                                }
                            ]
                        }
                        data.append(sample)

    # Guardar en JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Created {output_file} with {len(data)} samples")
    return len(data)


if __name__ == "__main__":
    
    # 1. Ruta a los videos (en tu carpeta WSL o Windows, como la tengas)
    # AsumirÃ© que esta ruta tambiÃ©n estÃ¡ en tu D: para mantenerlo simple
    videos_root = "/home/diego/tesis/data/IEMOCAP_original/videos"
    labels_root = "/home/diego/tesis/data/IEMOCAP_original/labels"
    
    # 3. Esta es la ruta clave que preguntaste
    transcriptions_root = "/mnt/d/tesis/tesis/data/text"

    # ðŸ‘‰ Sesiones
    train_sessions = ["Session1", "Session3", "Session4", "Session5"]
    val_sessions = ["Session2"]

    # Generar JSONs
    train_total = prepare_iemocap_split(
        videos_root, labels_root, transcriptions_root, train_sessions, "train.json"
    )
    val_total = prepare_iemocap_split(
        videos_root, labels_root, transcriptions_root, val_sessions, "val.json"
    )

    print(f"Train samples: {train_total}, Val samples: {val_total}")