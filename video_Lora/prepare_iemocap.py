import os
import json
import csv

# Mapeo de emociones
emotion_map = {
    "Neutral": "neutral",
    "Happy": "happy",
    "Excited": "happy",   # se fusiona con Happy
    "Sad": "sad",
    "Angry": "angry"
}


def prepare_iemocap_split(videos_root, labels_root, sessions, output_file):
    """
    Genera un JSON con los samples de las sesiones especificadas.
    """
    data = []

    for session in sessions:
        session_path = os.path.join(labels_root, session)
        if not os.path.exists(session_path):
            print(f"Warning: session not found {session_path}")
            continue

        for gender in os.listdir(session_path):
            label_dir = os.path.join(session_path, gender)

            for csv_file in os.listdir(label_dir):
                if not csv_file.endswith(".csv"):
                    continue
                
                csv_path = os.path.join(label_dir, csv_file)

                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) < 3:
                            continue

                        # Ejemplo de fila: [6.2901 - 8.2357],Ses01F_impro01_F000,Neutral
                        segment_id = row[1].strip()
                        emotion = row[2].strip()

                        if emotion not in emotion_map:
                            continue  # ignorar emociones que no usamos

                        mapped_emotion = emotion_map[emotion]

                        # Ruta al video
                        video_path = os.path.join(
                            videos_root, session, gender, f"{segment_id}.mp4"
                        )

                        if not os.path.exists(video_path):
                            print(f"Warning: Video not found: {video_path}")
                            continue

                        sample = {
                            "id": segment_id,
                            "video": video_path,
                            "conversations": [
                                {
                                    "from": "user",
                                    "value": "<video>\nWhat emotion is being expressed in this video? Choose one: neutral, happy, sad, or angry."
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
    videos_root = "/home/diego/tesis/data/IEMOCAP_original/videos"
    labels_root = "/home/diego/tesis/data/IEMOCAP_original/labels"

    # 👉 Aquí defines qué sesiones son train y cuáles val
    train_sessions = ["Session1", "Session2", "Session3", "Session5"]
    val_sessions = ["Session4"]

    # Generar JSONs
    train_total = prepare_iemocap_split(videos_root, labels_root, train_sessions, "train.json")
    val_total = prepare_iemocap_split(videos_root, labels_root, val_sessions, "val.json")

    print(f"Train samples: {train_total}, Val samples: {val_total}")
