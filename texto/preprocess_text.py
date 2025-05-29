from transformers import RobertaTokenizer
import os
import csv
import numpy as np
from tqdm import tqdm
import whisper

labels_dir = "data/labels"
text_dir = "data/text"
audio_dir = "data/speech"
output_dir = "data/text_tokenized"

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
whisper_model = whisper.load_model("base")  # "small,medium o large" hay varios modelos

def collect_text_data():
    data = []
    for session in os.listdir(labels_dir):
        session_path = os.path.join(labels_dir, session)
        for gender in os.listdir(session_path):
            gender_path = os.path.join(session_path, gender)
            for file in os.listdir(gender_path):
                if file.endswith(".csv"):
                    with open(os.path.join(gender_path, file)) as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if not row:
                                continue
                            start_end, name, emotion = row
                            txt_path = os.path.join(text_dir, session, gender, name + ".txt")

                            # por si falta texto aqui usamos whisper
                            if not os.path.exists(txt_path):
                                wav_path = os.path.join(audio_dir, session, gender, name + ".wav")
                                if os.path.exists(wav_path):
                                    try:
                                        result = whisper_model.transcribe(wav_path)
                                        print("////////////////////////////////////////////////")
                                        print(result)
                                        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                                        with open(txt_path, "w", encoding="utf-8") as out_f:
                                            out_f.write(result["text"].strip())
                                        print(f"Transcripción generada para {name}")
                                    except Exception as e:
                                        print(f"Error transcribiendo {name}: {e}")
                                        continue
                                else:
                                    print(f"Audio no encontrado para {name}")
                                    continue

                            # Si ya existe (o fue generado), cargamos
                            with open(txt_path, 'r', encoding='utf-8') as tf:
                                text = tf.read().strip()
                            if not text:
                                continue  # saltamos si está vacío
                            data.append({
                                "text": text,
                                "emotion": emotion.strip(),
                                "session": session,
                                "gender": gender,
                                "name": name
                            })
    return data

def tokenize_and_save():
    dataset = collect_text_data()
    for item in tqdm(dataset):
        text = item["text"]
        emotion = item["emotion"]
        session = item["session"]
        gender = item["gender"]
        name = item["name"]

        try:
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors="np"
            )
            output_path = os.path.join(output_dir, session, gender)
            os.makedirs(output_path, exist_ok=True)
            data_to_save = {
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "emotion": emotion
            }

            np.save(os.path.join(output_path, name + ".npy"), data_to_save)
        except Exception as e:
            print(f"Error procesando {name}: {e}")

if __name__ == "__main__":
    tokenize_and_save()
