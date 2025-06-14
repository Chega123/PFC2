import os
import csv
labels_="data/labels"
audio ="data/speech"

def collect_dataset_info():
    data=[]
    for session in os.listdir(labels_):
        session_path=os.path.join(labels_,session)
        for gender in os.listdir(session_path):
            gender_path=os.path.join(session_path,gender)
            for file in os.listdir(gender_path):
                if file.endswith(".csv"):
                    with open(os.path.join(gender_path,file)) as f:
                        reader=csv.reader(f)
                        for row in reader:
                            if not row: continue
                            start_end,name,emotion=row;
                            audio_file = os.path.join(audio, session, gender, name + ".wav")
                            if os.path.exists(audio_file):
                                data.append({
                                    "path": audio_file,
                                    "emotion": emotion.strip(),
                                    "session": session,
                                    "gender": gender,
                                    "name": name
                                })
    return data
