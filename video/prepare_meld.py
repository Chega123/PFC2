import os
import json
import pandas as pd
from pathlib import Path

emotion_map = {
    "neutral": "neutral",
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust"
}


def is_valid_video(video_path):
    try:
        return video_path.exists() and video_path.stat().st_size > 1024
    except:
        return False


def prepare_meld_split(meld_data_dir, session_name, output_file):
    data = []
    meld_data_dir = Path(meld_data_dir)
    
    csv_path = meld_data_dir / session_name / f"{session_name}_processed.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    skipped_emotion = 0
    skipped_missing = 0
    skipped_corrupt = 0
    
    for idx, row in df.iterrows():
        try:
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            emotion = row['Emotion'].lower().strip()
            utterance = str(row['Utterance']).strip()
            
            if emotion not in emotion_map:
                skipped_emotion += 1
                continue
            
            mapped_emotion = emotion_map[emotion]
            
            video_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
            video_path = meld_data_dir / session_name / "video" / video_filename
            
            if not video_path.exists():
                skipped_missing += 1
                continue
            
            if not is_valid_video(video_path):
                skipped_corrupt += 1
                continue
            
            transcription_escaped = utterance.replace('"', '\\"')
            
            user_prompt = f'<video>\nTranscription: "{transcription_escaped}"\nBased on the video and text, what emotion is being expressed?'
            
            system_prompt = "You are an expert in multimodal emotion recognition. Your task is to analyze the video and the accompanying text transcription to determine the speaker's emotion. The possible emotions are: neutral, joy, sadness, anger, surprise, fear, or disgust. Respond only with the emotion label."
            
            sample = {
                "id": f"dia{dialogue_id}_utt{utterance_id}",
                "video": str(video_path.absolute()),
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
            
        except Exception as e:
            continue
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Created {output_file} with {len(data)} samples")
    print(f"  Skipped - unknown emotion: {skipped_emotion}")
    print(f"  Skipped - missing video: {skipped_missing}")
    print(f"  Skipped - corrupt video: {skipped_corrupt}")
    
    return len(data)


if __name__ == "__main__":
    MELD_DATA_DIR = "/mnt/d/tesis/tesis/data/meld_data"
    
    train_total = prepare_meld_split(MELD_DATA_DIR, "Session1", "train.json")
    val_total = prepare_meld_split(MELD_DATA_DIR, "Session5", "val.json")
    test_total = prepare_meld_split(MELD_DATA_DIR, "Session6", "test.json")
    
    print(f"\nTrain: {train_total}, Val: {val_total}, Test: {test_total}")
    print(f"Total: {train_total + val_total + test_total}")