import os
import json
from pathlib import Path

emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

sentences = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes"
}


def load_actor_splits(splits_file='cremad_actor_splits.json'):
    if not os.path.exists(splits_file):
        raise FileNotFoundError(
            f"Archivo de splits no encontrado: {splits_file}\n"
            f"Ejecuta primero: python generate_actor_splits.py"
        )
    
    with open(splits_file, 'r') as f:
        data = json.load(f)
    
    return data['splits']


def parse_filename(filename):
    name = filename.replace('.mp4', '').replace('.flv', '').replace('.avi', '')
    parts = name.split('_')
    
    if len(parts) != 4:
        return None
    
    actor_id, sentence_code, emotion_code, level = parts
    
    if emotion_code not in emotion_map or sentence_code not in sentences:
        return None
    
    return {
        'actor_id': actor_id,
        'sentence_code': sentence_code,
        'sentence': sentences[sentence_code],
        'emotion': emotion_map[emotion_code],
        'emotion_code': emotion_code,
        'level': level,
        'filename': filename
    }


def prepare_cremad_split(videos_root, sessions_actors, output_file):
    data = []
    videos_root = Path(videos_root)
    
    if not videos_root.exists():
        raise FileNotFoundError(f"Videos root not found: {videos_root}")
    
    video_files = list(videos_root.glob("*.mp4"))
    
    if len(video_files) == 0:
        print("Warning: No .mp4 files found, trying .flv...")
        video_files = list(videos_root.glob("*.flv"))
    
    print(f"Total videos found: {len(video_files)}")
    
    for video_path in video_files:
        parsed = parse_filename(video_path.name)
        if parsed is None:
            continue
        
        actor_id = parsed['actor_id']
        
        if actor_id not in sessions_actors:
            continue
        
        transcription = parsed['sentence']
        emotion = parsed['emotion']
        
        transcription_escaped = transcription.replace('"', '\\"')
        
        user_prompt = f'<video>\nTranscription: "{transcription_escaped}"\nBased on the video and text, what emotion is being expressed?'
        
        system_prompt = "You are an expert in multimodal emotion recognition. Your task is to analyze the video and the accompanying text transcription to determine the speaker's emotion. The possible emotions are: neutral, happy, sad, angry, disgust, or fear. Respond only with the emotion label."
        
        sample = {
            "id": video_path.stem,
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
                    "value": emotion
                }
            ]
        }
        data.append(sample)
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Created {output_file} with {len(data)} samples")
    return len(data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare CREMA-D videos for training')
    parser.add_argument('--videos_root', type=str, 
                        default="/home/diego/tesis/data/VideoMP4",
                        help='Path to CREMA-D video folder')
    parser.add_argument('--splits_file', type=str,
                        default="cremad_actor_splits.json",
                        help='Path to actor splits JSON')
    
    args = parser.parse_args()
    
    videos_root = args.videos_root
    splits_file = args.splits_file
    
    print("="*70)
    print("PREPARING CREMA-D FOR VIDEO TRAINING")
    print("="*70)
    print(f"Videos: {videos_root}")
    print(f"Splits: {splits_file}")
    print("="*70 + "\n")
    
    splits = load_actor_splits(splits_file)
    
    train_actors = splits['Session1']
    val_actors = splits['Session5']
    test_actors = splits['Session6']
    
    print(f"Train actors: {len(train_actors)}")
    print(f"Val actors: {len(val_actors)}")
    print(f"Test actors: {len(test_actors)}\n")
    
    train_total = prepare_cremad_split(videos_root, train_actors, "train.json")
    val_total = prepare_cremad_split(videos_root, val_actors, "val.json")
    test_total = prepare_cremad_split(videos_root, test_actors, "test.json")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"Train samples: {train_total}")
    print(f"Val samples: {val_total}")
    print(f"Test samples: {test_total}")
    print(f"Total: {train_total + val_total + test_total}")
    print("="*70)