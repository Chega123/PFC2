import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast
from transformers import RobertaTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)

# System paths
sys.path.append('D:/tesis/audio')
sys.path.append('D:/tesis/texto')
sys.path.append('D:/tesis/video')
sys.path.append('D:/tesis/modelo de fusion')

from audio_model import Wav2VecEmotionClassifier
from texto_model import RobertaEmbeddingExtractor, get_tokenizer_and_model
from video_model import VideoEmbeddingExtractor
from hierarchical import HierarchicalFusionModule
from auto_attention import AutoAttentionFusionModule
from final_fusion import FinalFusionMLP
from Fusiondataset import get_multimodal_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

emotion_to_idx = {
    "neutral": 0, "happy": 1, "sad": 2, "angry": 3, "excited": 1
}

idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}

def load_models():
    #carga de modelos
    text_tokenizer, text_model = get_tokenizer_and_model(model_name="roberta-base", return_classifier=False, device=device)
    checkpoint = torch.load("D:/tesis/texto/checkpoints_final/roberta_best_f1_checkpoint.pth", map_location=device)
    text_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    #############################
    audio_model = Wav2VecEmotionClassifier(pretrained_model="facebook/wav2vec2-base", num_classes=4, dropout=0.3, num_frozen_layers=0).to(device)
    checkpoint = torch.load("D:/tesis/audio/checkpoint_audio/best_model.pth", map_location=device)
    audio_model.load_state_dict(checkpoint["model_state_dict"])
    #############################
    video_model = VideoEmbeddingExtractor(hidden_size=768, num_layers=1, dropout=0.0, num_frozen_layers=0).to(device)
    checkpoint = torch.load("D:/tesis/video/checkpoints/finetune/best_model.pth", map_location=device)
    video_model.load_state_dict(checkpoint, strict=False)

    hierarchical_fusion = HierarchicalFusionModule(embed_dim=768).to(device)
    auto_attention_fusion = AutoAttentionFusionModule(embed_dim=768).to(device)
    final_mlp = FinalFusionMLP(embed_dim=768).to(device)
    checkpoint = torch.load("D:/tesis/modelo de fusion/fusion_checkpoint_best.pth", map_location=device)
    hierarchical_fusion.load_state_dict(checkpoint["hierarchical_state_dict"])
    auto_attention_fusion.load_state_dict(checkpoint["auto_attention_state_dict"])
    final_mlp.load_state_dict(checkpoint["final_mlp_state_dict"])


    text_model.eval()
    audio_model.eval()
    video_model.eval()
    hierarchical_fusion.eval()
    auto_attention_fusion.eval()
    final_mlp.eval()

    return text_model, audio_model, video_model, hierarchical_fusion, auto_attention_fusion, final_mlp


def predict(dataloader, models, device):
    text_model, audio_model, video_model, hierarchical_fusion, auto_attention_fusion, final_mlp = models
    all_preds = []
    all_labels = []
    all_filenames = []
    incorrect_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            waveform = batch["waveform"].to(device)
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            filenames = batch["filename"]

            with autocast():
                text_emb = text_model.extract_features(input_ids, attention_mask)
                audio_emb = audio_model.embedding.extract_features(waveform)
                video_emb = video_model.extract_features(frames)
                v1 = hierarchical_fusion(text_emb, audio_emb, video_emb)
                v2 = auto_attention_fusion(text_emb, audio_emb, video_emb)
                logits = final_mlp(v1, v2)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_filenames.extend(filenames)


            for filename, true_label, pred_label in zip(filenames, labels, preds):
                if true_label != pred_label:
                    incorrect_predictions.append({
                        "filename": filename,
                        "true_label": true_label,
                        "true_emotion": idx_to_emotion.get(true_label, "unknown"),
                        "pred_label": pred_label,
                        "pred_emotion": idx_to_emotion.get(pred_label, "unknown")
                    })

            torch.cuda.empty_cache()
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    precision_per_class = precision_score(all_labels, all_preds, average=None, labels=[0, 1, 2, 3])
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2, 3])

    # matriz de confusion
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["neutral","happy","sad","angry"],
                yticklabels=["neutral","happy","sad","angry"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("D:/tesis/modelo de fusion/confusion_matrix.png")
    plt.close()

    return accuracy, f1, cm, all_filenames, all_labels, all_preds,incorrect_predictions, precision_per_class, f1_per_class

def main(sessions=["Session4"], batch_size=4):
    data_dirs = {
        "text_dir": "D:/tesis/data/text_tokenized",
        "audio_dir": "D:/tesis/data/audio_preprocessed",
        "video_dir": "D:/tesis/data/video_preprocessed",
        "label_dir": "D:/tesis/data/labels"
    }

    dataloader = get_multimodal_dataloader(**data_dirs, sessions=sessions, batch_size=batch_size, shuffle=False, max_waveform_length=160000)
    models = load_models()
    accuracy, f1, cm, filenames, labels, preds, incorrect_predictions,precision_per_class, f1_per_class = predict(dataloader, models, device)

    print(f"\nTotal samples: {len(filenames)}")
    print(f"Acc: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("\nmetrica por emocion:")
    emotions = [idx_to_emotion[i] for i in [0, 1, 2, 3]]
    for i, emotion in enumerate(emotions):
        print(f"Emotion: {emotion}")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  F1 Score: {f1_per_class[i]:.4f}")


if __name__ == "__main__":
    main(sessions=["Session5"], batch_size=4)
