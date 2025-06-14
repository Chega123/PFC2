import sys
import os
import torch
from transformers import RobertaTokenizer
from hierarchical import HierarchicalFusionModule
from auto_attention import AutoAttentionFusionModule
from final_fusion import FinalFusionMLP

# Add the directory containing your model.py files to the system path
sys.path.append('D:\\tesis\\audio')
sys.path.append('D:\\tesis\\texto')  # Assuming texto_model.py is in D:\tesis\texto
sys.path.append('D:\\tesis\\video')  # Assuming video_model.py is in D:\tesis\video

# Import models
from audio_model import Wav2VecEmbeddingExtractor  
from texto_model import RobertaEmbeddingExtractor, get_tokenizer_and_model  # Adjust path if needed
from video_model import VideoEmbeddingExtractor  # Adjust path if needed

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained models and weights
# Text Model
text_tokenizer, text_model = get_tokenizer_and_model(
    model_name="roberta-base", return_classifier=False, device=device
)
text_state_dict = torch.load("D:\\tesis\\text_model.pth", map_location=device)
text_model.load_state_dict(text_state_dict)

# Audio Model
audio_model = Wav2VecEmbeddingExtractor(pretrained_model="facebook/wav2vec2-base", dropout=0.3, num_frozen_layers=0)
audio_state_dict = torch.load("D:\\tesis\\audio\\audio_model.pth", map_location=device)
audio_model.load_state_dict(audio_state_dict)

# Video Model
video_model = VideoEmbeddingExtractor(hidden_size=768, num_layers=1, dropout=0.0, num_frozen_layers=0)
video_state_dict = torch.load("D:\\tesis\\video\\video_model.pth", map_location=device)
video_model.load_state_dict(video_state_dict)

# Move models to device
text_model.to(device)
audio_model.to(device)
video_model.to(device)

# Load fusion modules
hierarchical_fusion = HierarchicalFusionModule(embed_dim=768).to(device)
auto_attention_fusion = AutoAttentionFusionModule(embed_dim=768).to(device)
final_mlp = FinalFusionMLP(embed_dim=768).to(device)

# Set models to evaluation mode
text_model.eval()
audio_model.eval()
video_model.eval()
hierarchical_fusion.eval()
auto_attention_fusion.eval()
final_mlp.eval()

# Example input data (replace with your actual data)
batch_size = 1
text_input = "Sample emotional text"
audio_input = torch.randn(batch_size, 1, 16000).to(device)  # Example audio waveform
video_input = torch.randn(batch_size, 16, 3, 224, 224).to(device)  # Example video frames

# Extract embeddings
# Text
text_inputs = text_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
text_emb = text_model.extract_features(text_inputs["input_ids"], text_inputs["attention_mask"])

# Audio
audio_emb = audio_model.extract_features(audio_input)

# Video
video_emb = video_model.extract_features(video_input)

# Hierarchical Fusion (v1)
v1 = hierarchical_fusion(text_emb, audio_emb, video_emb)

# Auto-Attention Fusion (v2)
v2 = auto_attention_fusion(text_emb, audio_emb, video_emb)

# Final Prediction
with torch.no_grad():
    prediction = final_mlp(v1, v2)

print("Final Prediction:", prediction)

# Optional: Print shapes for debugging
print("Text Embedding Shape:", text_emb.shape)
print("Audio Embedding Shape:", audio_emb.shape)
print("Video Embedding Shape:", video_emb.shape)
print("v1 Shape:", v1.shape)
print("v2 Shape:", v2.shape)
print("Prediction Shape:", prediction.shape)