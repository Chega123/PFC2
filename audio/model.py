import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2VecEmotion(nn.Module):
    def __init__(self, pretrained_model="facebook/wav2vec2-base", num_classes=4):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x):
        x = x.squeeze(1)  # (batch, time)
        out = self.wav2vec(x)
        hidden_states = out.last_hidden_state  #(batch, seq_len, 768)
        attention_out, _ = self.attention(hidden_states, hidden_states, hidden_states)
        pooled = attention_out.mean(dim=1)  # (batch, 768)
        return pooled

    def forward(self, x):
        pooled = self.extract_features(x)
        return self.classifier(pooled)
