import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2VecEmbeddingExtractor(nn.Module):
    def __init__(self, pretrained_model="facebook/wav2vec2-base", dropout=0.3, num_frozen_layers=0):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
        self.freeze_layers(num_frozen_layers)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def freeze_layers(self, num_frozen_layers):
        if num_frozen_layers > 0:
            for param in self.wav2vec.feature_extractor.parameters():
                param.requires_grad = False
        encoder_layers = self.wav2vec.encoder.layers
        for i in range(min(num_frozen_layers, len(encoder_layers))):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False

    def extract_features(self, x):
        x = x.squeeze(1)
        out = self.wav2vec(x)
        hidden_states = out.last_hidden_state
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        pooled = attn_output.mean(dim=1)
        return self.dropout(pooled)

    def forward(self, x):
        return self.extract_features(x)


class Wav2VecEmotionClassifier(nn.Module):
    def __init__(self, pretrained_model="facebook/wav2vec2-base", num_classes=4, dropout=0.3, num_frozen_layers=0):
        super().__init__()
        self.embedding = Wav2VecEmbeddingExtractor(pretrained_model, dropout, num_frozen_layers)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        pooled = self.embedding(x)
        return self.classifier(pooled)
