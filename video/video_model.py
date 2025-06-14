
import torch
import torch.nn as nn
from transformers import ViTModel

class VideoEmbeddingExtractor(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_frozen_layers: int = 0,
        pretrained_vit: str = "google/vit-base-patch16-224-in21k"
    ):
        super(VideoEmbeddingExtractor, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_vit)
        vit_hidden_size = self.vit.config.hidden_size  # Typically 768
        self.gru = nn.GRU(
            input_size=vit_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0 if num_layers == 1 else dropout  # Dropout if num_layers > 1
        )
        self.dropout = nn.Dropout(dropout)
        self.embedding_projection = nn.Linear(hidden_size, 768)  # Projection to 768 dimensions
        self.hidden_size = hidden_size
        self.num_frozen_layers = num_frozen_layers
        self.freeze_layers(num_frozen_layers)

    def freeze_layers(self, num_frozen_layers: int):
        if num_frozen_layers > 0:
            for param in self.vit.embeddings.parameters():
                param.requires_grad = False
            max_layers = len(self.vit.encoder.layer)
            num_frozen = min(num_frozen_layers, max_layers)
            for layer in self.vit.encoder.layer[:num_frozen]:
                for param in layer.parameters():
                    param.requires_grad = False

    def extract_features(self, pixel_values: torch.Tensor):
        batch_size, num_frames, C, H, W = pixel_values.shape
        x = pixel_values.view(-1, C, H, W)  # (B*T, 3, 224, 224)
        outputs = self.vit(x)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # (B*T, 768)
        cls_seq = cls_tokens.view(batch_size, num_frames, -1)  # (B, T, 768)
        gru_out, _ = self.gru(cls_seq)  # (B, T, hidden_size)
        final_hidden = gru_out[:, -1, :]  # (B, hidden_size)
        embeddings = self.embedding_projection(final_hidden)  # (B, 768)
        return self.dropout(embeddings)

    def forward(self, pixel_values: torch.Tensor):
        return self.extract_features(pixel_values)

class VideoEmotionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 768,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_frozen_layers: int = 0,
        pretrained_vit: str = "google/vit-base-patch16-224-in21k"
    ):
        super(VideoEmotionClassifier, self).__init__()
        self.embedding = VideoEmbeddingExtractor(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_frozen_layers=num_frozen_layers,
            pretrained_vit=pretrained_vit
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values: torch.Tensor):
        pooled = self.embedding(pixel_values)  # (batch_size, 768)
        return self.classifier(pooled)