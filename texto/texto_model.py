import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel


class RobertaEmbeddingExtractor(nn.Module):
    def __init__(self, pretrained_model="roberta-base", hidden_dropout=0.3, attn_dropout=0.3, num_frozen_layers=0, num_labels=None):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        if num_frozen_layers > 0:
            for param in self.roberta.embeddings.parameters():
                param.requires_grad = False
            for idx, layer in enumerate(self.roberta.encoder.layer):
                if idx < num_frozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, dropout=attn_dropout, batch_first=True)
        self.hidden_dropout = nn.Dropout(hidden_dropout)

        # Self-attentive pooling
        self.att_pool = nn.Linear(768, 1)

        # Clasificador opcional
        self.classifier = None
        if num_labels is not None:
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(hidden_dropout),
                nn.Linear(256, num_labels)
            )

    def extract_features(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)

        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        attn_output = self.hidden_dropout(attn_output)

        # Self-attentive pooling
        weights = torch.softmax(self.att_pool(attn_output), dim=1)  # (batch, seq_len, 1)
        pooled = torch.sum(weights * attn_output, dim=1)            # (batch, 768)

        return self.hidden_dropout(pooled)

    def forward(self, input_ids, attention_mask):
        pooled = self.extract_features(input_ids, attention_mask)
        if self.classifier:
            return self.classifier(pooled)
        return pooled


def get_tokenizer_and_model(
    model_name: str = "roberta-base",
    num_labels: int = 4,
    dropout: float = 0.1,
    num_frozen_layers: int = 0,
    device: torch.device = None,
    return_classifier: bool = True
):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    if return_classifier:
        model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        if num_frozen_layers > 0:
            for idx, layer in enumerate(model.roberta.encoder.layer):
                if idx < num_frozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
    else:
        model = RobertaEmbeddingExtractor(
            pretrained_model=model_name,
            hidden_dropout=dropout,
            attn_dropout=dropout,
            num_frozen_layers=num_frozen_layers,
            num_labels=num_labels   # opcional
        )

    if device:
        model.to(device)

    return tokenizer, model
