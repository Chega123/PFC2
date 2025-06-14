import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel


class RobertaEmbeddingExtractor(nn.Module):
    def __init__(self, pretrained_model="roberta-base", dropout=0.3, num_frozen_layers=0):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model)

        if num_frozen_layers > 0:
            for idx, layer in enumerate(self.roberta.encoder.layer):
                if idx < num_frozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def extract_features(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        pooled = attn_output.mean(dim=1)  # (batch, 768)
        return self.dropout(pooled)

    def forward(self, input_ids, attention_mask):
        return self.extract_features(input_ids, attention_mask)


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
            dropout=dropout,
            num_frozen_layers=num_frozen_layers
        )

    if device:
        model.to(device)

    return tokenizer, model
