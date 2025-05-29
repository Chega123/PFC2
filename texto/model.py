import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
import torch.nn as nn


def get_tokenizer_and_model(model_name: str = "roberta-base", num_labels: int = 4, device: torch.device = None):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if device:
        model.to(device)
    return tokenizer, model



