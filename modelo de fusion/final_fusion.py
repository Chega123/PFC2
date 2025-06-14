import torch
import torch.nn as nn

class FinalFusionMLP(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.3):
        super(FinalFusionMLP, self).__init__()
        self.gate = nn.Linear(embed_dim * 2, 2)  # Gated fusion weights
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4)  # Assuming 4 classes for emotion prediction
        )

    def forward(self, v1, v2):
        # Combine v1 and v2 into a matrix V = [v1; v2]
        V = torch.cat((v1.unsqueeze(1), v2.unsqueeze(1)), dim=1)  # (batch_size, 2, embed_dim)

        # Gated fusion to compute weights w1 and w2
        gates = torch.softmax(self.gate(V.view(-1, 2 * self.embed_dim)), dim=-1)  # (batch_size, 2)
        w1, w2 = gates[:, 0], gates[:, 1]

        # Weighted combination
        vf = w1.unsqueeze(-1) * v1 + w2.unsqueeze(-1) * v2  # (batch_size, embed_dim)
        vf = self.norm(vf)
        vf = self.dropout(vf)

        # MLP for final prediction
        output = self.mlp(vf)  # (batch_size, num_classes)

        return output  # Final prediction y_hat

# Example usage
def predict_emotion(v1, v2):
    fusion_mlp = FinalFusionMLP(embed_dim=768)
    prediction = fusion_mlp(v1, v2)
    return prediction