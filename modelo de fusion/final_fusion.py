import torch
import numpy as np
import random
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
import torch.nn as nn

class FinalFusionMLP(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.3):
        super(FinalFusionMLP, self).__init__()
        self.embed_dim = embed_dim
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4) 
        )

    def forward(self, v1, v2):
        V = torch.cat((v1.unsqueeze(1), v2.unsqueeze(1)), dim=1) 

        # Gated fusion 
        gates = torch.softmax(self.gate(V.view(-1, 2 * self.embed_dim)), dim=-1) 
        w1, w2 = gates[:, 0], gates[:, 1]

        # combination
        vf = w1.unsqueeze(-1) * v1 + w2.unsqueeze(-1) * v2  
        vf = self.norm(vf)
        if self.training:  
            vf = self.dropout(vf)

        output = self.mlp(vf)  

        return output  