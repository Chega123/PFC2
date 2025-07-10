import torch
import torch.nn as nn

class AutoAttentionFusionModule(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.3):
        super(AutoAttentionFusionModule, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Final multimodal pooling layer
        self.pooling = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),  # 2304 -> 1536
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)  # 1536 -> 768
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, text_vector, audio_vector, video_vector):
        batch_size = text_vector.size(0)
        assert text_vector.dim() == 2 and text_vector.size(1) == self.embed_dim, \
            f"Text vector shape: {text_vector.shape}"
        assert audio_vector.dim() == 2 and audio_vector.size(1) == self.embed_dim, \
            f"Audio vector shape: {audio_vector.shape}"
        assert video_vector.dim() == 2 and video_vector.size(1) == self.embed_dim, \
            f"Video vector shape: {video_vector.shape}"

        text_vector = text_vector.unsqueeze(1)  # (batch_size, 1, embed_dim)
        audio_vector = audio_vector.unsqueeze(1) 
        video_vector = video_vector.unsqueeze(1)  

        combined = torch.cat((text_vector, audio_vector, video_vector), dim=1)

        # auto-atencion
        attn_output, _ = self.self_attention(combined, combined, combined)  # (batch_size, 3, embed_dim)
        if self.training: 
            attn_output = self.dropout(attn_output)
        attn_output = self.norm1(combined + attn_output)  # Residual connection

        # Concatenar
        pooled = attn_output.view(batch_size, -1)  

        #pooling
        output = self.pooling(pooled)
        output = self.norm2(output)

        return output  