import torch
import torch.nn as nn

class AutoAttentionFusionModule(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.3):
        super(AutoAttentionFusionModule, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Final multimodal pooling layer
        self.pooling = nn.Linear(embed_dim * 3, embed_dim)  # Combines all three modalities
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, text_vector, audio_vector, video_vector):
        # Ensure inputs are of shape (batch_size, 1, embed_dim)
        batch_size = text_vector.size(0)
        text_vector = text_vector.unsqueeze(1)  # (batch_size, 1, embed_dim)
        audio_vector = audio_vector.unsqueeze(1)  # (batch_size, 1, embed_dim)
        video_vector = video_vector.unsqueeze(1)  # (batch_size, 1, embed_dim)

        # Concatenate the vectors
        combined = torch.cat((text_vector, audio_vector, video_vector), dim=1)  # (batch_size, 3, embed_dim)

        # Apply self-attention
        attn_output, _ = self.self_attention(combined, combined, combined)
        attn_output = self.norm1(combined + self.dropout(attn_output))  # Residual connection

        # Pool the attended vectors into a single 768-dimensional vector
        pooled = attn_output.mean(dim=1)  # (batch_size, embed_dim)
        output = self.pooling(pooled)
        output = self.norm2(output)

        return output  # (batch_size, embed_dim) i.e., 768-dimensional vector

# Example usage (can be adjusted based on your input pipeline)
def fuse_with_auto_attention(text_emb, audio_emb, video_emb):
    fusion_module = AutoAttentionFusionModule(embed_dim=768)
    fused_vector = fusion_module(text_emb, audio_emb, video_emb)
    return fused_vector