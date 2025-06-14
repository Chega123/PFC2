import torch
import torch.nn as nn

class HierarchicalFusionModule(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.3):
        super(HierarchicalFusionModule, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Cross-attention layers
        self.cross_attention_text_audio = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention_text_video = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Attention pooling
        self.attn_pooling = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, text_vector, audio_vector, video_vector):
        # Ensure inputs are of shape (batch_size, 1, embed_dim)
        batch_size = text_vector.size(0)
        text_vector = text_vector.unsqueeze(1)  # (batch_size, 1, embed_dim)
        audio_vector = audio_vector.unsqueeze(1)  # (batch_size, 1, embed_dim)
        video_vector = video_vector.unsqueeze(1)  # (batch_size, 1, embed_dim)

        # Step 1: Enrich text with audio using cross-attention
        attn_output_text_audio, _ = self.cross_attention_text_audio(
            query=text_vector, key=audio_vector, value=audio_vector
        )
        enriched_text = text_vector + attn_output_text_audio  # Residual connection

        # Step 2: Enrich enriched text with video using cross-attention
        attn_output_text_video, _ = self.cross_attention_text_video(
            query=enriched_text, key=video_vector, value=video_vector
        )
        enriched_text = enriched_text + attn_output_text_video  # Residual connection

        # Step 3: Attention pooling to get a single vector
        attn_weights = torch.softmax(self.attn_pooling(enriched_text), dim=1)  # (batch_size, 1, 1)
        pooled = (enriched_text * attn_weights).sum(dim=1)  # (batch_size, embed_dim)
        output = self.dropout(pooled)
        output = self.norm(output)

        return output  # (batch_size, embed_dim) i.e., 768-dimensional vector v1