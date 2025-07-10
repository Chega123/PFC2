import torch
import torch.nn as nn

class HierarchicalFusionModule(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.3):
        super(HierarchicalFusionModule, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Cross-attention 
        self.cross_attention_text_audio = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention_text_video = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Attention pooling
        self.attn_pooling = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, text_vector, audio_vector, video_vector):
        batch_size = text_vector.size(0)
        text_vector = text_vector.unsqueeze(1)
        audio_vector = audio_vector.unsqueeze(1)
        video_vector = video_vector.unsqueeze(1)
        
        attn_output_text_audio, _ = self.cross_attention_text_audio(
            query=text_vector, key=audio_vector, value=audio_vector
        )
        enriched_text = text_vector + attn_output_text_audio
        
        attn_output_text_video, _ = self.cross_attention_text_video(
            query=enriched_text, key=video_vector, value=video_vector
        )
        enriched_text = enriched_text + attn_output_text_video
        
        attn_weights = torch.softmax(self.attn_pooling(enriched_text), dim=1)
        pooled = (enriched_text * attn_weights).sum(dim=1)
        if self.training:  # Solo aplica dropout durante entrenamiento
            pooled = self.dropout(pooled)
        output = self.norm(pooled)
        
        return output