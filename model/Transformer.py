import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=336):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.scale = nn.Parameter(torch.ones(1) * 0.02)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).expand(x.size(0), -1)
        pos_emb = self.pos_embed(positions) * self.scale
        return x + pos_emb

class Transformer(nn.Module):
    def __init__(self, input_dim=7, d_model=64, nhead=4, num_layers=2, output_size=24):
        super().__init__()
        self.feature_embed = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = LearnablePositionalEncoding(d_model)
        
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, 
            dropout=0.2, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        x_embed = self.feature_embed(x)  # [Batch, len, d_model]
        
        x = self.pos_encoder(x_embed)
        
        memory = self.encoder(x)         # [Batch, len, d_model]
        
        last_hidden = memory[:, -1, :]    # [Batch, d_model]
        return self.decoder(last_hidden)  # [Batch, output_size]