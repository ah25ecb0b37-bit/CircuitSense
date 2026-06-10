"""
CircuitSense — Multi-Head Self-Attention over LSTM outputs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, H)
        attn_out, attn_weights = self.attn(x, x, x)
        out = self.norm(x + attn_out)
        # Global average pool over time
        context = out.mean(dim=1)  # (B, H)
        return context, attn_weights
