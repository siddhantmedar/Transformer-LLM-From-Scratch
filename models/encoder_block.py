import torch
import torch.nn as nn
from attention import MultiHeadAttention
from config import *


class EncoderBlock(nn.Module):
    def __init__(self, dropout=dropout):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=True)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout=dropout)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=True)

    def forward(self, x):
        out = x + self.multi_head_attention(x)
        out = self.layer_norm1(out)

        return self.layer_norm2(out + self.fc2(self.dropout(self.gelu(self.fc1(out)))))
