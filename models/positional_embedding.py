import torch
import torch.nn as nn
import math
from config import *
from prepare_dataset import TextDataset


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

        pos = torch.arange(seq_length).unsqueeze(1)  # (seq_length,1)
        i = torch.arange(d_model // 2).unsqueeze(0)  # (1,d_model//2)

        div_term = torch.exp(i * (-math.log(10000) / (d_model // 2)))  # (1,d_model)
        pe = torch.zeros(seq_length, d_model)  # (seq_length,d_model)

        pe[:, ::2] = torch.sin(pos * div_term)  # (seq_length,d_model//2)
        pe[:, 1::2] = torch.cos(pos * div_term)  # (seq_length,d_model//2)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1,seq_length,d_model)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]
