import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
import math


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        batch_size, seq_length, _ = x.shape

        Q = self.W_q(x)  # (batch_size, seq_length, d_model)
        K = self.W_k(x)  # (batch_size, seq_length, d_model)
        V = self.W_v(x)  # (batch_size, seq_length, d_model)

        Q = Q.view(batch_size, seq_length, num_heads, d_k).transpose(
            1, 2
        )  # (batch, num_heads, seq_length, d_k)
        K = K.view(batch_size, seq_length, num_heads, d_k).transpose(
            1, 2
        )  # (batch, num_heads, seq_length, d_k)
        V = V.view(batch_size, seq_length, num_heads, d_k).transpose(
            1, 2
        )  # (batch, num_heads, seq_length, d_k)

        energy = torch.matmul(
            Q, K.transpose(-2, -1)
        )  # (batch_size, num_head, seq_length, seq_length)
        energy = energy / math.sqrt(d_k)
        attention = F.softmax(
            energy, dim=-1
        )  # (batch_size, num_head, seq_length, seq_length)
        output = torch.matmul(attention, V)  # (batch_size, num_head, seq_length, d_k)

        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        )  # (batch_size, seq_length, d_model)
        output = self.W_o(output)  # (batch_size, seq_length, d_model)

        return output
