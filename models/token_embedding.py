import torch
import torch.nn as nn
from config import *


class InputEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(n_vocab, d_model)

    def forward(self, x):
        return self.embedding_table(x)
