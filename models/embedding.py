import torch
import torch.nn as nn
import math
from config import *

from models.token_embedding import InputEmbedding
from models.positional_embedding import PositionalEncoding


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = InputEmbedding()
        self.position_embedding = PositionalEncoding()

    def forward(self, x):
        return self.token_embedding(x) + self.position_embedding(x)
