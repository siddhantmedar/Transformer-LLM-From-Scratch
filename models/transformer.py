import torch
import torch.nn as nn
from models.embedding import Embeddings
from models.encoder import Encoder
from config import *


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embeddings()
        self.encoder = Encoder()

    def forward(self, x):
        x = self.embedding(x)
        return self.encoder(x)
