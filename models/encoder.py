import torch
import torch.nn as nn
from models.encoder_block import EncoderBlock
from config import *


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([EncoderBlock() for _ in range(num_layers)])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
