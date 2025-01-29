import torch
import torch.nn as nn
import math
from config import *
from prepare_dataset import TextDataset


class PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        
        pos = torch.arange(seq_length)
        i = torch.arange(d_model)

        div_term = torch.exp(pos * (-2/d_model) * math.log(10000))
        pe = torch.zeros(seq_length,d_model)

        pe[:,::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)
        
        self.register_buffer('pe',pe)
        
    def forward(self, x):
        return x + self.pe

if __name__ == "__main__":
    text_dataset = TextDataset()
    X,y = text_dataset.get_random_batch()
    
    positional_embed = PositionalEmbedding()
    output = positional_embed(X)
    print(X.shape)
    print(y.shape)

    # output = encoder(x)