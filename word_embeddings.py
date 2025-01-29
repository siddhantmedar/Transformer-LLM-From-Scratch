import torch
import torch.nn as nn

from config import *
from prepare_dataset import TextDataset


class TextEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding()

    def forward(self, x):
        return self.embedding_table(x)


if __name__ == "__main__":
    text_dataset = TextDataset()
    X, y = text_dataset.get_random_batch()

    print(X.shape)
    print(y.shape)

    # output = encoder(x)
