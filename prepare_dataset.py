import torch
import torch.nn as nn
import re
import tiktoken
import random

from config import * 

random.seed(seed)

class TextDataset:
    def __init__(self, input_file='dataset/data.txt', encoding="cl100k_base", seed=313):
        super().__init__()
        
        self.encoder = tiktoken.get_encoding(encoding)

        with open(input_file, 'r') as file:
            text = file.read()

        self.data = self.preprocess_and_encode_text(text)

    def replace_multiple_newlines(self, text):
        return re.sub(r'\n+', '\n', text)

    def remove_bracketed_text(self, text):
        return re.sub(r'\[\s*[^]]+\]', '', text, flags=re.DOTALL)

    def preprocess_and_encode_text(self, text, return_tensor=False):
        text = self.remove_bracketed_text(text)
        text = self.replace_multiple_newlines(text)
        encoded_text = self.encoder.encode(text)

        return torch.tensor(encoded_text) if return_tensor else encoded_text

    def get_random_batch(self):
        random_indices = random.sample(range(len(self.data)), batch_size)
        X, y = [], []

        for idx in random_indices:
            while idx + seq_length >= len(self.data):
                idx = random.sample(range(len(self.data)), 1)[0]

            X.append(self.data[idx:idx + seq_length])
            y.append(self.data[idx + 1:idx + seq_length + 1])
        
        return torch.tensor(X), torch.tensor(y)