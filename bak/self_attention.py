import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()

        self.d_model = embed_size

        # Linear layers for query, key, and value
        self.query = nn.Linear(self.d_model, self.d_model)  # (d_model, d_model)
        self.key = nn.Linear(self.d_model, self.d_model)  # (d_model, d_model)
        self.value = nn.Linear(self.d_model, self.d_model)  # (d_model, d_model)

        # Output linear layer
        self.fc_out = nn.Linear(self.d_model, self.d_model)  # (d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, dim = x.shape  # (batch_size, seq_length, d_model)

        # Apply linear layers to input x
        Q = self.query(x)  # (batch_size, seq_length, d_model)
        K = self.key(x)  # (batch_size, seq_length, d_model)
        V = self.value(x)  # (batch_size, seq_length, d_model)

        # Compute energy scores
        energy = torch.matmul(
            Q, K.transpose(-2, -1)
        )  # (batch_size, seq_length, seq_length)
        scaling_factor = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        energy = energy / scaling_factor  # (batch_size, seq_length, seq_length)

        # Apply softmax to get attention weights
        attention = F.softmax(energy, dim=-1)  # (batch_size, seq_length, seq_length)

        # Compute the output
        output = torch.matmul(attention, V)  # (batch_size, seq_length, d_model)

        # Apply final linear layer
        output = self.fc_out(output)  # (batch_size, seq_length, d_model)

        return output


if __name__ == "__main__":
    embedding_size = 5
    x = torch.rand(2, 3, embedding_size)  # (batch_size=2, seq_length=3, d_model=5)

    model = SelfAttention(embed_size=embedding_size)

    out = model(x)

    print(
        f"Self Attention Output Shape: {out.shape}"
    )  # (batch_size=2, seq_length=3, d_model=5)
