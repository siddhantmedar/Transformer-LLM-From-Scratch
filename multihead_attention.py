import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_head):
        super(MultiHeadAttention, self).__init__()

        self.d_model = embed_size
        self.num_head = num_head
        self.d_head = self.d_model // num_head

        # Ensure the embedding size is divisible by the number of heads
        assert self.d_model % num_head == 0, "Embed size must be divisible by num_heads"

        # Linear layers to project the input into query, key, and value vectors
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)

        # Final linear layer to project the concatenated output
        self.fc_out = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        batch_size, seq_length, d_model = x.shape

        # Linear projections
        Q = self.query(x)  # (batch_size, seq_length, d_model)
        K = self.key(x)    # (batch_size, seq_length, d_model)
        V = self.value(x)  # (batch_size, seq_length, d_model)

        # Reshape and transpose for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_head, self.d_head).transpose(1, 2)  # (batch_size, num_head, seq_length, d_head)
        K = K.view(batch_size, seq_length, self.num_head, self.d_head).transpose(1, 2)  # (batch_size, num_head, seq_length, d_head)
        V = V.view(batch_size, seq_length, self.num_head, self.d_head).transpose(1, 2)  # (batch_size, num_head, seq_length, d_head)

        # Scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_head, seq_length, seq_length)
        scaling_factor = torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))
        energy = energy / scaling_factor
        attention = F.softmax(energy, dim=-1)  # (batch_size, num_head, seq_length, seq_length)
        output = torch.matmul(attention, V)    # (batch_size, num_head, seq_length, d_head)

        # Concatenate heads and put through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)  # (batch_size, seq_length, d_model)
        output = self.fc_out(output)  # (batch_size, seq_length, d_model)

        return output

if __name__ == "__main__":
    embedding_size = 128
    num_head = 8

    # Example input tensor
    x = torch.rand(2, 3, embedding_size)  # (batch_size, seq_length, embedding_size)
    
    model = MultiHeadAttention(embed_size=embedding_size, num_head=num_head)

    out = model(x)

    print(f"Multihead Attention Output Shape: {out.shape}")  # Expected output shape: (batch_size, seq_length, embedding_size)
