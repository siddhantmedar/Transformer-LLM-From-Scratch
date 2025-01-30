import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import Transformer
from config import *

# TODO: replace with the actual training data
sample_data = torch.randint(
    0, n_vocab, (batch_size, seq_length)
)  # Random tokenized input

# Initialize model
model = Transformer()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(10):  # Example: 10 epochs
    optimizer.zero_grad()
    output = model(sample_data)
    loss = criterion(
        output.view(-1, n_vocab), sample_data.view(-1)
    )  # Example loss calculation
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
