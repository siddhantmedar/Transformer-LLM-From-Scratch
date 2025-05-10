import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import time
import sys
from torch.cuda.amp import autocast, GradScaler

# Device definition
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
if device.type == "mps":
    print("Using MPS device. Some operations may have limitations.")

# Load and prepare data
dataset_dir = "dataset"
dataset_file = os.path.join(dataset_dir, "tiny_shakespeare.txt")
try:
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directory '{dataset_dir}' does not exist")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"File '{dataset_file}' does not exist")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        text = f.read()
except (FileNotFoundError, PermissionError, UnicodeDecodeError, IOError) as e:
    print(f"Error loading dataset: {e}")
    raise

if not text.strip():
    print("Error: 'dataset/tiny_shakespeare.txt' is empty")
    raise ValueError("Dataset file is empty")

# Train BPE tokenizer
tokenizer_dir = "tokenizer"
tokenizer_path = os.path.join(tokenizer_dir, "bpe_tokenizer.json")
expected_vocab_size = 20000
try:
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)
    if not os.access(tokenizer_dir, os.W_OK):
        raise PermissionError(f"Directory '{tokenizer_dir}' is not writable")
    if not os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=expected_vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        )
        start_time = time.time()
        tokenizer.train([dataset_file], trainer)
        print(f"Tokenizer training took {time.time() - start_time:.2f} seconds")
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        if tokenizer.get_vocab_size() != expected_vocab_size:
            print(f"Warning: Loaded tokenizer has vocab_size {tokenizer.get_vocab_size()}, expected {expected_vocab_size}. Retraining tokenizer.")
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(
                vocab_size=expected_vocab_size,
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
            )
            start_time = time.time()
            tokenizer.train([dataset_file], trainer)
            print(f"Tokenizer training took {time.time() - start_time:.2f} seconds")
            tokenizer.save(tokenizer_path)
except (FileNotFoundError, PermissionError, ValueError) as e:
    print(f"Error with tokenizer: {e}")
    raise

# Get vocabulary size
vocab_size = tokenizer.get_vocab_size()

def encode(s):
    return tokenizer.encode(s).ids

def decode(l):
    return tokenizer.decode(l)

# Check [UNK] frequency
sample_text = text[:min(10000, len(text))]
encoded = encode(sample_text)
unk_id = tokenizer.token_to_id("[UNK]")
unk_count = encoded.count(unk_id)
if encoded:
    print(f"UNK tokens: {unk_count} ({unk_count/len(encoded)*100:.2f}%)")
    if unk_count / len(encoded) > 0.01:
        print("Warning: High [UNK] frequency. Consider increasing vocab_size.")
else:
    print("Warning: Sample text too short for [UNK] frequency check")

# Encode dataset
data = torch.tensor(encode(text), dtype=torch.long).to(device)

# Check dataset size
max_data_size = 1e8  # Max 100M tokens
if len(data) > max_data_size:
    print(f"Warning: Dataset size ({len(data)} tokens) exceeds recommended limit ({max_data_size}). Consider subsampling.")

@dataclass
class Config:
    n_layers: int = 6
    n_vocab: int = vocab_size
    d_model: int = 128
    num_head: int = 4
    max_seq_length: int = 512
    block_size: int = 256
    batches_per_epoch: int = 100

config = Config()

# Adjust block_size for small datasets
if len(data) <= 1:
    print(f"Error: Dataset too small ({len(data)} tokens). Minimum length is 2 tokens.")
    raise ValueError("Dataset too small")
if len(data) < config.block_size:
    print(f"Warning: Dataset length ({len(data)}) is shorter than block_size ({config.block_size}). Adjusting block_size.")
    config.block_size = min(config.block_size, len(data) - 1)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size, block_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = []
    y = []
    pad_id = tokenizer.token_to_id("[PAD]")
    for i in ix:
        seq_x = data[i:i+block_size]
        seq_y = data[i+1:i+block_size+1]
        if len(seq_x) < block_size:
            seq_x = torch.cat([seq_x, torch.full((block_size - len(seq_x),), pad_id, dtype=torch.long, device=device)])
            seq_y = torch.cat([seq_y, torch.full((block_size - len(seq_y),), pad_id, dtype=torch.long, device=device)])
        x.append(seq_x)
        y.append(seq_y)
    return torch.stack(x).to(device), torch.stack(y).to(device)

class Embeddding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pos_embedding', pe)

    def forward(self, x):
        batch_size, seq_length = x.shape
        x = self.token_embedding(x)
        x = x + self.pos_embedding[:seq_length, :].unsqueeze(0)
        return self.dropout(x)

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super().__init__()
        assert d_model % num_head == 0
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_model // num_head
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, d_model = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(batch_size, seq_length, self.num_head, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_head, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_head, self.d_head).transpose(1, 2)
        attn_scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weight = torch.softmax(attn_scores, dim=-1)
        attn_weight = self.dropout(attn_weight)
        attn = (attn_weight @ V).transpose(1, 2)
        attn = attn.contiguous().view(batch_size, seq_length, d_model)
        return self.norm(x + attn)

class FNN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.fnn(x)
        return self.norm(x + out)

class Decoder(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super().__init__()
        self.mmha = MaskedMultiHeadAttention(d_model, num_head, dropout)
        self.fnn = FNN(d_model, dropout)

    def forward(self, x):
        return self.fnn(self.mmha(x))

class DecoderBlock(nn.Module):
    def __init__(self, n_layers, n_vocab, d_model, num_head, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = Embeddding(n_vocab, d_model, max_seq_len, dropout)
        self.decoder_layers = nn.ModuleList(
            [Decoder(d_model, num_head, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.norm(x)
        return self.fc(x)

def train(model, config, train_data, val_data, device, tokenizer, epochs=20, batch_size=64, patience=3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    scaler = GradScaler()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    checkpoint_dir = "model_checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for _ in range(config.batches_per_epoch):
            x, y = get_batch('train', batch_size, config.block_size)
            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = criterion(logits.view(-1, config.n_vocab), y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / config.batches_per_epoch

        total_val_loss = 0
        model.eval()
        valid_val_loss = len(val_data) > config.block_size
        if valid_val_loss:
            with torch.no_grad():
                for _ in range(config.batches_per_epoch // 10):
                    x, y = get_batch('val', batch_size, config.block_size)
                    with autocast():
                        logits = model(x)
                        loss = criterion(logits.view(-1, config.n_vocab), y.view(-1))
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / (config.batches_per_epoch // 10)
            perplexity = math.exp(avg_val_loss)
            scheduler.step(avg_val_loss)
        else:
            print("Warning: Validation data too short or empty, skipping validation")
            avg_val_loss = float('inf')
            perplexity = float('inf')

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}" +
              (f", Val Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}" if valid_val_loss else ""))

        if valid_val_loss and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                if not os.access(checkpoint_dir, os.W_OK):
                    raise PermissionError(f"Directory '{checkpoint_dir}' is not writable")
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved best model with Val Loss: {best_val_loss:.4f}")
            except (FileNotFoundError, PermissionError, OSError) as e:
                print(f"Error saving model checkpoint: {e}")
                raise
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

def generate(model, prompt, max_tokens=50, temperature=1.0, return_ids=False):
    if not prompt.strip():
        print("Warning: Empty prompt provided, using default")
        prompt = "[CLS]"
    model.eval()
    tokens = encode(prompt)
    if not tokens:
        tokens = [tokenizer.token_to_id("[CLS]")]
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_tokens):
        input_tokens = tokens[:, -config.block_size:]
        with autocast():
            logits = model(input_tokens)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    token_ids = tokens[0].tolist()
    if return_ids:
        return token_ids
    return decode(token_ids)

if __name__ == "__main__":
    try:
        model = DecoderBlock(
            config.n_layers, config.n_vocab, config.d_model, config.num_head, config.max_seq_length, dropout=0.1
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model size: {total_params:,} parameters ({total_params / 1e6:.2f}M)")
        train(model, config, train_data, val_data, device, tokenizer, patience=3)
        print(generate(model, "Music", max_tokens=50))
    except Exception as e:
        print(f"Error during execution: {e}")
        raise