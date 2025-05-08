import torch
import torch.nn as nn
import math

config = {
    "n_layers": 6,
    "n_vocab": 1000,
    "d_model": 128,
    "num_head": 4,
}


class Embeddding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = None

    def forward(self, x):

        x = self.token_embedding(x)  # (B,S) -> (B,S,D)
        batch_size, seq_length, d_model = x.shape

        if self.pos_embedding is None:
            self.pos_embedding = torch.zeros(batch_size, seq_length, d_model)

            for b in range(batch_size):
                for pos in range(seq_length):
                    for i in range(d_model):
                        if pos % 2 == 0:
                            self.pos_embedding[b, pos, i] = math.sin(
                                pos / 10000 ** ((2 * i) / d_model)
                            )
                        else:
                            self.pos_embedding[b, pos, i] = math.cos(
                                pos / 10000 ** ((2 * i) / d_model)
                            )

        return x + self.pos_embedding


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        assert (
            d_model % num_head == 0
        ), f"d_model is not divisible by num_head provided: {num_head}"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_model // num_head
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_length, d_model = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_length, self.num_head, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_head, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_head, self.d_head).transpose(1, 2)

        attn_weight = torch.softmax(
            (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_head), dim=-1
        )
        attn = (attn_weight @ V).transpose(1, 2)
        attn = attn.contiguous().view(
            batch_size, seq_length, self.num_head * self.d_head
        )

        attn = torch.tril(attn, diagonal=0)

        return self.norm(x + attn)


class FNN(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.fnn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.fnn(x)

        return self.norm(x + out)


class Decoder(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        self.mmha = MaskedMultiHeadAttention(d_model, num_head)
        self.fnn = FNN(d_model)

    def forward(self, x):
        return self.fnn(self.mmha(x))


class DecoderBlock(nn.Module):
    def __init__(self, n_layers, n_vocab, d_model, num_head):
        super().__init__()

        self.decoder_layers = nn.ModuleList(
            [Decoder(d_model, num_head) for _ in range(n_layers)]
        )
        self.fc = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        for layer in self.decoder_layers:
            x = layer(x)

        return torch.softmax(self.fc(x), dim=-1)


if __name__ == "__main__":
    embedding = Embeddding(config["n_vocab"], config["d_model"])
    out_embed = embedding(torch.randint(0, config["n_vocab"], (3, 5)))
    print(f"Embedding out shape: {out_embed.shape}")

    mmha = MaskedMultiHeadAttention(config["d_model"], config["num_head"])
    out_mmha = mmha(out_embed)
    print(f"MMHA out shape: {out_mmha.shape}")

    fnn = FNN(config["d_model"])
    out_fnn = fnn(out_mmha)
    print(f"FNN out: {out_fnn.shape}")

    decoder_block = DecoderBlock(
        config["n_layers"], config["n_vocab"], config["d_model"], config["num_head"]
    )
    out_decoder_block = decoder_block(out_embed)
    print(f"Decoder Block Output: {out_decoder_block.shape}")
