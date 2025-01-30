from prepare_dataset import TextDataset
from input_embedding import InputEmbedding
from positional_encoding import PositionalEncoding


if __name__ == "__main__":
    text_dataset = TextDataset()
    token_embedding = InputEmbedding()
    positional_embedding = PositionalEncoding()

    # get the data
    X, y = text_dataset.get_random_batch()

    # text embed
    X = token_embedding(X)
    print(f"Shape after adding token embedding: {X.shape}")

    # position embed
    X = positional_embedding(X)
    print(f"Shape after adding positional embedding: {X.shape}")
