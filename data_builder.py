from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset
import torch

# -----------------------
# Toy dataset (token-level) and tokenizer
# -----------------------
class BPETokenizer:
    def __init__(self, text_path, vocab_size=20000):
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>"]
        )

        # Train on your dataset
        self.tokenizer.train([text_path], trainer)

        # store important attributes
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)


class TextDataset(Dataset):
    """Create input-target sequences from raw text for causal LM training."""
    def __init__(self, data, tokenizer, block_size):
        self.data = data
        self.tok = tokenizer
        self.block_size = block_size
        self.arr = self.tok.encode(self.data)

    def __len__(self):
        return max(1, len(self.arr) - self.block_size)

    def __getitem__(self, idx):
        i = idx
        x = torch.tensor(self.arr[i:i+self.block_size], dtype=torch.long)
        y = torch.tensor(self.arr[i+1:i+1+self.block_size], dtype=torch.long)
        return x, y