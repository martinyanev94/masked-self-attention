"""
Minimal GPT-2 like model in PyTorch (causal transformer).
- Configurable model size
- Causal (autoregressive) attention
- Sampling/generation function
- Simple training loop skeleton (toy dataset)
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

from GPT.gpt2.chat import chat
from GPT.gpt2.data_builder import BPETokenizer, TextDataset


# -----------------------
# Config
# -----------------------
class GPTConfig:
    def __init__(self,
                 vocab_size=50257,
                 block_size=128,
                 n_layers=6,
                 n_heads=8,
                 n_embd=512,
                 dropout=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dropout = dropout
        self.device = device

# -----------------------
# Modules
# -----------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with softmax masking."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # register causal mask once (upper triangular) for efficiency
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        """
        x: (B, T, C)
        returns: (B, T, C)
        """
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)
        # reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, T, T)

        # causal mask: allow positions <= current position
        # mask shape (1,1,T,T) broadcast to (B,heads,T,T)
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)  # (B, heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y

class MLP(nn.Module):
    """Simple 2-layer MLP with GELU activation"""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block: LayerNorm -> Attention -> Add -> LayerNorm -> MLP -> Add"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------------
# GPT model
# -----------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T) token indices (for computing loss)
        returns: logits (B, T, V) and optionally loss
        """
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds model block size {self.block_size}"

        tok_embeddings = self.tok_emb(idx)          # (B, T, C)
        pos_embeddings = self.pos_emb[:, :T, :]     # (1, T, C)
        x = self.drop(tok_embeddings + pos_embeddings)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)

        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        device = next(self.parameters()).device

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)  # (B, T', V)
            logits = logits[:, -1, :] / max(1e-8, temperature)  # (B, V)

            # ---- FIX: Clamp top_k to vocabulary size ----
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))

                v, _ = torch.topk(logits, top_k)
                min_topk = v[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_topk,
                    torch.full_like(logits, float('-inf')),
                    logits
                )

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx



# -----------------------
# Training helper
# -----------------------
def train(model: GPT, data_loader, optimizer, device, epoch, max_grad_norm=1.0, scheduler=None, log_interval=100):
    model.train()
    total_loss = 0.0
    for step, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        if (step + 1) % log_interval == 0:
            avg = total_loss / log_interval
            print(f"Epoch {epoch} step {step+1}/{len(data_loader)} loss {avg:.4f}")
            total_loss = 0.0

# -----------------------
# Example usage (toy)
# -----------------------
def main():
    # Model config (reduce sizes for quick runs)
    config = GPTConfig()
    input_file = "wiki_datasets/wikisent_p1.txt"

    # Toy text (use larger dataset in practice)
    with open(input_file, "r", encoding="utf-8") as f:
        sample_text = f.read()

    tokenizer = BPETokenizer(input_file, vocab_size=config.vocab_size)
    print("Vocab size:", tokenizer.vocab_size)

    vocab = tokenizer.tokenizer.get_vocab()  # dict: token → id
    a = {}
    for token, idx in vocab.items():
        a[idx] = token

    sorted_d = {k: a[k] for k in sorted(a)}
    vocab_items = ""
    for key, value in list(sorted_d.items())[:50]:
        vocab_items += f"{key}: {value}, "

    print(vocab_items)
    block_size = 64


    model = GPT(config).to(config.device)

    # Dataset & DataLoader
    dataset = TextDataset(sample_text, tokenizer, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    # Optional: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    scheduler = None

    # Training loop (very small number of epochs for demo)
    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        train(model, dataloader, optimizer, config.device, epoch, scheduler=scheduler, log_interval=20)

        # sample text after each epoch
        model.eval()
        start = "To be"
        start_ids = torch.tensor([tokenizer.encode(start)], dtype=torch.long).to(config.device)
        generated = model.generate(start_ids, max_new_tokens=200, temperature=0.8, top_k=50)
        out = tokenizer.decode(generated[0].tolist())
        print(f"=== Sample after epoch {epoch} ===")
        print(out)
        print("="*60)

        # Save small checkpoint
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"mini_gpt_epoch{epoch}.pt"))

    # model.load_state_dict(torch.load("model.pth"))
    model.eval()

    chat(model, tokenizer)



if __name__ == "__main__":
    main()
