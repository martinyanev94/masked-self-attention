# Masked Self Attention – Minimal GPT-2 Style Model in Pure PyTorch

![17be2326-34b0-444a-bdfc-afe4bfa6242c](https://github.com/user-attachments/assets/590fddd3-093a-4bd1-ad54-fb5baa6132e5)


A lightweight, from-scratch implementation of a GPT-2-like causal transformer in PyTorch. Perfect for learning, experimentation, or as a starting point for your own models.

## Features

- Clean, well-commented causal (autoregressive) transformer architecture  
- Configurable model size (`n_layers`, `n_heads`, `n_embd`, `block_size`, etc.)  
- Multi-head self-attention with proper causal masking  
- GELU activation + LayerNorm (OpenAI-style)  
- Built-in text generation with temperature and top-k sampling  
- Simple training loop with gradient clipping and AdamW  
- BPE tokenizer (via `GPT/gpt2/data_builder.py`)  
- Interactive chat mode after training  

All in ~300 lines of core model code – no external transformer libraries required.

## Project Structure

- GPT/
  - wikidatasets/
    - wikisent_p1.py      # 1 percent of wiki data
    - wikisent_p001.py    # 0.1 percent of wiki data
- chat.py                   # Interactive chat loop
- data_builder.py           # BPETokenizer + TextDataset
- model.py                  # Full script (model + training + generation)
- checkpoints/              # Saved model weights (created automatically)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/martinyanev94/masked-self-attention.git
cd masked-self-attention

# Install dependencies
pip install torch tqdm

# Run training + generation + chat
python main.py
```

The script will:

- Train a small GPT on the text file
- Print generated samples after each epoch
- Save checkpoints to ./checkpoints/
- Launch an interactive chat session when training finishes

## Configuration

Edit the GPTConfig in main.py to change model size:

```python
config = GPTConfig(
    vocab_size=50257,
    block_size=128,    # max sequence length
    n_layers=6,        # number of transformer blocks
    n_heads=8,
    n_embd=512,
    dropout=0.1,
)
```

Smaller/faster config example:

```python
config = GPTConfig(n_layers=4, n_heads=4, n_embd=256, block_size=64)
```

## Generation Example

```python
start = "Once upon a time"
input_ids = torch.tensor([tokenizer.encode(start)]).to(device)
sample = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
print(tokenizer.decode(sample[0].tolist()))
```
