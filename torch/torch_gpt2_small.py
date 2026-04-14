import os
import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

load_dotenv()


# ======================== GPT-2 small hyper-parameters ========================
# Mirrors create_gpt2_small() in example_models.cpp
SEQ_LEN    = 1024
VOCAB_SIZE = 50257
EMBED_DIM  = 768
NUM_HEADS  = 12
NUM_LAYERS = 12
FFN_DIM    = EMBED_DIM * 4   # 3072
DROPOUT    = 0.1


# ======================== Dataset ========================

class OpenWebTextDataset(Dataset):
    """
    Streams tokenised OpenWebText from a flat uint16 binary file produced by
    python/openwebtext.py.

    Every item is a (input, target) pair of length SEQ_LEN where
    target = input shifted one position to the right (standard CLM).
    
    Uses memory-mapped file access to avoid loading the entire dataset into RAM.
    """
    def __init__(self, path: str, seq_len: int = SEQ_LEN):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        self.seq_len = seq_len
        # Keep as memmap - do NOT convert to torch tensor to avoid loading into memory
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        # number of full windows of size seq_len+1
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.seq_len
        # Only load the specific chunk needed, then convert to torch tensor
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        chunk = torch.from_numpy(chunk)
        x = chunk[:-1]   # [SEQ_LEN]
        y = chunk[1:]    # [SEQ_LEN]
        return x, y


# ======================== Model ========================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    Matches the attention layer built inside gpt_block() (is_causal=true).
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.embed_dim = embed_dim

        # Fused QKV projection  (matches a single dense layer typical of GPT-2)
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.embed_dim, dim=2)

        # reshape to (B, num_heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale           # (B, H, T, T)
        # causal mask: upper triangle = -inf
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v                                      # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        return out


class GPTBlock(nn.Module):
    """
    One GPT-2 transformer block matching gpt_block() in layer_builder.hpp:

    Attention sub-block  (ResidualBlock with "linear" / identity post-activation):
      x = x + Dropout(Attention(LayerNorm(x)))

    FFN sub-block  (ResidualBlock with "linear"):
      x = x + Dropout(Linear2(GELU(Linear1(LayerNorm(x)))))
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        # Attention sub-block
        self.ln_1  = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attn  = CausalSelfAttention(embed_dim, num_heads, dropout)

        # FFN sub-block
        self.ln_2    = nn.LayerNorm(embed_dim, eps=1e-5)
        self.mlp_fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.mlp_fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention residual (no post-add activation — "linear")
        x = x + self.attn(self.ln_1(x))

        # FFN residual (no post-add activation — "linear")
        h = self.mlp_fc2(F.gelu(self.mlp_fc1(self.ln_2(x))))
        x = x + self.ffn_drop(h)
        return x


class GPT2Small(nn.Module):
    """
    GPT-2 small for causal language modelling.
    Matches create_gpt2_small() in example_models.cpp:

      token_embed  : Embedding(50257, 768)
      pos_embed    : learned positional embedding (1, SEQ_LEN, 768)
      dropout
      12 x GPTBlock(768, 12, 3072, dropout=0.1, causal=True, activation=gelu)
      ln_f         : LayerNorm(768)
      head         : Linear(768, 50257, bias=True)

    Weight tying: token embedding and lm-head share the same weight matrix,
    following the standard GPT-2 practice.
    """
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 seq_len:    int = SEQ_LEN,
                 embed_dim:  int = EMBED_DIM,
                 num_heads:  int = NUM_HEADS,
                 num_layers: int = NUM_LAYERS,
                 ffn_dim:    int = FFN_DIM,
                 dropout:    float = DROPOUT):
        super().__init__()
        self.seq_len   = seq_len
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        # Learned positional embedding stored as a parameter (shape: 1, seq_len, embed_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.drop        = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim, eps=1e-5)
        self.head = nn.Linear(embed_dim, vocab_size, bias=True)

        # Weight tying
        self.head.weight = self.token_embed.weight

        # Parameter initialisation following GPT-2 paper
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.seq_len, f"Sequence length {T} exceeds max {self.seq_len}"

        tok = self.token_embed(x)                  # (B, T, embed_dim)
        pos = self.pos_embed[:, :T, :]             # (1, T, embed_dim)
        x   = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)                      # (B, T, vocab_size)
        return logits


# ======================== Training ========================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(">>> Running on device:", device)

    epochs      = int(os.getenv("EPOCHS",      "1"))
    batch_size  = int(os.getenv("BATCH_SIZE",  "8"))
    lr_initial  = float(os.getenv("LR_INITIAL", "3e-4"))
    grad_accum  = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
    data_path   = os.getenv("OPENWEBTEXT_PATH", "data/open-web-text/train.bin")
    max_steps   = int(os.getenv("MAX_STEPS", "-1"))   # -1 = no limit

    print(f">>> Data path      : {data_path}")
    print(f">>> Epochs         : {epochs}")
    print(f">>> Batch size     : {batch_size}")
    print(f">>> LR initial     : {lr_initial}")
    print(f">>> Grad accum     : {grad_accum}")
    print(f">>> Max steps      : {'unlimited' if max_steps < 0 else max_steps}")

    dataset = OpenWebTextDataset(data_path, seq_len=SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=2, pin_memory=True)

    print(f">>> Dataset tokens : {len(dataset.data):,}")
    print(f">>> Batches/epoch  : {len(loader)}")

    model = GPT2Small().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Parameters     : {total_params:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr_initial,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    # Cosine LR schedule with linear warm-up (standard for GPT-2 pretraining)
    warmup_steps = 2000
    total_steps  = len(loader) * epochs if max_steps < 0 else max_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        epoch_start = time.time()

        model.train()
        running_loss  = 0.0
        tokens_seen   = 0
        optimizer.zero_grad()

        last_log = time.time()

        for batch_idx, (inputs, targets) in enumerate(loader):
            if 0 <= max_steps <= global_step:
                break

            inputs  = inputs.to(device)   # (B, SEQ_LEN)
            targets = targets.to(device)  # (B, SEQ_LEN)

            logits = model(inputs)         # (B, T, vocab_size)
            B, T, V = logits.shape

            loss = criterion(logits.view(B * T, V), targets.view(B * T))
            loss = loss / grad_accum
            loss.backward()

            running_loss += loss.item() * grad_accum
            tokens_seen  += B * T

            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    avg_loss  = running_loss / (batch_idx + 1)
                    elapsed   = time.time() - last_log
                    tok_per_s = tokens_seen / elapsed if elapsed > 0 else 0.0
                    last_log  = time.time()
                    tokens_seen = 0
                    current_lr  = optimizer.param_groups[0]["lr"]
                    print(
                        f"[Step {global_step} | Batch {batch_idx+1}/{len(loader)}] "
                        f"Loss: {avg_loss:.4f} | "
                        f"Perplexity: {math.exp(min(avg_loss, 20)):.2f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Tok/s: {tok_per_s:,.0f}"
                    )

        epoch_time = time.time() - epoch_start
        avg_loss   = running_loss / max(1, len(loader))
        print(
            f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Perplexity: {math.exp(min(avg_loss, 20)):.2f}"
        )

        os.makedirs("model_snapshots", exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            "model_snapshots/gpt2_small.pth",
        )

    print("\n>>> GPT-2 small training completed.")


if __name__ == "__main__":
    main()
