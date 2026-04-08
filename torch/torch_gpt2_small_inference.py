import os
import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


# ======================== GPT-2 small hyper-parameters ========================
SEQ_LEN    = 1024
VOCAB_SIZE = 50257
EMBED_DIM  = 768
NUM_HEADS  = 12
NUM_LAYERS = 12
FFN_DIM    = EMBED_DIM * 4
DROPOUT    = 0.1


# ======================== Model ========================

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.embed_dim, dim=2)

        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        att   = (q @ k.transpose(-2, -1)) * scale
        mask  = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att   = att.masked_fill(mask, float("-inf"))
        att   = F.softmax(att, dim=-1)
        att   = self.attn_drop(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        return out


class GPTBlock(nn.Module):
    """
    GPT-2 transformer block (matches gpt_block() in layer_builder.hpp):
      x = x + Dropout(Attention(LayerNorm(x)))          # "linear" residual
      x = x + Dropout(FC2(GELU(FC1(LayerNorm(x)))))     # "linear" residual
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln_1     = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attn     = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2     = nn.LayerNorm(embed_dim, eps=1e-5)
        self.mlp_fc1  = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.mlp_fc2  = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn_drop(self.mlp_fc2(F.gelu(self.mlp_fc1(self.ln_2(x)))))
        return x


class GPT2Small(nn.Module):
    """
    GPT-2 small — matches create_gpt2_small() in example_models.cpp.
    Weight tying between token_embed and head.
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
        self.seq_len = seq_len

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.drop        = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim, eps=1e-5)
        self.head = nn.Linear(embed_dim, vocab_size, bias=True)
        self.head.weight = self.token_embed.weight  # weight tying

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        tok    = self.token_embed(x)
        pos    = self.pos_embed[:, :T, :]
        x      = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)   # (B, T, vocab_size)


# ======================== Helpers ========================

def top_k_sampling(logits: torch.Tensor, top_k: int = 50,
                   temperature: float = 1.0) -> int:
    """Sample next token from logits using top-k sampling."""
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[-1]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def greedy_decode(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=-1).item())


def perplexity_on_file(model: GPT2Small, path: str, device: torch.device,
                       seq_len: int = SEQ_LEN, batch_size: int = 8) -> float:
    """Compute perplexity over a tokenised binary file."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    tokens = torch.from_numpy(data.astype(np.int64))
    n = (len(tokens) - 1) // seq_len

    total_loss = 0.0
    total_toks = 0
    criterion  = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        for start in range(0, n * seq_len, batch_size * seq_len):
            batch_inputs, batch_targets = [], []
            for b in range(batch_size):
                s = start + b * seq_len
                if s + seq_len + 1 > len(tokens):
                    break
                batch_inputs.append(tokens[s : s + seq_len])
                batch_targets.append(tokens[s + 1 : s + seq_len + 1])
            if not batch_inputs:
                break
            x = torch.stack(batch_inputs).to(device)
            y = torch.stack(batch_targets).to(device)
            logits = model(x)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))
            total_loss += loss.item() * B * T
            total_toks += B * T

    return math.exp(total_loss / max(1, total_toks))


# ======================== Main ========================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(">>> Running GPT-2 small inference on device:", device)

    model_path  = os.getenv("MODEL_PATH",        "model_snapshots/gpt2_small.pth")
    data_path   = os.getenv("OPENWEBTEXT_PATH",  "data/open-web-text/train.bin")
    prompt_text = os.getenv("PROMPT",            "The quick brown fox")
    max_new     = int(os.getenv("MAX_NEW_TOKENS", "100"))
    top_k       = int(os.getenv("TOP_K",          "50"))
    temperature = float(os.getenv("TEMPERATURE",  "1.0"))
    eval_ppl    = os.getenv("EVAL_PPL", "0") == "1"

    print(f">>> Model path  : {model_path}")
    print(f">>> Prompt      : {prompt_text!r}")
    print(f">>> Max new tok : {max_new}")
    print(f">>> Top-k       : {top_k}")
    print(f">>> Temperature : {temperature}")

    model = GPT2Small().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Parameters  : {total_params:,}")

    if os.path.isfile(model_path):
        print(f">>> Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        print(">>> Weights loaded successfully")
        if "epoch" in checkpoint:
            print(f"    (saved at epoch {checkpoint['epoch']}, "
                  f"step {checkpoint.get('global_step', '?')}, "
                  f"loss {checkpoint.get('loss', '?'):.4f})")
    else:
        print(f">>> Warning: model file not found at {model_path}")
        print(">>> Running with randomly initialised weights")

    # ---- Optional perplexity evaluation ----
    if eval_ppl and os.path.isfile(data_path):
        print(f"\n>>> Computing perplexity on: {data_path}")
        ppl_start = time.time()
        ppl = perplexity_on_file(model, data_path, device)
        print(f">>> Perplexity: {ppl:.2f}  ({time.time()-ppl_start:.2f}s)")

    # ---- Text generation ----
    if not _TIKTOKEN_AVAILABLE:
        print("\n>>> tiktoken not installed — skipping text generation.")
        print("    Install with: pip install tiktoken")
        return

    enc = tiktoken.get_encoding("gpt2")

    prompt_ids = enc.encode_ordinary(prompt_text)
    if len(prompt_ids) == 0:
        prompt_ids = [enc.eot_token]

    print(f"\n>>> [PROMPT]: {prompt_text}")
    print(">>> [GENERATED]: ", end="", flush=True)

    model.eval()
    current_ids = list(prompt_ids)

    gen_start = time.time()

    with torch.no_grad():
        for _ in range(max_new):
            context = current_ids[-SEQ_LEN:]                        # truncate to seq_len
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)                                        # (1, T, vocab_size)
            next_logits = logits[0, -1, :]                          # last position

            if top_k > 1:
                next_token = top_k_sampling(next_logits, top_k=top_k, temperature=temperature)
            else:
                next_token = greedy_decode(next_logits)

            current_ids.append(next_token)
            print(enc.decode([next_token]), end="", flush=True)

            if next_token == enc.eot_token:
                break

    gen_time = time.time() - gen_start
    tokens_generated = len(current_ids) - len(prompt_ids)
    print(f"\n\n>>> Generated {tokens_generated} tokens in {gen_time:.2f}s "
          f"({tokens_generated / gen_time:.1f} tok/s)")


if __name__ == "__main__":
    main()
