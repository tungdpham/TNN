
#!/usr/bin/env python3
import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# =========================
# Utils
# =========================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

class CsvAppender:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.f = open(self.path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        if self.path.stat().st_size == 0:
            self.w.writeheader()
            self.f.flush()

    def write(self, row):
        self.w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass

class NetCounter:
    def __init__(self, ifname: Optional[str]):
        self.ifname = ifname
        self.available = False
        self.prev_tx = 0
        self.prev_rx = 0
        self.prev_t = time.time()
        if ifname:
            tx, rx = self._read_bytes()
            if tx is not None and rx is not None:
                self.available = True
                self.prev_tx = tx
                self.prev_rx = rx

    def _read_bytes(self) -> Tuple[Optional[int], Optional[int]]:
        try:
            base = Path("/sys/class/net") / self.ifname / "statistics"
            return int((base / "tx_bytes").read_text().strip()), int((base / "rx_bytes").read_text().strip())
        except Exception:
            return None, None

    def sample(self):
        t = time.time()
        if not self.available:
            return dict(ifname=self.ifname or "", dt_sec=0.0, tx_bytes_delta=0, rx_bytes_delta=0, tx_MBps=0.0, rx_MBps=0.0, counter_ok=0)
        tx, rx = self._read_bytes()
        if tx is None or rx is None:
            self.available = False
            return dict(ifname=self.ifname or "", dt_sec=0.0, tx_bytes_delta=0, rx_bytes_delta=0, tx_MBps=0.0, rx_MBps=0.0, counter_ok=0)
        dt = max(t - self.prev_t, 1e-9)
        dtx, drx = tx - self.prev_tx, rx - self.prev_rx
        self.prev_tx, self.prev_rx, self.prev_t = tx, rx, t
        return dict(
            ifname=self.ifname or "",
            dt_sec=dt,
            tx_bytes_delta=dtx,
            rx_bytes_delta=drx,
            tx_MBps=dtx / dt / (1024 * 1024),
            rx_MBps=drx / dt / (1024 * 1024),
            counter_ok=1,
        )

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

# =========================
# Datasets
# =========================

class CIFAR10Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.data, self.targets = [], []
        batch_files = [f"data_batch_{i}.bin" for i in range(1, 6)] if train else ["test_batch.bin"]
        for fname in batch_files:
            path = os.path.join(root, fname)
            with open(path, "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3073)
                self.data.append(arr[:, 1:].reshape(-1, 3, 32, 32))
                self.targets.append(arr[:, 0])
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].astype(np.float32) / 255.0)
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])

_CIFAR10_MEAN = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3, 1, 1)
_CIFAR10_STD = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3, 1, 1)
def _cifar10_transform(img): return (img - _CIFAR10_MEAN) / _CIFAR10_STD

class OpenWebTextBinDataset(Dataset):
    def __init__(self, path, seq_len, dtype="uint16"):
        self.path = Path(path)
        self.seq_len = seq_len
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        np_dtype = np.uint16 if dtype == "uint16" else np.int32
        self.tokens = np.fromfile(self.path, dtype=np_dtype)
        if len(self.tokens) < seq_len + 1:
            raise RuntimeError(f"Not enough tokens in {self.path}: total_tokens={len(self.tokens)}, seq_len={seq_len}")

        self.num_samples = (len(self.tokens) - 1) // seq_len
        if self.num_samples <= 0:
            raise RuntimeError(f"Built zero samples from {self.path}: total_tokens={len(self.tokens)}, seq_len={seq_len}")

        print(
            f"[OpenWebTextBinDataset] path={self.path} total_tokens={len(self.tokens)} "
            f"samples={self.num_samples} seq_len={seq_len}",
            flush=True,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.tokens[start:start + self.seq_len].astype(np.int64), dtype=torch.long)
        y = torch.tensor(self.tokens[start + 1:start + self.seq_len + 1].astype(np.int64), dtype=torch.long)
        return x, y

# =========================
# Models
# =========================

class _BasicResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)

class ResNet9Stage0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.res1 = _BasicResidualBlock(128)
        self.res2 = _BasicResidualBlock(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ResNet9Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.res3 = _BasicResidualBlock(256)
        self.res4 = _BasicResidualBlock(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(512)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.res5 = _BasicResidualBlock(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.maxpool2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.maxpool3(x)
        x = self.res5(x)
        x = self.avgpool(x)
        return self.fc(self.flatten(x))

_GPT2_VOCAB_SIZE = 50257
_GPT2_EMBED_DIM = 768
_GPT2_NUM_HEADS = 12
_GPT2_NUM_LAYERS = 12
_GPT2_FFN_DIM = _GPT2_EMBED_DIM * 4

class _CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = F.softmax(att.masked_fill(mask, float("-inf")), dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class _GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = _CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Stage0(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(_GPT2_VOCAB_SIZE, _GPT2_EMBED_DIM)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, _GPT2_EMBED_DIM))
        self.blocks = nn.Sequential(*[
            _GPTBlock(_GPT2_EMBED_DIM, _GPT2_NUM_HEADS, _GPT2_FFN_DIM)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        _, T = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :T, :]
        return self.blocks(h)

class GPT2Stage1(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.blocks = nn.Sequential(*[
            _GPTBlock(_GPT2_EMBED_DIM, _GPT2_NUM_HEADS, _GPT2_FFN_DIM)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(_GPT2_EMBED_DIM)
        self.head = nn.Linear(_GPT2_EMBED_DIM, _GPT2_VOCAB_SIZE, bias=False)

    def forward(self, h):
        return self.head(self.ln_f(self.blocks(h)))

# =========================
# Config
# =========================

def _make_model_configs(seq_len=512):
    cifar10_root = os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin")
    gpt_stage0_layers = int(os.getenv("GPT2_STAGE0_LAYERS", "6"))
    gpt_stage1_layers = _GPT2_NUM_LAYERS - gpt_stage0_layers
    return {
        "resnet9_cifar10": {
            "stage0_cls": ResNet9Stage0,
            "stage1_cls": ResNet9Stage1,
            "train_set": lambda: CIFAR10Bin(cifar10_root, train=True, transform=_cifar10_transform),
            "test_set": lambda: CIFAR10Bin(cifar10_root, train=False, transform=_cifar10_transform),
            "act_tail": (128, 16, 16),
            "batch_size": int(os.getenv("BATCH_SIZE", "128")),
            "epochs": int(os.getenv("EPOCHS", "10")),
            "lr": float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers": 2,
            "is_gpt2": False,
        },
        "gpt2": {
            "stage0_cls": lambda: GPT2Stage0(gpt_stage0_layers),
            "stage1_cls": lambda: GPT2Stage1(gpt_stage1_layers),
            "train_set": lambda: OpenWebTextBinDataset(
                path=os.getenv("OPENWEBTEXT_TRAIN_BIN", "data/open-web-text-1pct/train.bin"),
                seq_len=seq_len,
                dtype=os.getenv("OPENWEBTEXT_BIN_DTYPE", "uint16"),
            ),
            "test_set": lambda: OpenWebTextBinDataset(
                path=os.getenv("OPENWEBTEXT_VAL_BIN", "data/open-web-text-1pct/val.bin"),
                seq_len=seq_len,
                dtype=os.getenv("OPENWEBTEXT_BIN_DTYPE", "uint16"),
            ),
            "act_tail": (seq_len, _GPT2_EMBED_DIM),
            "batch_size": int(os.getenv("BATCH_SIZE", "8")),
            "epochs": int(os.getenv("EPOCHS", "1")),
            "lr": float(os.getenv("LR_INITIAL", "0.0001")),
            "num_workers": 0,
            "is_gpt2": True,
        },
    }

# =========================
# Protocol
# =========================

CMD_STOP = 0
CMD_TRAIN = 1
CMD_EVAL = 2

def send_header(device, cmd: int, sizes: List[int]):
    head = torch.tensor([cmd, len(sizes)], dtype=torch.int64, device=device)
    dist.send(head, dst=1)
    if len(sizes) > 0:
        dist.send(torch.tensor(sizes, dtype=torch.int64, device=device), dst=1)

def recv_header(device):
    head = torch.empty(2, dtype=torch.int64, device=device)
    dist.recv(head, src=0)
    cmd = int(head[0].item())
    nmb = int(head[1].item())
    sizes = []
    if nmb > 0:
        sz = torch.empty(nmb, dtype=torch.int64, device=device)
        dist.recv(sz, src=0)
        sizes = [int(x) for x in sz.tolist()]
    return cmd, sizes

def send_stats(device, loss_sum, correct, total):
    stats = torch.tensor([loss_sum, correct, total], dtype=torch.float64, device=device)
    dist.send(stats, dst=0)

def recv_stats(device):
    stats = torch.empty(3, dtype=torch.float64, device=device)
    dist.recv(stats, src=1)
    return float(stats[0].item()), float(stats[1].item()), float(stats[2].item())

# =========================
# Microbatch helpers
# =========================

def compute_micro_sizes(batch_size: int, micro_bs: int) -> List[int]:
    if micro_bs <= 0:
        raise ValueError("micro_bs must be > 0")
    out = []
    remain = batch_size
    while remain > 0:
        cur = min(remain, micro_bs)
        out.append(cur)
        remain -= cur
    return out

def split_by_sizes(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    chunks = []
    start = 0
    for s in sizes:
        chunks.append(x[start:start + s])
        start += s
    return chunks

# =========================
# 1F1B schedule (2 stages)
# =========================

def train_batch_rank0_1f1b(model, optimizer, x, y, device, cfg, micro_bs):
    sizes = compute_micro_sizes(x.size(0), micro_bs)
    x_mbs = split_by_sizes(x, sizes)
    y_mbs = split_by_sizes(y, sizes)

    optimizer.zero_grad(set_to_none=True)

    acts: List[Optional[torch.Tensor]] = [None] * len(sizes)

    send_header(device, CMD_TRAIN, sizes)

    # Warmup: forward first microbatch and send it.
    acts[0] = model(x_mbs[0])
    dist.send(acts[0].detach().contiguous(), dst=1)
    dist.send(y_mbs[0].contiguous(), dst=1)

    # Steady state: forward i, recv grad for i-1, send i, backward i-1
    for i in range(1, len(sizes)):
        acts[i] = model(x_mbs[i])

        g_prev = torch.empty((sizes[i - 1], *cfg["act_tail"]), dtype=torch.float32, device=device)
        dist.recv(g_prev, src=1)

        dist.send(acts[i].detach().contiguous(), dst=1)
        dist.send(y_mbs[i].contiguous(), dst=1)

        acts[i - 1].backward(g_prev)
        acts[i - 1] = None

    # Cooldown: recv grad for final microbatch and backward it.
    g_last = torch.empty((sizes[-1], *cfg["act_tail"]), dtype=torch.float32, device=device)
    dist.recv(g_last, src=1)
    acts[-1].backward(g_last)
    acts[-1] = None

    optimizer.step()
    return recv_stats(device)

def eval_batch_rank0(model, x, y, device, micro_bs):
    sizes = compute_micro_sizes(x.size(0), micro_bs)
    x_mbs = split_by_sizes(x, sizes)
    y_mbs = split_by_sizes(y, sizes)
    send_header(device, CMD_EVAL, sizes)
    for x_mb, y_mb in zip(x_mbs, y_mbs):
        a = model(x_mb)
        dist.send(a.contiguous(), dst=1)
        dist.send(y_mb.contiguous(), dst=1)
    return recv_stats(device)

def train_batch_rank1_1f1b(model, optimizer, device, cfg, sizes):
    is_gpt2 = cfg.get("is_gpt2", False)
    optimizer.zero_grad(set_to_none=True)

    batch_items = 0
    for s in sizes:
        batch_items += s * (cfg["act_tail"][0] - 1) if is_gpt2 else s

    stats_loss_sum = 0.0
    stats_correct = 0.0
    stats_total = 0.0

    for mbsz in sizes:
        act = torch.empty((mbsz, *cfg["act_tail"]), dtype=torch.float32, device=device)
        if is_gpt2:
            y = torch.empty((mbsz, cfg["act_tail"][0]), dtype=torch.int64, device=device)
        else:
            y = torch.empty((mbsz,), dtype=torch.int64, device=device)

        dist.recv(act, src=0)
        dist.recv(y, src=0)

        act.requires_grad_(True)
        logits = model(act)

        if is_gpt2:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss_sum = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            correct = (shift_logits.argmax(dim=-1) == shift_labels).sum().item()
            total = shift_labels.numel()
        else:
            loss_sum = F.cross_entropy(logits, y, reduction="sum")
            correct = (logits.argmax(dim=1) == y).sum().item()
            total = y.numel()

        loss = loss_sum / max(batch_items, 1)
        loss.backward()

        dist.send(act.grad.contiguous(), dst=0)

        stats_loss_sum += float(loss_sum.item())
        stats_correct += float(correct)
        stats_total += float(total)

    optimizer.step()
    send_stats(device, stats_loss_sum, stats_correct, stats_total)

def eval_batch_rank1(model, device, cfg, sizes):
    is_gpt2 = cfg.get("is_gpt2", False)
    stats_loss_sum = 0.0
    stats_correct = 0.0
    stats_total = 0.0

    with torch.no_grad():
        for mbsz in sizes:
            act = torch.empty((mbsz, *cfg["act_tail"]), dtype=torch.float32, device=device)
            if is_gpt2:
                y = torch.empty((mbsz, cfg["act_tail"][0]), dtype=torch.int64, device=device)
            else:
                y = torch.empty((mbsz,), dtype=torch.int64, device=device)

            dist.recv(act, src=0)
            dist.recv(y, src=0)
            logits = model(act)

            if is_gpt2:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = y[:, 1:].contiguous()
                loss_sum = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                ).item()
                correct = (shift_logits.argmax(dim=-1) == shift_labels).sum().item()
                total = shift_labels.numel()
            else:
                loss_sum = F.cross_entropy(logits, y, reduction="sum").item()
                correct = (logits.argmax(dim=1) == y).sum().item()
                total = y.numel()

            stats_loss_sum += float(loss_sum)
            stats_correct += float(correct)
            stats_total += float(total)

    send_stats(device, stats_loss_sum, stats_correct, stats_total)

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet9_cifar10", choices=["gpt2", "resnet9_cifar10"])
    parser.add_argument("--micro-bs", type=int, default=int(os.getenv("MICRO_BS", "64")))
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--print-freq", type=int, default=int(os.getenv("PRINT_FREQ", "10")))
    parser.add_argument("--log-dir", type=str, default=os.getenv("LOG_DIR", "logs"))
    args = parser.parse_args()

    max_steps = int(os.getenv("MAX_STEPS", "-1"))
    skip_eval = os.getenv("SKIP_EVAL", "0") == "1"

    backend = os.getenv("DIST_BACKEND", "nccl")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2:
        raise RuntimeError(f"world_size must be 2, got {world_size}")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    cfg = _make_model_configs(args.seq_len)[args.model]
    model = (cfg["stage0_cls"]() if rank == 0 else cfg["stage1_cls"]()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-3, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    log_dir = ensure_dir(args.log_dir)
    nic = os.getenv("NET_LOG_IFNAME") or os.getenv("NCCL_SOCKET_IFNAME") or os.getenv("GLOO_SOCKET_IFNAME")
    net_counter = NetCounter(nic)
    net_csv = CsvAppender(log_dir / f"{args.model}_rank{rank}_net.csv", ["timestamp", "rank", "phase", "epoch", "step", "batch_size", "ifname", "dt_sec", "tx_bytes_delta", "rx_bytes_delta", "tx_MBps", "rx_MBps", "counter_ok"])
    metrics_csv = None
    if rank == 0:
        metrics_csv = CsvAppender(log_dir / f"{args.model}_rank0_metrics.csv", ["timestamp", "phase", "epoch", "step", "batch_size", "micro_bs", "lr", "loss_sum", "samples", "avg_loss", "correct", "accuracy_percent", "epoch_time_sec"])

    print(
        f"[Init] rank={rank} model={args.model} params={count_params(model):,} micro_bs={args.micro_bs}",
        flush=True,
    )

    try:
        if rank == 0:
            train_loader = DataLoader(cfg["train_set"](), batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], drop_last=False, pin_memory=True)
            test_loader = DataLoader(cfg["test_set"](), batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], drop_last=False, pin_memory=True)

            for epoch in range(cfg["epochs"]):
                model.train()
                train_loss_sum = 0.0
                train_correct = 0
                train_total = 0
                epoch_start = time.time()

                for step, (x, y) in enumerate(train_loader):
                    if max_steps > 0 and step >= max_steps:
                        break

                    t0 = time.time()
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    loss_sum_b, correct_b, total_b = train_batch_rank0_1f1b(
                        model=model,
                        optimizer=optimizer,
                        x=x,
                        y=y,
                        device=device,
                        cfg=cfg,
                        micro_bs=args.micro_bs,
                    )
                    train_loss_sum += loss_sum_b
                    train_correct += int(correct_b)
                    train_total += int(total_b)

                    avg_loss = train_loss_sum / max(train_total, 1)
                    acc = 100.0 * train_correct / max(train_total, 1)
                    net = net_counter.sample()

                    net_csv.write(dict(timestamp=t0, rank=rank, phase="train_batch", epoch=epoch + 1, step=step, batch_size=int(x.size(0)), **net))
                    metrics_csv.write(dict(timestamp=t0, phase="train_batch", epoch=epoch + 1, step=step, batch_size=int(x.size(0)), micro_bs=args.micro_bs, lr=float(optimizer.param_groups[0]["lr"]), loss_sum=float(train_loss_sum), samples=int(train_total), avg_loss=float(avg_loss), correct=int(train_correct), accuracy_percent=float(acc), epoch_time_sec=float(time.time() - epoch_start)))

                    if step % args.print_freq == 0:
                        elapsed = time.time() - epoch_start
                        sec_per_step = elapsed / max(step + 1, 1)
                        eta_sec = (len(train_loader) - (step + 1)) * sec_per_step
                        if cfg.get("is_gpt2", False):
                            print(
                                f"[Rank 0] Epoch {epoch+1}/{cfg['epochs']} Step {step}/{len(train_loader)} "
                                f"| train_loss={avg_loss:.4f} | micro_bs={args.micro_bs} "
                                f"| net_tx={net['tx_MBps']:.4f}MB/s net_rx={net['rx_MBps']:.4f}MB/s "
                                f"| sec/step={sec_per_step:.3f} ETA={eta_sec/3600:.2f}h",
                                flush=True,
                            )
                        else:
                            print(
                                f"[Rank 0] Epoch {epoch+1}/{cfg['epochs']} Step {step}/{len(train_loader)} "
                                f"| train_loss={avg_loss:.4f} | train_acc={acc:.2f}% | micro_bs={args.micro_bs} "
                                f"| net_tx={net['tx_MBps']:.4f}MB/s net_rx={net['rx_MBps']:.4f}MB/s "
                                f"| sec/step={sec_per_step:.3f} ETA={eta_sec/3600:.2f}h",
                                flush=True,
                            )

                scheduler.step()
                epoch_time = time.time() - epoch_start

                if skip_eval:
                    print(f"[Rank 0] Epoch {epoch+1}/{cfg['epochs']} DONE | time={epoch_time:.2f}s | eval=skipped", flush=True)
                    send_header(device, CMD_STOP, [])
                    print("[Rank 0] Training complete.", flush=True)
                    return

                model.eval()
                test_loss_sum = 0.0
                test_correct = 0
                test_total = 0
                with torch.no_grad():
                    for step, (x, y) in enumerate(test_loader):
                        if max_steps > 0 and step >= max_steps:
                            break
                        t0 = time.time()
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        loss_sum_b, correct_b, total_b = eval_batch_rank0(model, x, y, device, args.micro_bs)
                        test_loss_sum += loss_sum_b
                        test_correct += int(correct_b)
                        test_total += int(total_b)
                        net_csv.write(dict(timestamp=t0, rank=rank, phase="test_batch", epoch=epoch + 1, step=step, batch_size=int(x.size(0)), **net_counter.sample()))

                train_loss = train_loss_sum / max(train_total, 1)
                train_acc = 100.0 * train_correct / max(train_total, 1)
                test_loss = test_loss_sum / max(test_total, 1)
                test_acc = 100.0 * test_correct / max(test_total, 1)
                metrics_csv.write(dict(timestamp=time.time(), phase="epoch_summary", epoch=epoch + 1, step=-1, batch_size=cfg["batch_size"], micro_bs=args.micro_bs, lr=float(optimizer.param_groups[0]["lr"]), loss_sum=float(test_loss_sum), samples=int(test_total), avg_loss=float(test_loss), correct=int(test_correct), accuracy_percent=float(test_acc), epoch_time_sec=float(epoch_time)))

                if cfg.get("is_gpt2", False):
                    print(
                        f"[Rank 0] Epoch {epoch+1}/{cfg['epochs']} DONE | time={epoch_time:.2f}s "
                        f"| train_loss={train_loss:.4f} | test_loss={test_loss:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[Rank 0] Epoch {epoch+1}/{cfg['epochs']} DONE | time={epoch_time:.2f}s "
                        f"| train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                        f"| test_loss={test_loss:.4f} test_acc={test_acc:.2f}%",
                        flush=True,
                    )

            send_header(device, CMD_STOP, [])
            print("[Rank 0] Training complete.", flush=True)

        else:
            while True:
                cmd, sizes = recv_header(device)
                if cmd == CMD_STOP:
                    print("[Rank 1] Received STOP.", flush=True)
                    break
                if cmd == CMD_TRAIN:
                    train_batch_rank1_1f1b(model, optimizer, device, cfg, sizes)
                elif cmd == CMD_EVAL:
                    eval_batch_rank1(model, device, cfg, sizes)
                else:
                    raise RuntimeError(f"Unknown cmd={cmd}")

    finally:
        net_csv.close()
        if metrics_csv is not None:
            metrics_csv.close()
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
