#!/usr/bin/env python3
"""
Unified single-GPU trainer for tunx PyTorch reference models.

Usage:
    python torch_unified_trainer.py --model resnet9_cifar10
    python torch_unified_trainer.py --model wrn16_8_cifar100 --epochs 50
    python torch_unified_trainer.py --model resnet50_tiny_imagenet --batch-size 128
    python torch_unified_trainer.py --model resnet50_imagenet100 --batch-size 64
    python torch_unified_trainer.py --model gpt2_small --batch-size 8

Environment variables can override defaults (see .env or shell):
    EPOCHS, BATCH_SIZE, LR_INITIAL, CIFAR10_BIN_ROOT, CIFAR100_BIN_ROOT,
    TINY_IMAGENET_ROOT, IMAGENET100_ROOT, OPENWEBTEXT_PATH, etc.
"""
import argparse
import csv
import datetime
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from dotenv import load_dotenv

load_dotenv()
torch.set_float32_matmul_precision('high')

# ======================== Datasets ========================

class CIFAR10Bin(Dataset):
    """CIFAR-10 binary format loader."""
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []

        if train:
            batch_files = [f"data_batch_{i}.bin" for i in range(1, 6)]
        else:
            batch_files = ["test_batch.bin"]

        for fname in batch_files:
            path = os.path.join(root, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8)
                arr = arr.reshape(-1, 3073)
                labels = arr[:, 0]
                images = arr[:, 1:].reshape(-1, 3, 32, 32)
                self.data.append(images)
                self.targets.append(labels)

        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        label = int(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR100Bin(Dataset):
    """CIFAR-100 binary format loader."""
    def __init__(self, root, train=True, transform=None):
        self.transform = transform

        fname = "train.bin" if train else "test.bin"
        path = os.path.join(root, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            arr = arr.reshape(-1, 3074)  # coarse(1) + fine(1) + pixels(3072)

        # fine labels are at index 1
        self.targets = arr[:, 1].astype(np.int64)
        self.data = arr[:, 2:].reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        label = int(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


class TinyImageNetDataset(Dataset):
    """Tiny ImageNet dataset loader."""
    def __init__(self, root: str, train: bool = True, transform=None):
        self.transform = transform
        root = Path(root)

        self.samples = []
        self.class_to_idx = {}

        # Build class -> idx mapping from train directory (canonical ordering)
        train_dir = root / "train"
        classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        if train:
            for cls in classes:
                cls_dir = train_dir / cls / "images"
                if not cls_dir.exists():
                    continue
                for img_path in cls_dir.glob("*.JPEG"):
                    self.samples.append((str(img_path), self.class_to_idx[cls]))
        else:
            # Validation set
            val_dir = root / "val"
            annotations_file = val_dir / "val_annotations.txt"
            with open(annotations_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img_name = parts[0]
                    cls = parts[1]
                    img_path = val_dir / "images" / img_name
                    if cls in self.class_to_idx and img_path.exists():
                        self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class ImageNet100Dataset(Dataset):
    """ImageNet-100 dataset loader."""
    def __init__(self, root: str, train: bool = True, transform=None):
        self.transform = transform
        root = Path(root)

        self.samples = []
        self.class_to_idx = {}

        # Build class -> idx mapping from all train directories
        all_classes = set()
        for train_subdir in ["train.X1", "train.X2", "train.X3", "train.X4"]:
            d = root / train_subdir
            if d.exists():
                all_classes.update(x.name for x in d.iterdir() if x.is_dir())
        
        classes = sorted(all_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        if train:
            for train_subdir in ["train.X1", "train.X2", "train.X3", "train.X4"]:
                data_dir = root / train_subdir
                if not data_dir.exists():
                    continue
                for cls in classes:
                    img_dir = data_dir / cls
                    if not img_dir.exists():
                        continue
                    for ext in ["*.JPEG", "*.jpg", "*.png"]:
                        for p in img_dir.glob(ext):
                            self.samples.append((str(p), self.class_to_idx[cls]))
        else:
            data_dir = root / "val.X"
            for cls in classes:
                img_dir = data_dir / cls
                if not img_dir.exists():
                    continue
                for ext in ["*.JPEG", "*.jpg", "*.png"]:
                    for p in img_dir.glob(ext):
                        self.samples.append((str(p), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class OpenWebTextDataset(Dataset):
    """OpenWebText tokenized dataset for GPT-2."""
    def __init__(self, path: str, seq_len: int = 1024):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        self.seq_len = seq_len
        # Memory-mapped file to avoid loading entire dataset into RAM
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        chunk = torch.from_numpy(chunk)
        x = chunk[:-1]  # [SEQ_LEN]
        y = chunk[1:]   # [SEQ_LEN]
        return x, y


# ======================== Models ========================

# --- ResNet-9 for CIFAR-10 ---

class BasicResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class ResNet9CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.res1 = BasicResidualBlock(128)
        self.res2 = BasicResidualBlock(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.res3 = BasicResidualBlock(256)
        self.res4 = BasicResidualBlock(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.res5 = BasicResidualBlock(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.maxpool2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.maxpool3(x)
        x = self.res5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# --- Wide ResNet-16-8 for CIFAR-100 ---

class WideResidualBlock(nn.Module):
    """Pre-activation wide residual block."""
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=True)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1,
                                      stride=stride, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        return out + sc


class WRN16_8CIFAR100(nn.Module):
    """WRN-16-8 for CIFAR-100."""
    def __init__(self, num_classes: int = 100):
        super().__init__()
        width_factor = 8
        dropout_rate = 0.3
        c1 = 16 * width_factor   # 128
        c2 = 32 * width_factor   # 256
        c3 = 64 * width_factor   # 512

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=True)

        # Group 1: 16 -> 128, stride 1
        self.group1_block1 = WideResidualBlock(16, c1, stride=1, dropout_rate=dropout_rate)
        self.group1_block2 = WideResidualBlock(c1, c1, stride=1, dropout_rate=dropout_rate)

        # Group 2: 128 -> 256, stride 2 (32x32 -> 16x16)
        self.group2_block1 = WideResidualBlock(c1, c2, stride=2, dropout_rate=dropout_rate)
        self.group2_block2 = WideResidualBlock(c2, c2, stride=1, dropout_rate=dropout_rate)

        # Group 3: 256 -> 512, stride 2 (16x16 -> 8x8)
        self.group3_block1 = WideResidualBlock(c2, c3, stride=2, dropout_rate=dropout_rate)
        self.group3_block2 = WideResidualBlock(c3, c3, stride=1, dropout_rate=dropout_rate)

        # Final BN+ReLU before pooling
        self.bn_final = nn.BatchNorm2d(c3, eps=1e-5, momentum=0.1)

        # Global average pool: 8x8 -> 1x1
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c3, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.group1_block1(x)
        x = self.group1_block2(x)
        x = self.group2_block1(x)
        x = self.group2_block2(x)
        x = self.group3_block1(x)
        x = self.group3_block2(x)
        x = F.relu(self.bn_final(x), inplace=True)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# --- Bottleneck block for ResNet-50 ---

class BottleneckBlock(nn.Module):
    """Bottleneck residual block for ResNet-50."""
    def __init__(self, in_channels: int, mid_channels: int,
                 out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        out = out + sc
        out = F.relu(out, inplace=True)
        return out


# --- ResNet-50 for Tiny ImageNet ---

class ResNet50TinyImageNet(nn.Module):
    """ResNet-50 for Tiny ImageNet (64x64 inputs, 200 classes)."""
    def __init__(self, num_classes: int = 200):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BottleneckBlock(64, 64, 256, stride=1),
            BottleneckBlock(256, 64, 256, stride=1),
            BottleneckBlock(256, 64, 256, stride=1),
        )

        self.layer2 = nn.Sequential(
            BottleneckBlock(256, 128, 512, stride=2),
            BottleneckBlock(512, 128, 512, stride=1),
            BottleneckBlock(512, 128, 512, stride=1),
            BottleneckBlock(512, 128, 512, stride=1),
        )

        self.layer3 = nn.Sequential(
            BottleneckBlock(512, 256, 1024, stride=2),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
        )

        self.layer4 = nn.Sequential(
            BottleneckBlock(1024, 512, 2048, stride=2),
            BottleneckBlock(2048, 512, 2048, stride=1),
            BottleneckBlock(2048, 512, 2048, stride=1),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# --- ResNet-50 for ImageNet-100 ---

class ResNet50ImageNet100(nn.Module):
    """ResNet-50 for ImageNet-100 (224x224 inputs, 100 classes)."""
    def __init__(self, num_classes: int = 100):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BottleneckBlock(64, 64, 256, stride=1),
            BottleneckBlock(256, 64, 256, stride=1),
            BottleneckBlock(256, 64, 256, stride=1),
        )

        self.layer2 = nn.Sequential(
            BottleneckBlock(256, 128, 512, stride=2),
            BottleneckBlock(512, 128, 512, stride=1),
            BottleneckBlock(512, 128, 512, stride=1),
            BottleneckBlock(512, 128, 512, stride=1),
        )

        self.layer3 = nn.Sequential(
            BottleneckBlock(512, 256, 1024, stride=2),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
        )

        self.layer4 = nn.Sequential(
            BottleneckBlock(1024, 512, 2048, stride=2),
            BottleneckBlock(2048, 512, 2048, stride=1),
            BottleneckBlock(2048, 512, 2048, stride=1),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# --- GPT-2 small ---

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.embed_dim, dim=2)

        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
        out = self.attn_drop(out)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        return out


class GPTBlock(nn.Module):
    """One GPT-2 transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.mlp_fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.mlp_fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        h = self.mlp_fc2(F.gelu(self.mlp_fc1(self.ln_2(x))))
        x = x + self.ffn_drop(h)
        return x


class GPT2Small(nn.Module):
    """GPT-2 small for causal language modelling."""
    def __init__(self,
                 vocab_size: int = 50257,
                 seq_len: int = 1024,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 ffn_dim: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim, eps=1e-5)
        self.head = nn.Linear(embed_dim, vocab_size, bias=True)

        # Weight tying
        self.head.weight = self.token_embed.weight

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

        tok = self.token_embed(x)
        pos = self.pos_embed[:, :T, :]
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# ======================== Model Registry ========================

# Normalization constants
CIFAR10_MEAN = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3, 1, 1)

CIFAR100_MEAN = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(3, 1, 1)
CIFAR100_STD = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(3, 1, 1)

TINY_MEAN = [0.4802, 0.4481, 0.3975]
TINY_STD = [0.2770, 0.2691, 0.2821]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Returns configuration dictionary for the specified model.
    
    Each config contains:
        - model_cls: Model class constructor
        - train_dataset: Training dataset factory
        - test_dataset: Test/validation dataset factory
        - criterion: Loss function
        - epochs: Number of training epochs
        - batch_size: Batch size
        - lr: Learning rate
        - num_workers: DataLoader workers
        - optimizer_type: 'adam' or 'adamw'
        - scheduler_type: 'step' or 'cosine'
        - is_language_model: bool flag for GPT-2 models
    """
    
    configs = {
        "resnet9_cifar10": {
            "model_cls": lambda: ResNet9CIFAR10(num_classes=10),
            "train_dataset": lambda: CIFAR10Bin(
                root=os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin"),
                train=True,
                transform=lambda img: (img - CIFAR10_MEAN) / CIFAR10_STD
            ),
            "test_dataset": lambda: CIFAR10Bin(
                root=os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin"),
                train=False,
                transform=lambda img: (img - CIFAR10_MEAN) / CIFAR10_STD
            ),
            "criterion": nn.CrossEntropyLoss(),
            "epochs": int(os.getenv("EPOCHS", "10")),
            "batch_size": int(os.getenv("BATCH_SIZE", "256")),
            "lr": float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers": 2,
            "optimizer_type": "adam",
            "scheduler_type": "step",
            "is_language_model": False,
        },
        
        "wrn16_8_cifar100": {
            "model_cls": lambda: WRN16_8CIFAR100(num_classes=100),
            "train_dataset": lambda: CIFAR100Bin(
                root=os.getenv("CIFAR100_BIN_ROOT", "data/cifar-100-binary"),
                train=True,
                transform=lambda img: (img - CIFAR100_MEAN) / CIFAR100_STD
            ),
            "test_dataset": lambda: CIFAR100Bin(
                root=os.getenv("CIFAR100_BIN_ROOT", "data/cifar-100-binary"),
                train=False,
                transform=lambda img: (img - CIFAR100_MEAN) / CIFAR100_STD
            ),
            "criterion": nn.CrossEntropyLoss(),
            "epochs": int(os.getenv("EPOCHS", "50")),
            "batch_size": int(os.getenv("BATCH_SIZE", "128")),
            "lr": float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers": 2,
            "optimizer_type": "adam",
            "scheduler_type": "step",
            "is_language_model": False,
        },
        
        "resnet50_tiny_imagenet": {
            "model_cls": lambda: ResNet50TinyImageNet(num_classes=200),
            "train_dataset": lambda: TinyImageNetDataset(
                root=os.getenv("TINY_IMAGENET_ROOT", "data/tiny-imagenet-200"),
                train=True,
                transform=T.Compose([
                    T.ToTensor(),
                    T.Normalize(TINY_MEAN, TINY_STD),
                ])
            ),
            "test_dataset": lambda: TinyImageNetDataset(
                root=os.getenv("TINY_IMAGENET_ROOT", "data/tiny-imagenet-200"),
                train=False,
                transform=T.Compose([
                    T.ToTensor(),
                    T.Normalize(TINY_MEAN, TINY_STD),
                ])
            ),
            "criterion": nn.CrossEntropyLoss(),
            "epochs": int(os.getenv("EPOCHS", "90")),
            "batch_size": int(os.getenv("BATCH_SIZE", "128")),
            "lr": float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers": 4,
            "optimizer_type": "adam",
            "scheduler_type": "step",
            "is_language_model": False,
        },
        
        "resnet50_imagenet100": {
            "model_cls": lambda: ResNet50ImageNet100(num_classes=100),
            "train_dataset": lambda: ImageNet100Dataset(
                root=os.getenv("IMAGENET100_ROOT", "data/imagenet-100"),
                train=True,
                transform=T.Compose([
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
            ),
            "test_dataset": lambda: ImageNet100Dataset(
                root=os.getenv("IMAGENET100_ROOT", "data/imagenet-100"),
                train=False,
                transform=T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
            ),
            "criterion": nn.CrossEntropyLoss(),
            "epochs": int(os.getenv("EPOCHS", "90")),
            "batch_size": int(os.getenv("BATCH_SIZE", "64")),
            "lr": float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers": 4,
            "optimizer_type": "adam",
            "scheduler_type": "step",
            "is_language_model": False,
        },
        
        "gpt2_small": {
            "model_cls": lambda: GPT2Small(
                vocab_size=50257,
                seq_len=1024,
                embed_dim=768,
                num_heads=12,
                num_layers=12,
                ffn_dim=3072,
                dropout=0.1
            ),
            "train_dataset": lambda: OpenWebTextDataset(
                path=os.getenv("OPENWEBTEXT_PATH", "data/open-web-text/train.bin"),
                seq_len=1024
            ),
            "test_dataset": lambda: OpenWebTextDataset(
                path=os.getenv("OPENWEBTEXT_VAL_PATH", "data/open-web-text/val.bin"),
                seq_len=1024
            ),
            "criterion": nn.CrossEntropyLoss(),
            "epochs": int(os.getenv("EPOCHS", "1")),
            "batch_size": int(os.getenv("BATCH_SIZE", "8")),
            "lr": float(os.getenv("LR_INITIAL", "3e-4")),
            "num_workers": 2,
            "optimizer_type": "adamw",
            "scheduler_type": "cosine",
            "is_language_model": True,
        },
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(configs.keys())}")
    
    return configs[model_name]


# ======================== Training ========================

def train_epoch(model, train_loader, criterion, optimizer, device, cfg, epoch, batch_writer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    is_lm = cfg["is_language_model"]
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        step_start = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if is_lm:
            # GPT-2: outputs (B, T, vocab_size), targets (B, T)
            B, T, V = outputs.shape
            loss = criterion(outputs.view(B * T, V), targets.view(B * T))
            # For accuracy, use the predictions
            _, predicted = outputs.view(B * T, V).max(1)
            correct = predicted.eq(targets.view(B * T)).sum().item()
            total = B * T
        else:
            # Classification: outputs (B, num_classes), targets (B,)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            total = targets.size(0)
        
        loss.backward()
        optimizer.step()
        
        step_ms = int((time.time() - step_start) * 1000)
        
        running_loss += loss.item() * total
        running_correct += correct
        running_total += total
        
        batch_loss = loss.item()
        batch_acc = 100.0 * correct / total
        batch_writer.writerow([
            epoch, batch_idx + 1, f"{batch_loss:.6f}",
            f"{batch_acc:.4f}", step_ms
        ])
        
        if (batch_idx + 1) % 100 == 0:
            print(
                f"[Train Batch {batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}% | "
                f"Step time: {step_ms}ms"
            )
    
    train_loss = running_loss / running_total
    train_acc = 100.0 * running_correct / running_total
    return train_loss, train_acc


def validate(model, test_loader, criterion, device, cfg, epoch, val_writer):
    """Validate the model."""
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    
    is_lm = cfg["is_language_model"]
    
    with torch.no_grad():
        for val_step, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if is_lm:
                B, T, V = outputs.shape
                loss = criterion(outputs.view(B * T, V), targets.view(B * T))
                _, predicted = outputs.view(B * T, V).max(1)
                correct = predicted.eq(targets.view(B * T)).sum().item()
                total = B * T
            else:
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                total = targets.size(0)
            
            val_loss_sum += loss.item() * total
            val_correct += correct
            val_total += total
            
            step_loss = loss.item()
            step_acc = 100.0 * correct / total
            val_writer.writerow([epoch, val_step + 1, f"{step_loss:.6f}", f"{step_acc:.4f}"])
    
    val_loss = val_loss_sum / val_total
    val_acc = 100.0 * val_correct / val_total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Unified tunx PyTorch Trainer")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet9_cifar10", "wrn16_8_cifar100",
                                 "resnet50_tiny_imagenet", "resnet50_imagenet100",
                                 "gpt2_small"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides config default)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides config default)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config default)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda:0', 'cpu')")
    args = parser.parse_args()
    
    # Get model configuration
    cfg = get_model_config(args.model)
    
    # Override config with command-line arguments
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f">>> Running on device: {device}")
    print(f">>> Model: {args.model}")
    print(f">>> Epochs: {cfg['epochs']}")
    print(f">>> Batch size: {cfg['batch_size']}")
    print(f">>> Learning rate: {cfg['lr']}")
    
    # Load datasets
    print(">>> Loading datasets...")
    train_set = cfg["train_dataset"]()
    test_set = cfg["test_dataset"]()
    print(f">>> Train samples: {len(train_set)}")
    print(f">>> Val samples: {len(test_set)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )
    
    # Create model
    print(">>> Creating model...")
    model = cfg["model_cls"]().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Parameters: {total_params:,}")

    if hasattr(torch, "compile"):
        print(">>> Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune")
    else:
        print(">>> torch.compile is not supported in this PyTorch version.")
    
    # Loss function
    criterion = cfg["criterion"]
    
    # Optimizer
    if cfg["optimizer_type"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            betas=(0.9, 0.999),
            eps=1e-3,
            weight_decay=3e-4,
            amsgrad=False,
        )
    
    # Scheduler
    if cfg["scheduler_type"] == "cosine":
        warmup_steps = 2000
        total_steps = len(train_loader) * cfg["epochs"]
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Logging setup
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    batch_csv_path = os.path.join(log_dir, f"{args.model}_batch_{ts}.csv")
    epoch_csv_path = os.path.join(log_dir, f"{args.model}_epoch_{ts}.csv")
    val_csv_path = os.path.join(log_dir, f"{args.model}_val_{ts}.csv")
    
    batch_csv_file = open(batch_csv_path, "w", newline="")
    epoch_csv_file = open(epoch_csv_path, "w", newline="")
    val_csv_file = open(val_csv_path, "w", newline="")
    
    batch_writer = csv.writer(batch_csv_file)
    epoch_writer = csv.writer(epoch_csv_file)
    val_writer = csv.writer(val_csv_file)
    
    batch_writer.writerow(["epoch", "step", "loss", "accuracy_pct", "time_ms"])
    epoch_writer.writerow(["epoch", "train_loss", "train_accuracy_pct", "val_loss", "val_accuracy_pct"])
    val_writer.writerow(["epoch", "step", "loss", "accuracy_pct"])
    
    # Training loop
    print(f"\n>>> Starting training for {cfg['epochs']} epochs...")
    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\n===== Epoch {epoch}/{cfg['epochs']} =====")
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, cfg,
            epoch, batch_writer
        )
        batch_csv_file.flush()
        
        # Learning rate schedule step
        if cfg["scheduler_type"] == "cosine":
            # For cosine, step after each batch (already done in train_epoch if needed)
            # But for simplicity, we'll step per epoch here
            pass
        else:
            scheduler.step()
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, cfg,
            epoch, val_writer
        )
        val_csv_file.flush()
        
        epoch_time = time.time() - epoch_start
        
        # Log epoch summary
        epoch_writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.4f}",
            f"{val_loss:.6f}",
            f"{val_acc:.4f}"
        ])
        epoch_csv_file.flush()
        
        print(
            f"Epoch {epoch}/{cfg['epochs']} completed in {epoch_time:.2f}s\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%"
        )
        
        if cfg["is_language_model"]:
            perplexity = math.exp(min(val_loss, 20))
            print(f"Val Perplexity: {perplexity:.2f}")
    
    # Cleanup
    batch_csv_file.close()
    epoch_csv_file.close()
    val_csv_file.close()
    
    print(f"\n>>> Logs saved to {log_dir}/{args.model}_*_{ts}.csv")
    print(f">>> Training completed for {args.model}!")


if __name__ == "__main__":
    main()
