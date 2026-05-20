#!/usr/bin/env python3
"""
Unified PyTorch inference script.
Usage: python torch_inference.py --model resnet9_cifar10
       python torch_inference.py --model gpt2_small --prompt "Hello world"
"""
import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from dotenv import load_dotenv

load_dotenv()

# Try to import tiktoken for GPT-2 text generation
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


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
                self.targets.append(arr[:, 0])
                self.data.append(arr[:, 1:].reshape(-1, 3, 32, 32))
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].astype(np.float32) / 255.0)
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


class CIFAR100Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        fname = "train.bin" if train else "test.bin"
        path = os.path.join(root, fname)
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3074)
        self.targets = arr[:, 1].astype(np.int64)
        self.data = arr[:, 2:].reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].astype(np.float32) / 255.0)
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


class TinyImageNetDataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None):
        self.transform = transform
        root = Path(root)
        self.samples = []
        train_dir = root / "train"
        classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        if train:
            for cls in classes:
                img_dir = train_dir / cls / "images"
                if not img_dir.exists():
                    continue
                idx = self.class_to_idx[cls]
                for p in img_dir.glob("*.JPEG"):
                    self.samples.append((str(p), idx))
        else:
            val_dir = root / "val"
            annotations = val_dir / "val_annotations.txt"
            with open(annotations) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img_name, cls = parts[0], parts[1]
                    img_path = val_dir / "images" / img_name
                    if img_path.exists() and cls in self.class_to_idx:
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
    def __init__(self, root: str, train: bool = True, transform=None):
        self.transform = transform
        root = Path(root)
        self.samples = []
        all_classes = set()
        for sub in ["train.X1", "train.X2", "train.X3", "train.X4"]:
            train_dir = root / sub
            if train_dir.exists():
                all_classes.update(d.name for d in train_dir.iterdir() if d.is_dir())
        classes = sorted(all_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        if train:
            for sub in ["train.X1", "train.X2", "train.X3", "train.X4"]:
                data_dir = root / sub
                if not data_dir.exists():
                    continue
                for cls in classes:
                    img_dir = data_dir / cls
                    if not img_dir.exists():
                        continue
                    idx = self.class_to_idx[cls]
                    for ext in ["*.JPEG", "*.jpg", "*.png"]:
                        for p in img_dir.glob(ext):
                            self.samples.append((str(p), idx))
        else:
            data_dir = root / "val.X"
            for cls in classes:
                img_dir = data_dir / cls
                if not img_dir.exists():
                    continue
                idx = self.class_to_idx[cls]
                for ext in ["*.JPEG", "*.jpg", "*.png"]:
                    for p in img_dir.glob(ext):
                        self.samples.append((str(p), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class OpenWebTextBinDataset(Dataset):
    def __init__(self, path, seq_len, dtype="uint16"):
        self.path = Path(path)
        self.seq_len = seq_len
        np_dtype = np.uint16 if dtype == "uint16" else np.int32
        self.tokens = np.fromfile(self.path, dtype=np_dtype)
        self.num_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.tensor(self.tokens[start:start + self.seq_len].astype(np.int64), dtype=torch.long)
        y = torch.tensor(self.tokens[start + 1:start + self.seq_len + 1].astype(np.int64), dtype=torch.long)
        return x, y


# =========================
# Transforms
# =========================

_CIFAR10_MEAN = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3, 1, 1)
_CIFAR10_STD = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3, 1, 1)

_CIFAR100_MEAN = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(3, 1, 1)
_CIFAR100_STD = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(3, 1, 1)

_TINY_MEAN = [0.4802, 0.4481, 0.3975]
_TINY_STD = [0.2770, 0.2691, 0.2821]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _cifar10_transform(img):
    return (img - _CIFAR10_MEAN) / _CIFAR10_STD


def _cifar100_transform(img):
    return (img - _CIFAR100_MEAN) / _CIFAR100_STD


_tiny_imagenet_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(_TINY_MEAN, _TINY_STD),
])

_imagenet100_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


# =========================
# Models
# =========================

class _BasicResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ResNet9CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.res1 = _BasicResidualBlock(128)
        self.res2 = _BasicResidualBlock(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.res3 = _BasicResidualBlock(256)
        self.res4 = _BasicResidualBlock(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.res5 = _BasicResidualBlock(512)
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
        return self.fc(self.flatten(x))


class _WideResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)

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
    def __init__(self, num_classes: int = 100):
        super().__init__()
        width_factor = 8
        dropout_rate = 0.3
        c1, c2, c3 = 16 * width_factor, 32 * width_factor, 64 * width_factor
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=True)
        self.group1_block1 = _WideResidualBlock(16, c1, stride=1, dropout_rate=dropout_rate)
        self.group1_block2 = _WideResidualBlock(c1, c1, stride=1, dropout_rate=dropout_rate)
        self.group2_block1 = _WideResidualBlock(c1, c2, stride=2, dropout_rate=dropout_rate)
        self.group2_block2 = _WideResidualBlock(c2, c2, stride=1, dropout_rate=dropout_rate)
        self.group3_block1 = _WideResidualBlock(c2, c3, stride=2, dropout_rate=dropout_rate)
        self.group3_block2 = _WideResidualBlock(c3, c3, stride=1, dropout_rate=dropout_rate)
        self.bn_final = nn.BatchNorm2d(c3, eps=1e-5, momentum=0.1)
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
        return self.fc(x)


class _BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        out = out + sc
        out = F.relu(out, inplace=True)
        return out


class ResNet50TinyImageNet(nn.Module):
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            _BottleneckBlock(64, 64, 256, stride=1),
            _BottleneckBlock(256, 64, 256, stride=1),
            _BottleneckBlock(256, 64, 256, stride=1))
        self.layer2 = nn.Sequential(
            _BottleneckBlock(256, 128, 512, stride=2),
            _BottleneckBlock(512, 128, 512, stride=1),
            _BottleneckBlock(512, 128, 512, stride=1),
            _BottleneckBlock(512, 128, 512, stride=1))
        self.layer3 = nn.Sequential(
            _BottleneckBlock(512, 256, 1024, stride=2),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1))
        self.layer4 = nn.Sequential(
            _BottleneckBlock(1024, 512, 2048, stride=2),
            _BottleneckBlock(2048, 512, 2048, stride=1),
            _BottleneckBlock(2048, 512, 2048, stride=1))
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
        return self.fc(x)


class ResNet50ImageNet100(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            _BottleneckBlock(64, 64, 256, stride=1),
            _BottleneckBlock(256, 64, 256, stride=1),
            _BottleneckBlock(256, 64, 256, stride=1))
        self.layer2 = nn.Sequential(
            _BottleneckBlock(256, 128, 512, stride=2),
            _BottleneckBlock(512, 128, 512, stride=1),
            _BottleneckBlock(512, 128, 512, stride=1),
            _BottleneckBlock(512, 128, 512, stride=1))
        self.layer3 = nn.Sequential(
            _BottleneckBlock(512, 256, 1024, stride=2),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1))
        self.layer4 = nn.Sequential(
            _BottleneckBlock(1024, 512, 2048, stride=2),
            _BottleneckBlock(2048, 512, 2048, stride=1),
            _BottleneckBlock(2048, 512, 2048, stride=1))
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
        return self.fc(x)


# GPT-2 constants
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
        self.embed_dim = embed_dim
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.embed_dim, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = F.softmax(att.masked_fill(mask, float("-inf")), dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class _GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attn = _CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-5)
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


class GPT2Small(nn.Module):
    def __init__(self, vocab_size=_GPT2_VOCAB_SIZE, seq_len=1024, embed_dim=_GPT2_EMBED_DIM,
                 num_heads=_GPT2_NUM_HEADS, num_layers=_GPT2_NUM_LAYERS, ffn_dim=_GPT2_FFN_DIM, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[_GPTBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim, eps=1e-5)
        self.head = nn.Linear(embed_dim, vocab_size, bias=True)
        self.head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :T, :]
        h = self.emb_drop(h)
        h = self.blocks(h)
        h = self.ln_f(h)
        return self.head(h)


# =========================
# Config
# =========================

def _make_inference_configs():
    cifar10_root = os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin")
    cifar100_root = os.getenv("CIFAR100_BIN_ROOT", "data/cifar-100-binary")
    tiny_imagenet_root = os.getenv("TINY_IMAGENET_ROOT", "data/tiny-imagenet-200")
    imgnet100_root = os.getenv("IMAGENET100_ROOT", "data/imagenet-100")

    return {
        "resnet9_cifar10": {
            "model_cls": ResNet9CIFAR10,
            "model_kwargs": {"num_classes": 10},
            "dataset": lambda: CIFAR10Bin(cifar10_root, train=False, transform=_cifar10_transform),
            "batch_size": int(os.getenv("BATCH_SIZE", "256")),
            "num_workers": 2,
            "model_path": os.getenv("MODEL_PATH", "model_snapshots/resnet9_cifar10.pth"),
            "is_gpt2": False,
        },
        "wrn16_8_cifar100": {
            "model_cls": WRN16_8CIFAR100,
            "model_kwargs": {"num_classes": 100},
            "dataset": lambda: CIFAR100Bin(cifar100_root, train=False, transform=_cifar100_transform),
            "batch_size": int(os.getenv("BATCH_SIZE", "256")),
            "num_workers": 2,
            "model_path": os.getenv("MODEL_PATH", "model_snapshots/wrn16_8_cifar100.pth"),
            "is_gpt2": False,
        },
        "resnet50_tiny_imagenet": {
            "model_cls": ResNet50TinyImageNet,
            "model_kwargs": {"num_classes": 200},
            "dataset": lambda: TinyImageNetDataset(tiny_imagenet_root, train=False, transform=_tiny_imagenet_transform),
            "batch_size": int(os.getenv("BATCH_SIZE", "128")),
            "num_workers": 4,
            "model_path": os.getenv("MODEL_PATH", "model_snapshots/resnet50_tiny_imagenet.pth"),
            "is_gpt2": False,
        },
        "resnet50_imagenet100": {
            "model_cls": ResNet50ImageNet100,
            "model_kwargs": {"num_classes": 100},
            "dataset": lambda: ImageNet100Dataset(imgnet100_root, train=False, transform=_imagenet100_transform),
            "batch_size": int(os.getenv("BATCH_SIZE", "128")),
            "num_workers": 4,
            "model_path": os.getenv("MODEL_PATH", "model_snapshots/resnet50_imagenet100.pth"),
            "is_gpt2": False,
        },
        "gpt2_small": {
            "model_cls": GPT2Small,
            "model_kwargs": {},
            "dataset": lambda: OpenWebTextBinDataset(
                path=os.getenv("OPENWEBTEXT_VAL_BIN", "data/open-web-text-1pct/val.bin"),
                seq_len=1024,
                dtype=os.getenv("OPENWEBTEXT_BIN_DTYPE", "uint16"),
            ),
            "batch_size": int(os.getenv("BATCH_SIZE", "8")),
            "num_workers": 0,
            "model_path": os.getenv("MODEL_PATH", "model_snapshots/gpt2_small.pth"),
            "is_gpt2": True,
        },
    }


# =========================
# GPT-2 Generation Helpers
# =========================

def top_k_sampling(logits: torch.Tensor, top_k: int = 50, temperature: float = 1.0) -> int:
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[-1]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def greedy_decode(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=-1).item())


def generate_text(model, prompt_text, device, max_new=100, top_k=50, temperature=1.0):
    """Generate text from GPT-2 model."""
    if not _TIKTOKEN_AVAILABLE:
        print(">>> tiktoken not installed — skipping text generation.")
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
            context = current_ids[-1024:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1, :]

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


def compute_perplexity(model, dataset, device, batch_size):
    """Compute perplexity on dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), targets.view(B * T))
            total_loss += loss.item() * B * T
            total_tokens += B * T

    return math.exp(total_loss / max(1, total_tokens))


# =========================
# Classification Inference
# =========================

def run_classification_inference(model, dataset, device, batch_size, num_workers):
    """Run classification inference and compute accuracy."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    total = 0

    inference_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 10 == 0:
                batch_acc = 100.0 * predicted.eq(targets).sum().item() / targets.size(0)
                print(f"[Inference Batch {batch_idx + 1}/{len(loader)}] Batch Acc: {batch_acc:.2f}%")

    inference_time = time.time() - inference_start
    test_loss = loss_sum / total
    test_acc = 100.0 * correct / total

    print(f"\n>>> Inference completed in {inference_time:.2f}s")
    print(f">>> Test Loss: {test_loss:.4f}")
    print(f">>> Test Accuracy: {test_acc:.2f}%")
    print(f">>> Throughput: {total / inference_time:.2f} samples/sec")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["resnet9_cifar10", "wrn16_8_cifar100", "resnet50_tiny_imagenet",
                               "resnet50_imagenet100", "gpt2_small"])
    parser.add_argument("--prompt", type=str, default="The quick brown fox",
                       help="Prompt for GPT-2 text generation")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                       help="Maximum new tokens to generate for GPT-2")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling for GPT-2 (use 1 for greedy)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for GPT-2 sampling")
    parser.add_argument("--eval-perplexity", action="store_true",
                       help="Compute perplexity for GPT-2")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">>> Running inference on device: {device}")
    print(f">>> Model: {args.model}")

    cfg = _make_inference_configs()[args.model]

    # Create model
    model = cfg["model_cls"](**cfg["model_kwargs"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Parameters: {total_params:,}")

    # Load checkpoint
    model_path = cfg["model_path"]
    if os.path.isfile(model_path):
        print(f">>> Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "epoch" in checkpoint:
                print(f"    (checkpoint from epoch {checkpoint['epoch']})")
        else:
            model.load_state_dict(checkpoint)
        print(">>> Weights loaded successfully")
    else:
        print(f">>> Warning: model file not found at {model_path}")
        print(">>> Running with randomly initialized weights")

    # Run inference
    if cfg["is_gpt2"]:
        # GPT-2 text generation
        if args.eval_perplexity:
            print("\n>>> Computing perplexity...")
            dataset = cfg["dataset"]()
            ppl_start = time.time()
            ppl = compute_perplexity(model, dataset, device, cfg["batch_size"])
            print(f">>> Perplexity: {ppl:.2f} ({time.time() - ppl_start:.2f}s)")

        generate_text(model, args.prompt, device,
                     max_new=args.max_new_tokens,
                     top_k=args.top_k,
                     temperature=args.temperature)
    else:
        # Classification inference
        dataset = cfg["dataset"]()
        print(f">>> Test samples: {len(dataset)}")
        run_classification_inference(model, dataset, device, cfg["batch_size"], cfg["num_workers"])


if __name__ == "__main__":
    main()
