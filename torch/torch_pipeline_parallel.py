"""
Pipeline-parallel training for multiple model/dataset combinations.

Usage (2-node, 1 GPU each):
  export GLOO_SOCKET_IFNAME=lo
  export NCCL_SOCKET_IFNAME=lo
  torchrun \\
    --nproc_per_node=1 --nnodes=2 --node_rank=[RANK] \\
    --rdzv_id=pipeline --rdzv_backend=c10d --rdzv_endpoint=[ENDPOINT] \\
    torch/torch_pipeline_parallel.py --model [MODEL]

Models:
  gpt2                  GPT-2 Small (synthetic data)
  resnet9_cifar10       ResNet-9 on CIFAR-10     (env: CIFAR10_BIN_ROOT)
  wrn16_8_cifar100      WRN-16-8 on CIFAR-100    (env: CIFAR100_BIN_ROOT)
  resnet50_tiny_imagenet  ResNet-50 on Tiny ImageNet (env: TINY_IMAGENET_ROOT)
  resnet50_imagenet100  ResNet-50 on ImageNet-100  (env: IMAGENET100_ROOT)

Model-specific env vars (override defaults matching each standalone script):
  EPOCHS, BATCH_SIZE, LR_INITIAL
"""

import argparse
import math
import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import Schedule1F1B

NUM_STAGES = 2


# =============================================================================
# CIFAR-10 Dataset  (binary format, matches torch_resnet9_cifar10.py)
# =============================================================================

class CIFAR10Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []
        batch_files = [f"data_batch_{i}.bin" for i in range(1, 6)] if train else ["test_batch.bin"]
        for fname in batch_files:
            path = os.path.join(root, fname)
            with open(path, "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3073)
                self.data.append(arr[:, 1:].reshape(-1, 3, 32, 32))
                self.targets.append(arr[:, 0])
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].astype(np.float32) / 255.0)
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


_CIFAR10_MEAN = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3, 1, 1)
_CIFAR10_STD  = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3, 1, 1)

def _cifar10_transform(img):
    return (img - _CIFAR10_MEAN) / _CIFAR10_STD


# =============================================================================
# ResNet-9  (matches torch_resnet9_cifar10.py)
# =============================================================================

class _BasicResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ResNet9Stage0(nn.Module):
    """conv1->conv2->maxpool->res1->res2  |  output: [B, 128, 16, 16]"""
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.bn1     = nn.BatchNorm2d(64,  eps=1e-5, momentum=0.1)
        self.conv2   = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True)
        self.bn2     = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.res1    = _BasicResidualBlock(128)
        self.res2    = _BasicResidualBlock(128)

    def forward(self, x):                               # [B, 3, 32, 32]
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # [B, 64, 32, 32]
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)  # [B, 128, 32, 32]
        x = self.maxpool(x)                                 # [B, 128, 16, 16]
        x = self.res1(x)
        x = self.res2(x)
        return x                                            # [B, 128, 16, 16]


class ResNet9Stage1(nn.Module):
    """conv3->res3->res4->conv4->res5->head  |  input: [B, 128, 16, 16]"""
    def __init__(self):
        super().__init__()
        self.conv3    = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True)
        self.bn3      = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.res3     = _BasicResidualBlock(256)
        self.res4     = _BasicResidualBlock(256)
        self.conv4    = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=True)
        self.bn4      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.res5     = _BasicResidualBlock(512)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten  = nn.Flatten()
        self.fc       = nn.Linear(512, 10, bias=True)

    def forward(self, x):                                   # [B, 128, 16, 16]
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)  # [B, 256, 16, 16]
        x = self.maxpool2(x)                                # [B, 256,  8,  8]
        x = self.res3(x)
        x = self.res4(x)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)  # [B, 512,  8,  8]
        x = self.maxpool3(x)                                # [B, 512,  4,  4]
        x = self.res5(x)
        x = self.avgpool(x)                                 # [B, 512,  1,  1]
        return self.fc(self.flatten(x))                     # [B, 10]


# =============================================================================
# CIFAR-100 Dataset  (binary format, matches torch_wrn16_8_cifar100.py)
# =============================================================================

class CIFAR100Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        fname = "train.bin" if train else "test.bin"
        path = os.path.join(root, fname)
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3074)
        self.targets = arr[:, 1].astype(np.int64)          # fine labels
        self.data    = arr[:, 2:].reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].astype(np.float32) / 255.0)
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[idx])


_CIFAR100_MEAN = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(3, 1, 1)
_CIFAR100_STD  = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(3, 1, 1)

def _cifar100_transform(img):
    return (img - _CIFAR100_MEAN) / _CIFAR100_STD


# =============================================================================
# WRN-16-8  (matches torch_wrn16_8_cifar100.py)
# =============================================================================

class _WideResidualBlock(nn.Module):
    """Pre-activation wide residual block."""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_channels,  eps=1e-5, momentum=0.1)
        self.conv1 = nn.Conv2d(in_channels,  out_channels, 3,
                               stride=stride, padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=True)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1,
                                      stride=stride, padding=0, bias=False)

    def forward(self, x):
        sc  = self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        return out + sc


class WRN16_8Stage0(nn.Module):
    """conv1->group1->group2  |  output: [B, 256, 16, 16]"""
    def __init__(self):
        super().__init__()
        dr = 0.3
        self.conv1          = nn.Conv2d(3,   16,  3, stride=1, padding=1, bias=True)
        self.group1_block1  = _WideResidualBlock( 16, 128, stride=1, dropout_rate=dr)
        self.group1_block2  = _WideResidualBlock(128, 128, stride=1, dropout_rate=dr)
        self.group2_block1  = _WideResidualBlock(128, 256, stride=2, dropout_rate=dr)
        self.group2_block2  = _WideResidualBlock(256, 256, stride=1, dropout_rate=dr)

    def forward(self, x):              # [B,   3, 32, 32]
        x = self.conv1(x)             # [B,  16, 32, 32]
        x = self.group1_block1(x)     # [B, 128, 32, 32]
        x = self.group1_block2(x)     # [B, 128, 32, 32]
        x = self.group2_block1(x)     # [B, 256, 16, 16]
        x = self.group2_block2(x)     # [B, 256, 16, 16]
        return x


class WRN16_8Stage1(nn.Module):
    """group3->bn_final->avgpool->fc  |  input: [B, 256, 16, 16]"""
    def __init__(self):
        super().__init__()
        dr = 0.3
        self.group3_block1 = _WideResidualBlock(256, 512, stride=2, dropout_rate=dr)
        self.group3_block2 = _WideResidualBlock(512, 512, stride=1, dropout_rate=dr)
        self.bn_final      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.avgpool       = nn.AvgPool2d(kernel_size=8, stride=1)
        self.flatten       = nn.Flatten()
        self.fc            = nn.Linear(512, 100, bias=True)

    def forward(self, x):                                    # [B, 256, 16, 16]
        x = self.group3_block1(x)                            # [B, 512,  8,  8]
        x = self.group3_block2(x)                            # [B, 512,  8,  8]
        x = F.relu(self.bn_final(x), inplace=True)           # [B, 512,  8,  8]
        x = self.avgpool(x)                                  # [B, 512,  1,  1]
        return self.fc(self.flatten(x))                      # [B, 100]


# =============================================================================
# Tiny ImageNet Dataset  (matches torch_resnet50_tiny_imagenet.py)
# =============================================================================

class TinyImageNetDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        root = Path(root)
        self.samples = []
        train_dir = root / "train"
        classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        if train:
            for cls in classes:
                img_dir = train_dir / cls / "images"
                if not img_dir.exists():
                    continue
                for p in img_dir.glob("*.JPEG"):
                    self.samples.append((str(p), class_to_idx[cls]))
        else:
            val_dir = root / "val"
            with open(val_dir / "val_annotations.txt") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img_path = val_dir / "images" / parts[0]
                    if img_path.exists() and parts[1] in class_to_idx:
                        self.samples.append((str(img_path), class_to_idx[parts[1]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


_TINY_MEAN = [0.4802, 0.4481, 0.3975]
_TINY_STD  = [0.2770, 0.2691, 0.2821]

_tiny_train_transform = T.Compose([T.ToTensor(), T.Normalize(_TINY_MEAN, _TINY_STD)])
_tiny_test_transform  = T.Compose([T.ToTensor(), T.Normalize(_TINY_MEAN, _TINY_STD)])


# =============================================================================
# ImageNet-100 Dataset  (matches torch_resnet50_imagenet100.py)
# =============================================================================

class ImageNet100Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        root = Path(root)
        self.samples = []
        all_classes = set()
        for sub in ["train.X1", "train.X2", "train.X3", "train.X4"]:
            d = root / sub
            if d.exists():
                all_classes.update(x.name for x in d.iterdir() if x.is_dir())
        classes = sorted(all_classes)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        if train:
            for sub in ["train.X1", "train.X2", "train.X3", "train.X4"]:
                data_dir = root / sub
                if not data_dir.exists():
                    continue
                for cls in classes:
                    img_dir = data_dir / cls
                    if not img_dir.exists():
                        continue
                    for ext in ["*.JPEG", "*.jpg", "*.png"]:
                        for p in img_dir.glob(ext):
                            self.samples.append((str(p), class_to_idx[cls]))
        else:
            data_dir = root / "val.X"
            for cls in classes:
                img_dir = data_dir / cls
                if not img_dir.exists():
                    continue
                for ext in ["*.JPEG", "*.jpg", "*.png"]:
                    for p in img_dir.glob(ext):
                        self.samples.append((str(p), class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_imagenet100_train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])
_imagenet100_test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


# =============================================================================
# Bottleneck Block (shared by both ResNet-50 variants)
# =============================================================================

class _BottleneckBlock(nn.Module):
    """Matches bottleneck_residual_block() in layer_builder.hpp."""
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  mid_channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            )

    def forward(self, x):
        sc  = self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(self.bn1(self.conv1(x)),  inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        out = out + sc
        return F.relu(out, inplace=True)


# =============================================================================
# ResNet-50 for Tiny ImageNet  (matches torch_resnet50_tiny_imagenet.py)
# =============================================================================

class ResNet50TinyStage0(nn.Module):
    """conv1->bn1->maxpool->layer1->layer2  |  output: [B, 512, 16, 16]"""
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.bn1     = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)   # 64->32
        self.layer1  = nn.Sequential(
            _BottleneckBlock( 64,  64, 256, stride=1),
            _BottleneckBlock(256,  64, 256, stride=1),
            _BottleneckBlock(256,  64, 256, stride=1),
        )
        self.layer2 = nn.Sequential(
            _BottleneckBlock(256, 128, 512, stride=2),         # 32->16
            _BottleneckBlock(512, 128, 512, stride=1),
            _BottleneckBlock(512, 128, 512, stride=1),
            _BottleneckBlock(512, 128, 512, stride=1),
        )

    def forward(self, x):                                      # [B,   3, 64, 64]
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)     # [B,  64, 64, 64]
        x = self.maxpool(x)                                    # [B,  64, 32, 32]
        x = self.layer1(x)                                     # [B, 256, 32, 32]
        x = self.layer2(x)                                     # [B, 512, 16, 16]
        return x


class ResNet50TinyStage1(nn.Module):
    """layer3->layer4->avgpool->fc  |  input: [B, 512, 16, 16]"""
    def __init__(self):
        super().__init__()
        self.layer3 = nn.Sequential(
            _BottleneckBlock( 512, 256, 1024, stride=2),       # 16->8
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
        )
        self.layer4 = nn.Sequential(
            _BottleneckBlock(1024, 512, 2048, stride=2),       # 8->4
            _BottleneckBlock(2048, 512, 2048, stride=1),
            _BottleneckBlock(2048, 512, 2048, stride=1),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(2048, 200, bias=True)

    def forward(self, x):                  # [B,  512, 16, 16]
        x = self.layer3(x)                 # [B, 1024,  8,  8]
        x = self.layer4(x)                 # [B, 2048,  4,  4]
        x = self.avgpool(x)                # [B, 2048,  1,  1]
        return self.fc(self.flatten(x))    # [B, 200]


# =============================================================================
# ResNet-50 for ImageNet-100  (matches torch_resnet50_imagenet100.py)
# =============================================================================

class ResNet50ImgNet100Stage0(nn.Module):
    """conv1->bn1->maxpool->layer1->layer2  |  output: [B, 512, 28, 28]"""
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True)
        self.bn1     = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)   # 224->112->56
        self.layer1  = nn.Sequential(
            _BottleneckBlock( 64,  64, 256, stride=1),
            _BottleneckBlock(256,  64, 256, stride=1),
            _BottleneckBlock(256,  64, 256, stride=1),
        )
        self.layer2 = nn.Sequential(
            _BottleneckBlock(256, 128, 512, stride=2),         # 56->28
            _BottleneckBlock(512, 128, 512, stride=1),
            _BottleneckBlock(512, 128, 512, stride=1),
            _BottleneckBlock(512, 128, 512, stride=1),
        )

    def forward(self, x):                                      # [B,   3, 224, 224]
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)     # [B,  64, 112, 112]
        x = self.maxpool(x)                                    # [B,  64,  56,  56]
        x = self.layer1(x)                                     # [B, 256,  56,  56]
        x = self.layer2(x)                                     # [B, 512,  28,  28]
        return x


class ResNet50ImgNet100Stage1(nn.Module):
    """layer3->layer4->avgpool->fc  |  input: [B, 512, 28, 28]"""
    def __init__(self):
        super().__init__()
        self.layer3 = nn.Sequential(
            _BottleneckBlock( 512, 256, 1024, stride=2),       # 28->14
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
            _BottleneckBlock(1024, 256, 1024, stride=1),
        )
        self.layer4 = nn.Sequential(
            _BottleneckBlock(1024, 512, 2048, stride=2),       # 14->7
            _BottleneckBlock(2048, 512, 2048, stride=1),
            _BottleneckBlock(2048, 512, 2048, stride=1),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(2048, 100, bias=True)

    def forward(self, x):                  # [B,  512, 28, 28]
        x = self.layer3(x)                 # [B, 1024, 14, 14]
        x = self.layer4(x)                 # [B, 2048,  7,  7]
        x = self.avgpool(x)                # [B, 2048,  1,  1]
        return self.fc(self.flatten(x))    # [B, 100]


# =============================================================================
# GPT-2 Small  (synthetic data, matches original pipeline script)
# =============================================================================

_GPT2_VOCAB_SIZE  = 50257
_GPT2_EMBED_DIM   = 768
_GPT2_NUM_HEADS   = 12
_GPT2_NUM_LAYERS  = 12
_GPT2_FFN_DIM     = _GPT2_EMBED_DIM * 4


class _CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att  = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att  = F.softmax(att.masked_fill(mask, float('-inf')), dim=-1)
        y    = att @ v
        y    = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class _GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = _CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp  = nn.Sequential(
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
    """Embeddings + first 6 transformer blocks  |  output: [B, T, 768]"""
    def __init__(self, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(_GPT2_VOCAB_SIZE, _GPT2_EMBED_DIM)
        self.pos_emb   = nn.Parameter(torch.zeros(1, 1024, _GPT2_EMBED_DIM))
        self.blocks    = nn.Sequential(*[
            _GPTBlock(_GPT2_EMBED_DIM, _GPT2_NUM_HEADS, _GPT2_FFN_DIM)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :T, :]
        return self.blocks(h)


class GPT2Stage1(nn.Module):
    """Last 6 transformer blocks + LM head  |  input: [B, T, 768]"""
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


class _GPT2SyntheticDataset(Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        x = torch.randint(0, _GPT2_VOCAB_SIZE, (self.seq_len,))
        return x, x


# =============================================================================
# Model configuration registry
# =============================================================================
# Each entry specifies how to build stages, load datasets, and configure
# training — mirroring the standalone scripts exactly.

def _make_model_configs(seq_len):
    def _image_loss(logits, labels):
        return F.cross_entropy(logits, labels)

    def _gpt2_loss(logits, labels):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    cifar10_root  = os.getenv("CIFAR10_BIN_ROOT",   "data/cifar-10-batches-bin")
    cifar100_root = os.getenv("CIFAR100_BIN_ROOT",  "data/cifar-100-binary")
    tiny_root     = os.getenv("TINY_IMAGENET_ROOT",  "data/tiny-imagenet-200")
    imgnet100_root= os.getenv("IMAGENET100_ROOT",    "data/imagenet-100")

    return {
        "gpt2": {
            "stage0_cls":  lambda: GPT2Stage0(_GPT2_NUM_LAYERS // 2),
            "stage1_cls":  lambda: GPT2Stage1(_GPT2_NUM_LAYERS // 2),
            "train_set":   lambda: _GPT2SyntheticDataset(seq_len),
            "loss_fn":     _gpt2_loss,
            "batch_size":  int(os.getenv("BATCH_SIZE", "16")),
            "epochs":      int(os.getenv("EPOCHS",     "1")),
            "lr":          float(os.getenv("LR_INITIAL","0.0001")),
            "num_workers": 0,
        },
        "resnet9_cifar10": {
            "stage0_cls":  ResNet9Stage0,
            "stage1_cls":  ResNet9Stage1,
            "train_set":   lambda: CIFAR10Bin(cifar10_root, train=True,
                                              transform=_cifar10_transform),
            "loss_fn":     _image_loss,
            "batch_size":  int(os.getenv("BATCH_SIZE", "128")),
            "epochs":      int(os.getenv("EPOCHS",     "10")),
            "lr":          float(os.getenv("LR_INITIAL","0.001")),
            "num_workers": 2,
        },
        "wrn16_8_cifar100": {
            "stage0_cls":  WRN16_8Stage0,
            "stage1_cls":  WRN16_8Stage1,
            "train_set":   lambda: CIFAR100Bin(cifar100_root, train=True,
                                               transform=_cifar100_transform),
            "loss_fn":     _image_loss,
            "batch_size":  int(os.getenv("BATCH_SIZE", "128")),
            "epochs":      int(os.getenv("EPOCHS",     "50")),
            "lr":          float(os.getenv("LR_INITIAL","0.001")),
            "num_workers": 2,
        },
        "resnet50_tiny_imagenet": {
            "stage0_cls":  ResNet50TinyStage0,
            "stage1_cls":  ResNet50TinyStage1,
            "train_set":   lambda: TinyImageNetDataset(tiny_root, train=True,
                                                       transform=_tiny_train_transform),
            "loss_fn":     _image_loss,
            "batch_size":  int(os.getenv("BATCH_SIZE", "128")),
            "epochs":      int(os.getenv("EPOCHS",     "90")),
            "lr":          float(os.getenv("LR_INITIAL","0.001")),
            "num_workers": 4,
        },
        "resnet50_imagenet100": {
            "stage0_cls":  ResNet50ImgNet100Stage0,
            "stage1_cls":  ResNet50ImgNet100Stage1,
            "train_set":   lambda: ImageNet100Dataset(imgnet100_root, train=True,
                                                      transform=_imagenet100_train_transform),
            "loss_fn":     _image_loss,
            "batch_size":  int(os.getenv("BATCH_SIZE", "64")),
            "epochs":      int(os.getenv("EPOCHS",     "90")),
            "lr":          float(os.getenv("LR_INITIAL","0.001")),
            "num_workers": 4,
        },
    }


# =============================================================================
# Main Training Logic
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline-parallel training")
    parser.add_argument(
        "--model", type=str, default="gpt2",
        choices=["gpt2", "resnet9_cifar10", "wrn16_8_cifar100",
                 "resnet50_tiny_imagenet", "resnet50_imagenet100"],
        help="Model/dataset combination to train",
    )
    parser.add_argument("--micro-bs", type=int, default=4,
                        help="Micro-batch size for pipeline schedule")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="GPT-2 sequence length (ignored for image models)")
    args = parser.parse_args()

    #  Distributed init 
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    cfg = _make_model_configs(args.seq_len)[args.model]

    print(f"[Rank {rank}] model={args.model}  device={device}")

    #  Build model stage 
    model = cfg["stage0_cls"]() if rank == 0 else cfg["stage1_cls"]()
    model.to(device)

    stage = PipelineStage(
        submodule=model,
        stage_index=rank,
        num_stages=NUM_STAGES,
        device=device,
    )

    #  Optimizer & scheduler (identical to each standalone script) 
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        eps=1e-3,
        weight_decay=3e-4,
        amsgrad=False,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    #  Dataset 
    # Both ranks load the same dataset and iterate in lock-step.
    # Rank 0 feeds x into the pipeline; rank 1 uses y as the loss target.
    batch_size = cfg["batch_size"]
    dataset    = cfg["train_set"]()
    loader     = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
    )

    #  Pipeline schedule 
    num_microbatches = batch_size // args.micro_bs
    schedule = Schedule1F1B(
        stage,
        n_microbatches=num_microbatches,
        loss_fn=cfg["loss_fn"] if rank == 1 else None,
    )

    #  Training loop 
    model.train()
    epochs = cfg["epochs"]
    for epoch in range(epochs):
        data_iter = iter(loader)
        num_steps = len(dataset) // batch_size

        for step in range(num_steps):
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            if rank == 0:
                schedule.step(x)
            else:  # rank == 1
                losses = []
                schedule.step(target=y, losses=losses)
                if step % 5 == 0 and losses:
                    avg_loss = sum(l.item() for l in losses) / len(losses)
                    print(f"[Rank {rank}] Epoch {epoch+1}/{epochs} "
                          f"Step {step}/{num_steps} | Loss: {avg_loss:.4f}")

            optimizer.step()

        scheduler.step()
        if rank == 1:
            print(f"[Rank {rank}] Epoch {epoch+1}/{epochs} complete.")

    print(f"[Rank {rank}] Training complete.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()