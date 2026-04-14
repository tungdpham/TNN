
#!/usr/bin/env python3
import argparse, csv, math, os, time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def ensure_dir(path):
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
            self.f.flush(); self.f.close()
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
            return int((base/"tx_bytes").read_text().strip()), int((base/"rx_bytes").read_text().strip())
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
        return dict(ifname=self.ifname or "", dt_sec=dt, tx_bytes_delta=dtx, rx_bytes_delta=drx, tx_MBps=dtx/dt/(1024*1024), rx_MBps=drx/dt/(1024*1024), counter_ok=1)

class CIFAR10Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.data, self.targets = [], []
        batch_files = [f"data_batch_{i}.bin" for i in range(1, 6)] if train else ["test_batch.bin"]
        for fname in batch_files:
            path = os.path.join(root, fname)
            with open(path, "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3073)
                self.data.append(arr[:,1:].reshape(-1,3,32,32))
                self.targets.append(arr[:,0])
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].astype(np.float32)/255.0)
        if self.transform: img = self.transform(img)
        return img, int(self.targets[idx])

_CIFAR10_MEAN = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3,1,1)
_CIFAR10_STD  = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3,1,1)
def _cifar10_transform(img): return (img - _CIFAR10_MEAN) / _CIFAR10_STD

class CIFAR100Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        fname = "train.bin" if train else "test.bin"
        path = os.path.join(root, fname)
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3074)
        self.targets = arr[:, 1].astype(np.int64)
        self.data = arr[:, 2:].reshape(-1, 3, 32, 32)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].astype(np.float32)/255.0)
        if self.transform: img = self.transform(img)
        return img, int(self.targets[idx])

_CIFAR100_MEAN = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(3,1,1)
_CIFAR100_STD  = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(3,1,1)
def _cifar100_transform(img): return (img - _CIFAR100_MEAN) / _CIFAR100_STD

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
                if not img_dir.exists(): continue
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
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

_TINY_MEAN = [0.4802, 0.4481, 0.3975]
_TINY_STD  = [0.2770, 0.2691, 0.2821]
_tiny_train_transform = T.Compose([T.ToTensor(), T.Normalize(_TINY_MEAN, _TINY_STD)])
_tiny_test_transform  = T.Compose([T.ToTensor(), T.Normalize(_TINY_MEAN, _TINY_STD)])

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
                if not data_dir.exists(): continue
                for cls in classes:
                    img_dir = data_dir / cls
                    if not img_dir.exists(): continue
                    for ext in ["*.JPEG", "*.jpg", "*.png"]:
                        for p in img_dir.glob(ext):
                            self.samples.append((str(p), class_to_idx[cls]))
        else:
            data_dir = root / "val.X"
            for cls in classes:
                img_dir = data_dir / cls
                if not img_dir.exists(): continue
                for ext in ["*.JPEG", "*.jpg", "*.png"]:
                    for p in img_dir.glob(ext):
                        self.samples.append((str(p), class_to_idx[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_imagenet100_train_transform = T.Compose([
    T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
_imagenet100_test_transform = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])

_GPT2_VOCAB_SIZE = 50257
_GPT2_EMBED_DIM = 768
_GPT2_NUM_HEADS = 12
_GPT2_NUM_LAYERS = 12
_GPT2_FFN_DIM = _GPT2_EMBED_DIM * 4

class _GPT2SyntheticDataset(Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
    def __len__(self): return 1000
    def __getitem__(self, idx):
        x = torch.randint(0, _GPT2_VOCAB_SIZE, (self.seq_len,))
        return x, x

class _BasicResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(channels)
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
        self.maxpool = nn.MaxPool2d(2,2)
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
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.res3 = _BasicResidualBlock(256)
        self.res4 = _BasicResidualBlock(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(512)
        self.maxpool3 = nn.MaxPool2d(2,2)
        self.res5 = _BasicResidualBlock(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
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

class _WideResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
    def forward(self, x):
        sc = self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout is not None: out = self.dropout(out)
        out = self.conv2(out)
        return out + sc

class WRN16_8Stage0(nn.Module):
    def __init__(self):
        super().__init__()
        dr = 0.3
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=True)
        self.group1_block1 = _WideResidualBlock(16, 128, stride=1, dropout_rate=dr)
        self.group1_block2 = _WideResidualBlock(128, 128, stride=1, dropout_rate=dr)
        self.group2_block1 = _WideResidualBlock(128, 256, stride=2, dropout_rate=dr)
        self.group2_block2 = _WideResidualBlock(256, 256, stride=1, dropout_rate=dr)
    def forward(self, x):
        x = self.conv1(x)
        x = self.group1_block1(x)
        x = self.group1_block2(x)
        x = self.group2_block1(x)
        x = self.group2_block2(x)
        return x

class WRN16_8Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        dr = 0.3
        self.group3_block1 = _WideResidualBlock(256, 512, stride=2, dropout_rate=dr)
        self.group3_block2 = _WideResidualBlock(512, 512, stride=1, dropout_rate=dr)
        self.bn_final = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 100, bias=True)
    def forward(self, x):
        x = self.group3_block1(x)
        x = self.group3_block2(x)
        x = F.relu(self.bn_final(x), inplace=True)
        x = self.avgpool(x)
        return self.fc(self.flatten(x))

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
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1))
    def forward(self, x):
        sc = self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        out = out + sc
        return F.relu(out, inplace=True)

class ResNet50TinyStage0(nn.Module):
    def __init__(self):
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
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class ResNet50TinyStage1(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.fc = nn.Linear(2048, 200, bias=True)
    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(self.flatten(x))

class ResNet50ImgNet100Stage0(nn.Module):
    def __init__(self):
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
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class ResNet50ImgNet100Stage1(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.fc = nn.Linear(2048, 100, bias=True)
    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(self.flatten(x))

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
        att = F.softmax(att.masked_fill(mask, float('-inf')), dim=-1)
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
            nn.Linear(embed_dim, ffn_dim), nn.GELU(),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout))
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
            for _ in range(num_layers)])
    def forward(self, x):
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :T, :]
        return self.blocks(h)

class GPT2Stage1(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.blocks = nn.Sequential(*[
            _GPTBlock(_GPT2_EMBED_DIM, _GPT2_NUM_HEADS, _GPT2_FFN_DIM)
            for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(_GPT2_EMBED_DIM)
        self.head = nn.Linear(_GPT2_EMBED_DIM, _GPT2_VOCAB_SIZE, bias=False)
    def forward(self, h):
        return self.head(self.ln_f(self.blocks(h)))

def _make_model_configs(seq_len=512):
    cifar10_root = os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin")
    cifar100_root = os.getenv("CIFAR100_BIN_ROOT", "data/cifar-100-binary")
    tiny_root = os.getenv("TINY_IMAGENET_ROOT", "data/tiny-imagenet-200")
    imgnet100_root = os.getenv("IMAGENET100_ROOT", "data/imagenet-100")
    return {
        "gpt2": {
            "stage0_cls": lambda: GPT2Stage0(_GPT2_NUM_LAYERS // 2),
            "stage1_cls": lambda: GPT2Stage1(_GPT2_NUM_LAYERS // 2),
            "train_set": lambda: _GPT2SyntheticDataset(seq_len),
            "test_set": lambda: _GPT2SyntheticDataset(seq_len),
            "act_tail": (seq_len, _GPT2_EMBED_DIM),
            "batch_size": int(os.getenv("BATCH_SIZE", "16")),
            "epochs": int(os.getenv("EPOCHS", "1")),
            "lr": float(os.getenv("LR_INITIAL", "0.0001")),
            "num_workers": 0,
            "is_gpt2": True,
        },
        "resnet9_cifar10": {
            "stage0_cls": ResNet9Stage0,
            "stage1_cls": ResNet9Stage1,
            "train_set": lambda: CIFAR10Bin(cifar10_root, train=True, transform=_cifar10_transform),
            "test_set": lambda: CIFAR10Bin(cifar10_root, train=False, transform=_cifar10_transform),
            "act_tail": (128,16,16),
            "batch_size": int(os.getenv("BATCH_SIZE", "128")),
            "epochs": int(os.getenv("EPOCHS", "10")),
            "lr": float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers": 2,
            "is_gpt2": False,
        },
        "wrn16_8_cifar100": {
            "stage0_cls": WRN16_8Stage0,
            "stage1_cls": WRN16_8Stage1,
            "train_set": lambda: CIFAR100Bin(cifar100_root, train=True, transform=_cifar100_transform),
            "test_set": lambda: CIFAR100Bin(cifar100_root, train=False, transform=_cifar100_transform),
            "act_tail": (256,16,16),
            "batch_size": int(os.getenv("BATCH_SIZE", "128")),
            "epochs": int(os.getenv("EPOCHS", "50")),
            "lr": float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers": 2,
            "is_gpt2": False,
        },
        "resnet50_tiny_imagenet": {
            "stage0_cls": ResNet50TinyStage0,
            "stage1_cls": ResNet50TinyStage1,
            "train_set": lambda: TinyImageNetDataset(tiny_root, train=True, transform=_tiny_train_transform),
            "test_set": lambda: TinyImageNetDataset(tiny_root, train=False, transform=_tiny_test_transform),
            "act_tail": (512,16,16),
            "batch_size": int(os.getenv("BATCH_SIZE", "128")),
            "epochs": int(os.getenv("EPOCHS", "90")),
            "lr": float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers": 4,
            "is_gpt2": False,
        },
        "resnet50_imagenet100": {
            "stage0_cls": ResNet50ImgNet100Stage0,
            "stage1_cls": ResNet50ImgNet100Stage1,
            "train_set": lambda: ImageNet100Dataset(imgnet100_root, train=True, transform=_imagenet100_train_transform),
            "test_set": lambda: ImageNet100Dataset(imgnet100_root, train=False, transform=_imagenet100_test_transform),
            "act_tail": (512,28,28),
            "batch_size": int(os.getenv("BATCH_SIZE", "64")),
            "epochs": int(os.getenv("EPOCHS", "90")),
            "lr": float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers": 4,
            "is_gpt2": False,
        },
    }

CMD_STOP, CMD_TRAIN, CMD_EVAL = 0, 1, 2
def send_ctrl(device, cmd, batch_size):
    dist.send(torch.tensor([cmd, batch_size], dtype=torch.int64, device=device), dst=1)
def recv_ctrl(device):
    ctrl = torch.empty(2, dtype=torch.int64, device=device); dist.recv(ctrl, src=0); return int(ctrl[0].item()), int(ctrl[1].item())
def send_stats(device, loss_sum, correct, total):
    dist.send(torch.tensor([loss_sum, correct, total], dtype=torch.float64, device=device), dst=0)
def recv_stats(device):
    t = torch.empty(3, dtype=torch.float64, device=device); dist.recv(t, src=1); return float(t[0]), float(t[1]), float(t[2])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet9_cifar10",
                       choices=["gpt2", "resnet9_cifar10", "wrn16_8_cifar100",
                               "resnet50_tiny_imagenet", "resnet50_imagenet100"])
    parser.add_argument("--micro-bs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512, help="GPT-2 sequence length")
    parser.add_argument("--print-freq", type=int, default=int(os.getenv("PRINT_FREQ", "10")))
    parser.add_argument("--log-dir", type=str, default=os.getenv("LOG_DIR", "logs"))
    args = parser.parse_args()

    backend = os.getenv("DIST_BACKEND", "nccl")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2: raise RuntimeError(f"world_size must be 2, got {world_size}")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    cfg = _make_model_configs(args.seq_len)[args.model]
    stage0_cls = cfg["stage0_cls"]
    stage1_cls = cfg["stage1_cls"]
    model = (stage0_cls() if rank == 0 else stage1_cls()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9,0.999), eps=1e-3, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    log_dir = ensure_dir(args.log_dir)
    nic = os.getenv("NET_LOG_IFNAME") or os.getenv("NCCL_SOCKET_IFNAME") or os.getenv("GLOO_SOCKET_IFNAME")
    net_counter = NetCounter(nic)
    net_csv = CsvAppender(log_dir / f"{args.model}_rank{rank}_net.csv", ["timestamp","rank","phase","epoch","step","batch_size","ifname","dt_sec","tx_bytes_delta","rx_bytes_delta","tx_MBps","rx_MBps","counter_ok"])
    metrics_csv = None
    if rank == 0:
        metrics_csv = CsvAppender(log_dir / f"{args.model}_rank0_metrics.csv", ["timestamp","phase","epoch","step","batch_size","lr","loss_sum","samples","avg_loss","correct","accuracy_percent","epoch_time_sec"])

    try:
        if rank == 0:
            train_loader = DataLoader(cfg["train_set"](), batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], drop_last=False, pin_memory=True)
            test_loader = DataLoader(cfg["test_set"](), batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], drop_last=False, pin_memory=True)
            for epoch in range(cfg["epochs"]):
                model.train()
                train_loss_sum = 0.0; train_correct = 0; train_total = 0; epoch_start = time.time()
                for step, (x,y) in enumerate(train_loader):
                    t0 = time.time()
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    a = model(x)
                    send_ctrl(device, CMD_TRAIN, x.size(0))
                    dist.send(a.detach().contiguous(), dst=1)
                    dist.send(y.contiguous(), dst=1)
                    grad_a = torch.empty_like(a); dist.recv(grad_a, src=1)
                    a.backward(grad_a); optimizer.step()
                    loss_sum_b, correct_b, total_b = recv_stats(device)
                    train_loss_sum += loss_sum_b; train_correct += int(correct_b); train_total += int(total_b)
                    avg_loss = train_loss_sum / max(train_total, 1); acc = 100.0 * train_correct / max(train_total, 1)
                    net = net_counter.sample()
                    net_csv.write(dict(timestamp=t0, rank=rank, phase="train_batch", epoch=epoch+1, step=step, batch_size=int(x.size(0)), **net))
                    metrics_csv.write(dict(timestamp=t0, phase="train_batch", epoch=epoch+1, step=step, batch_size=int(x.size(0)), lr=float(optimizer.param_groups[0]["lr"]), loss_sum=float(train_loss_sum), samples=int(train_total), avg_loss=float(avg_loss), correct=int(train_correct), accuracy_percent=float(acc), epoch_time_sec=float(time.time()-epoch_start)))
                    if step % args.print_freq == 0:
                        if cfg.get("is_gpt2", False):
                            print(f"[Rank 0] Epoch {epoch+1}/{cfg['epochs']} Step {step}/{len(train_loader)} | train_loss={avg_loss:.4f} | net_tx={net['tx_MBps']:.2f}MB/s net_rx={net['rx_MBps']:.2f}MB/s", flush=True)
                        else:
                            print(f"[Rank 0] Epoch {epoch+1}/{cfg['epochs']} Step {step}/{len(train_loader)} | train_loss={avg_loss:.4f} | train_acc={acc:.2f}% | net_tx={net['tx_MBps']:.2f}MB/s net_rx={net['rx_MBps']:.2f}MB/s", flush=True)
                scheduler.step()
                epoch_time = time.time() - epoch_start
                model.eval()
                test_loss_sum = 0.0; test_correct = 0; test_total = 0
                with torch.no_grad():
                    for step, (x,y) in enumerate(test_loader):
                        t0 = time.time()
                        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                        a = model(x)
                        send_ctrl(device, CMD_EVAL, x.size(0))
                        dist.send(a.contiguous(), dst=1)
                        dist.send(y.contiguous(), dst=1)
                        loss_sum_b, correct_b, total_b = recv_stats(device)
                        test_loss_sum += loss_sum_b; test_correct += int(correct_b); test_total += int(total_b)
                        net_csv.write(dict(timestamp=t0, rank=rank, phase="test_batch", epoch=epoch+1, step=step, batch_size=int(x.size(0)), **net_counter.sample()))
                train_loss = train_loss_sum / max(train_total, 1); train_acc = 100.0 * train_correct / max(train_total, 1)
                test_loss = test_loss_sum / max(test_total, 1); test_acc = 100.0 * test_correct / max(test_total, 1)
                metrics_csv.write(dict(timestamp=time.time(), phase="epoch_summary", epoch=epoch+1, step=-1, batch_size=cfg["batch_size"], lr=float(optimizer.param_groups[0]["lr"]), loss_sum=float(test_loss_sum), samples=int(test_total), avg_loss=float(test_loss), correct=int(test_correct), accuracy_percent=float(test_acc), epoch_time_sec=float(epoch_time)))
                if cfg.get("is_gpt2", False):
                    print(f"[Rank 0] Epoch {epoch+1}/{cfg['epochs']} DONE | time={epoch_time:.2f}s | train_loss={train_loss:.4f} | test_loss={test_loss:.4f}", flush=True)
                else:
                    print(f"[Rank 0] Epoch {epoch+1}/{cfg['epochs']} DONE | time={epoch_time:.2f}s | train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | test_loss={test_loss:.4f} test_acc={test_acc:.2f}%", flush=True)
            send_ctrl(device, CMD_STOP, 0)
            print("[Rank 0] Training complete.", flush=True)
        else:
            is_gpt2 = cfg.get("is_gpt2", False)
            tr_step = 0; ev_step = 0
            while True:
                cmd, bsz = recv_ctrl(device)
                if cmd == CMD_STOP:
                    print("[Rank 1] Received STOP.", flush=True); break
                t0 = time.time()
                act = torch.empty((bsz, *cfg["act_tail"]), dtype=torch.float32, device=device)
                if is_gpt2:
                    y = torch.empty((bsz, cfg["act_tail"][0]), dtype=torch.int64, device=device)
                else:
                    y = torch.empty((bsz,), dtype=torch.int64, device=device)
                dist.recv(act, src=0); dist.recv(y, src=0)
                if cmd == CMD_TRAIN:
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    act.requires_grad_(True)
                    logits = model(act)
                    if is_gpt2:
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = y[:, 1:].contiguous()
                        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        correct = (shift_logits.argmax(dim=-1) == shift_labels).sum().item()
                        total = shift_labels.numel()
                    else:
                        loss = F.cross_entropy(logits, y)
                        correct = (logits.argmax(dim=1) == y).sum().item()
                        total = bsz
                    loss.backward(); optimizer.step()
                    dist.send(act.grad.contiguous(), dst=0)
                    send_stats(device, loss.item()*total, float(correct), float(total))
                    net_csv.write(dict(timestamp=t0, rank=rank, phase="train_batch", epoch=-1, step=tr_step, batch_size=int(bsz), **net_counter.sample()))
                    tr_step += 1
                elif cmd == CMD_EVAL:
                    model.eval()
                    with torch.no_grad():
                        logits = model(act)
                        if is_gpt2:
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = y[:, 1:].contiguous()
                            loss_sum = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum").item()
                            correct = (shift_logits.argmax(dim=-1) == shift_labels).sum().item()
                            total = shift_labels.numel()
                        else:
                            loss_sum = F.cross_entropy(logits, y, reduction="sum").item()
                            correct = (logits.argmax(dim=1) == y).sum().item()
                            total = bsz
                    send_stats(device, loss_sum, float(correct), float(total))
                    net_csv.write(dict(timestamp=t0, rank=rank, phase="test_batch", epoch=-1, step=ev_step, batch_size=int(bsz), **net_counter.sample()))
                    ev_step += 1
                else:
                    raise RuntimeError(f"Unknown cmd {cmd}")
    finally:
        net_csv.close()
        if metrics_csv is not None: metrics_csv.close()
        if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
