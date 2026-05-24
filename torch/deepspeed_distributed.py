#!/usr/bin/env python3
import argparse, csv, math, os, time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

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
        try: self.f.flush(); self.f.close()
        except Exception: pass

class NetCounter:
    def __init__(self, ifname: Optional[str]):
        self.ifname = ifname
        self.available = False
        self.prev_tx, self.prev_rx = 0, 0
        self.prev_t = time.time()
        if ifname:
            tx, rx = self._read_bytes()
            if tx is not None:
                self.available = True
                self.prev_tx, self.prev_rx = tx, rx
    def _read_bytes(self) -> Tuple[Optional[int], Optional[int]]:
        try:
            base = Path("/sys/class/net") / self.ifname / "statistics"
            return int((base/"tx_bytes").read_text().strip()), int((base/"rx_bytes").read_text().strip())
        except Exception: return None, None
    def sample(self):
        t = time.time()
        if not self.available:
            return dict(ifname=self.ifname or "", dt_sec=0.0, tx_MBps=0.0, rx_MBps=0.0, counter_ok=0)
        tx, rx = self._read_bytes()
        if tx is None: return dict(ifname=self.ifname or "", dt_sec=0.0, tx_MBps=0.0, rx_MBps=0.0, counter_ok=0)
        dt = max(t - self.prev_t, 1e-9)
        dtx, drx = tx - self.prev_tx, rx - self.prev_rx
        self.prev_tx, self.prev_rx, self.prev_t = tx, rx, t
        return dict(ifname=self.ifname or "", dt_sec=dt, tx_MBps=dtx/dt/(1024*1024), rx_MBps=drx/dt/(1024*1024), counter_ok=1)

def classification_loss_fn(logits, labels):
    return F.cross_entropy(logits, labels)

def gpt2_loss_fn(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

_CIFAR10_MEAN = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3,1,1)
_CIFAR10_STD  = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3,1,1)
def _cifar10_transform(img): return (img - _CIFAR10_MEAN) / _CIFAR10_STD

_CIFAR100_MEAN = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(3,1,1)
_CIFAR100_STD  = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(3,1,1)
def _cifar100_transform(img): return (img - _CIFAR100_MEAN) / _CIFAR100_STD

_TINY_MEAN = [0.4802, 0.4481, 0.3975]
_TINY_STD  = [0.2770, 0.2691, 0.2821]
_tiny_train_transform = T.Compose([T.ToTensor(), T.Normalize(_TINY_MEAN, _TINY_STD)])

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_imagenet100_train_transform = T.Compose([
    T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])

class CIFAR10Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.data, self.targets = [], []
        batch_files = [f"data_batch_{i}.bin" for i in range(1, 6)] if train else ["test_batch.bin"]
        for fname in batch_files:
            path = os.path.join(root, fname)
            if not os.path.exists(path): continue
            with open(path, "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3073)
                self.data.append(arr[:,1:].reshape(-1,3,32,32))
                self.targets.append(arr[:,0])
        if self.data:
            self.data = np.concatenate(self.data, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].astype(np.float32)/255.0)
        if self.transform: img = self.transform(img)
        return img, int(self.targets[idx])

class CIFAR100Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        fname = "train.bin" if train else "test.bin"
        path = os.path.join(root, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3074)
            self.targets = arr[:, 1].astype(np.int64)
            self.data = arr[:, 2:].reshape(-1, 3, 32, 32)
        else:
            self.targets, self.data = [], []
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx].astype(np.float32)/255.0)
        if self.transform: img = self.transform(img)
        return img, int(self.targets[idx])

class TinyImageNetDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        root = Path(root)
        self.samples = []
        train_dir = root / "train"
        if not train_dir.exists(): return
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
            if (val_dir / "val_annotations.txt").exists():
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

class OpenWebTextDataset(Dataset):
    """
    Streams tokenised OpenWebText from a flat uint16 binary file produced by
    python/openwebtext.py.

    Every item is a (input, target) pair of length SEQ_LEN where
    target = input shifted one position to the right (standard CLM).
    
    Uses memory-mapped file access to avoid loading the entire dataset into RAM.
    """
    def __init__(self, path: str, seq_len: int):
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

class _BasicResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)

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

# ResNet50 Sequence Wrappers
class _ResNet50TinyLayer1(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(_BottleneckBlock(64, 64, 256, stride=1), _BottleneckBlock(256, 64, 256, stride=1), _BottleneckBlock(256, 64, 256, stride=1))
    def forward(self, x): return self.seq(x)

class _ResNet50TinyLayer2(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(_BottleneckBlock(256, 128, 512, stride=2), _BottleneckBlock(512, 128, 512, stride=1), _BottleneckBlock(512, 128, 512, stride=1), _BottleneckBlock(512, 128, 512, stride=1))
    def forward(self, x): return self.seq(x)

class _ResNet50TinyLayer3(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(_BottleneckBlock(512, 256, 1024, stride=2), *[_BottleneckBlock(1024, 256, 1024, stride=1) for _ in range(5)])
    def forward(self, x): return self.seq(x)

class _ResNet50TinyLayer4(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(_BottleneckBlock(1024, 512, 2048, stride=2), _BottleneckBlock(2048, 512, 2048, stride=1), _BottleneckBlock(2048, 512, 2048, stride=1))
    def forward(self, x): return self.seq(x)

class _GPT2Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(50257, 768)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 768))
    def forward(self, x):
        return self.token_emb(x) + self.pos_emb[:, :x.size(1), :]

class _GPTBlock(nn.Module):
    def __init__(self, d=768, h=12):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x):
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.ln2(x))

def _make_model_configs(seq_len):
    cifar_root = os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin")
    cifar100_root = os.getenv("CIFAR100_BIN_ROOT", "data/cifar-100-binary")
    tiny_root = os.getenv("TINY_IMAGENET_ROOT", "data/tiny-imagenet-200")
    imgnet100_root = os.getenv("IMAGENET100_ROOT", "data/imagenet-100")
    openwebtext_path = os.getenv("OPENWEBTEXT_PATH", "data/open-web-text/train.bin")

    return {
        "gpt2": {
            "layers": [LayerSpec(_GPT2Embedding), *[LayerSpec(_GPTBlock) for _ in range(12)], 
                       LayerSpec(nn.LayerNorm, 768), LayerSpec(nn.Linear, 768, 50257, bias=False)],
            "loss_fn": gpt2_loss_fn,
            "train_set": lambda: OpenWebTextDataset(openwebtext_path, seq_len),
            "lr": 1e-4, "batch_size": 16, "num_workers": 0
        },
        "resnet9_cifar10": {
            "layers": [LayerSpec(nn.Conv2d, 3, 64, 3, padding=1), LayerSpec(nn.BatchNorm2d, 64), LayerSpec(nn.ReLU, True),
                       LayerSpec(nn.Conv2d, 64, 128, 3, padding=1), LayerSpec(nn.MaxPool2d, 2), LayerSpec(_BasicResidualBlock, 128),
                       LayerSpec(nn.Conv2d, 128, 256, 3, padding=1), LayerSpec(nn.MaxPool2d, 2), LayerSpec(_BasicResidualBlock, 256),
                       LayerSpec(nn.AdaptiveAvgPool2d, (1, 1)), LayerSpec(nn.Flatten), LayerSpec(nn.Linear, 256, 10)],
            "loss_fn": classification_loss_fn,
            "train_set": lambda: CIFAR10Bin(cifar_root, train=True, transform=_cifar10_transform),
            "lr": 1e-3, "batch_size": 128, "num_workers": 2
        },
        "wrn16_8_cifar100": {
            "layers": [LayerSpec(nn.Conv2d, 3, 16, 3, stride=1, padding=1, bias=True),
                       LayerSpec(_WideResidualBlock, 16, 128, stride=1, dropout_rate=0.3),
                       LayerSpec(_WideResidualBlock, 128, 128, stride=1, dropout_rate=0.3),
                       LayerSpec(_WideResidualBlock, 128, 256, stride=2, dropout_rate=0.3),
                       LayerSpec(_WideResidualBlock, 256, 256, stride=1, dropout_rate=0.3),
                       LayerSpec(_WideResidualBlock, 256, 512, stride=2, dropout_rate=0.3),
                       LayerSpec(_WideResidualBlock, 512, 512, stride=1, dropout_rate=0.3),
                       LayerSpec(nn.BatchNorm2d, 512, eps=1e-5, momentum=0.1), LayerSpec(nn.ReLU, inplace=True),
                       LayerSpec(nn.AvgPool2d, kernel_size=8, stride=1), LayerSpec(nn.Flatten),
                       LayerSpec(nn.Linear, 512, 100, bias=True)],
            "loss_fn": classification_loss_fn,
            "train_set": lambda: CIFAR100Bin(cifar100_root, train=True, transform=_cifar100_transform),
            "lr": 1e-3, "batch_size": 128, "num_workers": 2
        },
        "resnet50_tiny_imagenet": {
            "layers": [LayerSpec(nn.Conv2d, 3, 64, 3, stride=1, padding=1, bias=True),
                       LayerSpec(nn.BatchNorm2d, 64, eps=1e-5, momentum=0.1), LayerSpec(nn.ReLU, inplace=True),
                       LayerSpec(nn.MaxPool2d, 3, stride=2, padding=1),
                       LayerSpec(_ResNet50TinyLayer1), LayerSpec(_ResNet50TinyLayer2),
                       LayerSpec(_ResNet50TinyLayer3), LayerSpec(_ResNet50TinyLayer4),
                       LayerSpec(nn.AvgPool2d, kernel_size=4, stride=1), LayerSpec(nn.Flatten),
                       LayerSpec(nn.Linear, 2048, 200, bias=True)],
            "loss_fn": classification_loss_fn,
            "train_set": lambda: TinyImageNetDataset(tiny_root, train=True, transform=_tiny_train_transform),
            "lr": 1e-3, "batch_size": 128, "num_workers": 4
        },
        "resnet50_imagenet100": {
            "layers": [LayerSpec(nn.Conv2d, 3, 64, 7, stride=2, padding=3, bias=True),
                       LayerSpec(nn.BatchNorm2d, 64, eps=1e-5, momentum=0.1), LayerSpec(nn.ReLU, inplace=True),
                       LayerSpec(nn.MaxPool2d, 3, stride=2, padding=1),
                       LayerSpec(_ResNet50TinyLayer1), LayerSpec(_ResNet50TinyLayer2),
                       LayerSpec(_ResNet50TinyLayer3), LayerSpec(_ResNet50TinyLayer4),
                       LayerSpec(nn.AvgPool2d, kernel_size=7, stride=1), LayerSpec(nn.Flatten),
                       LayerSpec(nn.Linear, 2048, 100, bias=True)],
            "loss_fn": classification_loss_fn,
            "train_set": lambda: ImageNet100Dataset(imgnet100_root, train=True, transform=_imagenet100_train_transform),
            "lr": 1e-3, "batch_size": 64, "num_workers": 4
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet9_cifar10",
                        choices=["gpt2", "resnet9_cifar10", "wrn16_8_cifar100", 
                                 "resnet50_tiny_imagenet", "resnet50_imagenet100"])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum number of training steps (-1 for unlimited)")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)

    cfg = _make_model_configs(args.seq_len)[args.model]
    
    net = PipelineModule(layers=cfg["layers"], num_stages=world_size, loss_fn=cfg["loss_fn"])

    ds_config = {
        "train_batch_size": cfg["batch_size"],
        "train_micro_batch_size_per_gpu": cfg["batch_size"] // world_size,
        "optimizer": {"type": "Adam", "params": {"lr": cfg["lr"]}},
        "fp16": {"enabled": False}
    }

    engine, _, _, _ = deepspeed.initialize(model=net, config=ds_config)
    
    train_loader = DataLoader(cfg["train_set"](), batch_size=engine.train_micro_batch_size_per_gpu(), 
                              shuffle=False, num_workers=cfg["num_workers"], drop_last=True)

    log_dir = ensure_dir(args.log_dir)
    nic = os.getenv("NCCL_SOCKET_IFNAME") or "eth0"
    net_counter = NetCounter(nic)
    net_csv = CsvAppender(log_dir / f"deepspeed_{args.model}_rank{rank}_net.csv", ["timestamp","rank","epoch","step","tx_MBps","rx_MBps"])
    epoch_csv = CsvAppender(log_dir / f"deepspeed_{args.model}_rank{rank}_epoch.csv", ["timestamp","rank","epoch","epoch_time_sec"])

    accum_steps = engine.gradient_accumulation_steps()
    steps_per_epoch = len(train_loader) // accum_steps

    global_step = 0
    for epoch in range(10):
        epoch_start = time.time()
        engine.train()
        train_iter = iter(train_loader)
        
        for step in range(steps_per_epoch):
            if 0 <= args.max_steps <= global_step:
                break
                
            t0 = time.time()
            loss = engine.train_batch(data_iter=train_iter)

            stats = net_counter.sample()
            net_csv.write(dict(timestamp=t0, rank=rank, epoch=epoch+1, step=global_step, **stats))

            if engine.is_last_stage() and step % args.print_freq == 0:
                print(f"[Rank {rank}] Epoch {epoch+1} Step {global_step}/{steps_per_epoch} | Loss: {loss.item():.4f} | Net: {stats['tx_MBps']:.1f}MB/s", flush=True)
            
            global_step += 1

        epoch_time = time.time() - epoch_start
        epoch_csv.write(dict(timestamp=epoch_start, rank=rank, epoch=epoch+1, epoch_time_sec=epoch_time))
        if rank == 0:
            print(f"[Rank {rank}] Epoch {epoch+1} completed in {epoch_time:.2f}s", flush=True)
        
        if 0 <= args.max_steps <= global_step:
            break

    net_csv.close()
    epoch_csv.close()
    if rank == 0: print("Training Complete.")

if __name__ == "__main__":
    main()