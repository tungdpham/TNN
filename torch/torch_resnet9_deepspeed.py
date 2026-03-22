"""
ResNet-9 CIFAR-10 training with DeepSpeed distributed over 2 GPUs (TCP/IP).

Usage:
  Single-node, 2 GPUs:
    deepspeed --num_gpus=2 torch_resnet9_deepspeed.py --deepspeed --deepspeed_config ds_config.json

  Two nodes (1 GPU each), TCP/IP:
    # On node 0 (master):
    deepspeed --hostfile hostfile.txt torch_resnet9_deepspeed.py \
        --deepspeed --deepspeed_config ds_config.json

    # hostfile.txt (one line per node: <hostname/IP> slots=<num_gpus>):
    #   192.168.1.10 slots=1
    #   192.168.1.11 slots=1

  Environment variables (override via .env or shell export):
    EPOCHS, BATCH_SIZE, LR_INITIAL, CIFAR10_BIN_ROOT
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import deepspeed
from deepspeed import comm as dist
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CIFAR10Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []

        batch_files = (
            [f"data_batch_{i}.bin" for i in range(1, 6)] if train
            else ["test_batch.bin"]
        )

        for fname in batch_files:
            path = os.path.join(root, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
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
        label = int(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

CIFAR10_MEAN = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3, 1, 1)
CIFAR10_STD  = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3, 1, 1)


def normalize(img: torch.Tensor) -> torch.Tensor:
    return (img - CIFAR10_MEAN) / CIFAR10_STD


def random_horizontal_flip(img: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if random.random() < p:
        img = torch.flip(img, dims=[2])
    return img


def random_crop_with_padding(img: torch.Tensor, padding: int = 4) -> torch.Tensor:
    c, h, w = img.shape
    padded = torch.zeros((c, h + 2 * padding, w + 2 * padding), dtype=img.dtype)
    padded[:, padding:padding + h, padding:padding + w] = img
    max_offset = 2 * padding
    x = random.randint(0, max_offset)
    y = random.randint(0, max_offset)
    return padded[:, y:y + h, x:x + w]


def train_transform(img: torch.Tensor) -> torch.Tensor:
    img = random_crop_with_padding(img, padding=4)
    img = random_horizontal_flip(img, p=0.5)
    return normalize(img)


def test_transform(img: torch.Tensor) -> torch.Tensor:
    return normalize(img)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BasicResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.bn1   = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.bn2   = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class ResNet9CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1    = nn.Conv2d(3,   64,  3, 1, 1, bias=True)
        self.bn1      = nn.BatchNorm2d(64,  eps=1e-5, momentum=0.1)
        self.conv2    = nn.Conv2d(64,  128, 3, 1, 1, bias=True)
        self.bn2      = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.maxpool  = nn.MaxPool2d(2, 2)
        self.res1     = BasicResidualBlock(128)
        self.res2     = BasicResidualBlock(128)
        self.conv3    = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.bn3      = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.res3     = BasicResidualBlock(256)
        self.res4     = BasicResidualBlock(256)
        self.conv4    = nn.Conv2d(256, 512, 3, 1, 1, bias=True)
        self.bn4      = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.res5     = BasicResidualBlock(512)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten  = nn.Flatten()
        self.fc       = nn.Linear(512, num_classes, bias=True)

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
        return self.fc(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- DeepSpeed adds its own argument parser; parse known args only ----
    import argparse
    parser = argparse.ArgumentParser(description="ResNet-9 CIFAR-10 with DeepSpeed")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # ---- Init distributed (DeepSpeed handles NCCL/Gloo over TCP) ----------
    deepspeed.init_distributed(dist_backend="nccl")      # use "gloo" for CPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Only rank-0 prints progress
    is_master = (global_rank == 0)

    if is_master:
        print(f">>> World size: {world_size}  |  backend: nccl over TCP/IP")
        print(f">>> Running on device: {device}")

    # ---- Hyper-parameters --------------------------------------------------
    epochs      = int(os.getenv("EPOCHS",     "10"))
    batch_size  = int(os.getenv("BATCH_SIZE", "128"))   # per-GPU micro-batch
    data_root   = os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin")

    if is_master:
        print(f">>> Data: {data_root} | Epochs: {epochs} | Per-GPU batch: {batch_size}")

    # ---- Datasets & distributed samplers ----------------------------------
    train_set = CIFAR10Bin(root=data_root, train=True,  transform=train_transform)
    test_set  = CIFAR10Bin(root=data_root, train=False, transform=test_transform)

    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=global_rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_set,  num_replicas=world_size, rank=global_rank, shuffle=False
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, sampler=test_sampler,
        num_workers=2, pin_memory=True
    )

    # ---- Model + DeepSpeed engine -----------------------------------------
    # DeepSpeed wraps the model, handles DDP, gradient sync, optimizer, scaler
    model = ResNet9CIFAR10(num_classes=10)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    criterion = nn.CrossEntropyLoss()

    # ---- Training loop -----------------------------------------------------
    for epoch in range(1, epochs + 1):
        if is_master:
            print(f"\n===== Epoch {epoch}/{epochs} =====")
        epoch_start = time.time()

        # -- Train -----------------------------------------------------------
        model_engine.train()
        train_sampler.set_epoch(epoch)   # ensures different shuffle each epoch

        running_loss    = 0.0
        running_correct = 0
        running_total   = 0
        last_100_start  = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs  = inputs.to(device)
            targets = targets.to(device)

            outputs = model_engine(inputs)
            loss    = criterion(outputs, targets)

            model_engine.backward(loss)   # DeepSpeed backward
            model_engine.step()           # DeepSpeed optimizer step

            running_loss    += loss.item() * inputs.size(0)
            _, predicted     = outputs.max(1)
            running_total   += targets.size(0)
            running_correct += predicted.eq(targets).sum().item()

            if is_master and (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - last_100_start
                last_100_start = time.time()
                print(
                    f"[Train {batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} | "
                    f"100-batch time: {elapsed:.2f}s"
                )

        # Aggregate metrics across all ranks
        stats = torch.tensor(
            [running_loss, running_correct, running_total], dtype=torch.float64, device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        train_loss = stats[0].item() / stats[2].item()
        train_acc  = 100.0 * stats[1].item() / stats[2].item()

        # -- Validation ------------------------------------------------------
        model_engine.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        val_total    = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_engine(inputs)
                loss    = criterion(outputs, targets)

                val_loss_sum += loss.item() * inputs.size(0)
                _, predicted  = outputs.max(1)
                val_total    += targets.size(0)
                val_correct  += predicted.eq(targets).sum().item()

        val_stats = torch.tensor(
            [val_loss_sum, val_correct, val_total], dtype=torch.float64, device=device
        )
        dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
        val_loss = val_stats[0].item() / val_stats[2].item()
        val_acc  = 100.0 * val_stats[1].item() / val_stats[2].item()

        if is_master:
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch}/{epochs} done in {epoch_time:.2f}s\n"
                f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
                f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%"
            )

    if is_master:
        print("\n>>> Training complete.")


if __name__ == "__main__":
    main()
