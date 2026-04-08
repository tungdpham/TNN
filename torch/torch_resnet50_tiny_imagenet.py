import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

load_dotenv()


# ======================== Dataset ========================

class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet dataset loader.
    Expects the standard tiny-imagenet-200 directory layout:
      train/<wnid>/images/*.JPEG
      val/images/*.JPEG  +  val/val_annotations.txt
    """
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


# ImageNet-normalised mean/std precomputed on Tiny ImageNet training set
TINY_MEAN = [0.4802, 0.4481, 0.3975]
TINY_STD  = [0.2770, 0.2691, 0.2821]

import torchvision.transforms as T

def train_transform():
    return T.Compose([
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(TINY_MEAN, TINY_STD),
    ])

def test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(TINY_MEAN, TINY_STD),
    ])


# ======================== Model ========================

class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block matching bottleneck_residual_block() in layer_builder.hpp.

    Main path: Conv1x1(no bias) -> BN+ReLU
               -> Conv3x3(stride, no bias) -> BN+ReLU
               -> Conv1x1(no bias) -> BN+ReLU   <- ReLU is fused into last BN
    Shortcut:  Conv1x1(stride, no bias) -> BN (no ReLU)  when shapes differ; else identity
    Post-add:  ReLU  (ResidualBlock constructed with "relu" activation)
    """
    def __init__(self, in_channels: int, mid_channels: int,
                 out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.shortcut(x) if self.shortcut is not None else x

        out = F.relu(self.bn1(self.conv1(x)),   inplace=True)
        out = F.relu(self.bn2(self.conv2(out)),  inplace=True)
        out = F.relu(self.bn3(self.conv3(out)),  inplace=True)  # BN+ReLU before add

        out = out + sc
        out = F.relu(out, inplace=True)  # post-addition ReLU ("relu" activation)
        return out


class ResNet50TinyImageNet(nn.Module):
    """
    ResNet-50 for Tiny ImageNet (64x64 inputs, 200 classes).

    Matches create_tiny_imagenet_resnet50() in example_models.cpp:
      conv1   : 3  -> 64, 3x3, stride 1, pad 1, bias=True
      bn1     : BN + ReLU
      maxpool : 3x3, stride 2, pad 1  (64x64 -> 32x32)
      layer1  : 3 x bottleneck(64,  64,  256, stride=1)
      layer2  : 4 x bottleneck(256, 128, 512, first stride=2)  (32x32 -> 16x16)
      layer3  : 6 x bottleneck(512, 256, 1024, first stride=2) (16x16 ->  8x8)
      layer4  : 3 x bottleneck(1024, 512, 2048, first stride=2) (8x8 ->   4x4)
      avgpool : 4x4, stride 1 -> 1x1
      flatten
      fc      : 2048 -> 200
    """
    def __init__(self, num_classes: int = 200):
        super().__init__()

        self.conv1   = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.bn1     = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)  # 64x64 -> 32x32

        # Layer 1: 3 blocks, all stride=1
        self.layer1 = nn.Sequential(
            BottleneckBlock( 64,  64, 256, stride=1),  # layer1_block1
            BottleneckBlock(256,  64, 256, stride=1),  # layer1_block2
            BottleneckBlock(256,  64, 256, stride=1),  # layer1_block3
        )

        # Layer 2: 4 blocks, first stride=2  (32x32 -> 16x16)
        self.layer2 = nn.Sequential(
            BottleneckBlock(256, 128, 512, stride=2),  # layer2_block1
            BottleneckBlock(512, 128, 512, stride=1),  # layer2_block2
            BottleneckBlock(512, 128, 512, stride=1),  # layer2_block3
            BottleneckBlock(512, 128, 512, stride=1),  # layer2_block4
        )

        # Layer 3: 6 blocks, first stride=2  (16x16 -> 8x8)
        self.layer3 = nn.Sequential(
            BottleneckBlock( 512, 256, 1024, stride=2),  # layer3_block1
            BottleneckBlock(1024, 256, 1024, stride=1),  # layer3_block2
            BottleneckBlock(1024, 256, 1024, stride=1),  # layer3_block3
            BottleneckBlock(1024, 256, 1024, stride=1),  # layer3_block4
            BottleneckBlock(1024, 256, 1024, stride=1),  # layer3_block5
            BottleneckBlock(1024, 256, 1024, stride=1),  # layer3_block6
        )

        # Layer 4: 3 blocks, first stride=2  (8x8 -> 4x4)
        self.layer4 = nn.Sequential(
            BottleneckBlock(1024, 512, 2048, stride=2),  # layer4_block1
            BottleneckBlock(2048, 512, 2048, stride=1),  # layer4_block2
            BottleneckBlock(2048, 512, 2048, stride=1),  # layer4_block3
        )

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)  # 4x4 -> 1x1
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # BN+ReLU
        x = self.maxpool(x)                                 # 32x32

        x = self.layer1(x)                                  # 32x32 x256
        x = self.layer2(x)                                  # 16x16 x512
        x = self.layer3(x)                                  #  8x8  x1024
        x = self.layer4(x)                                  #  4x4  x2048

        x = self.avgpool(x)                                 #  1x1  x2048
        x = self.flatten(x)                                 #  2048
        x = self.fc(x)                                      #  200
        return x


# ======================== Training ========================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(">>> Running on device:", device)

    epochs     = int(os.getenv("EPOCHS",     "90"))
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    lr_initial = float(os.getenv("LR_INITIAL", "0.001"))
    data_root  = os.getenv("TINY_IMAGENET_ROOT", "data/tiny-imagenet-200")

    print(f">>> Data root  : {data_root}")
    print(f">>> Epochs     : {epochs}")
    print(f">>> Batch size : {batch_size}")
    print(f">>> LR initial : {lr_initial}")

    train_set = TinyImageNetDataset(root=data_root, train=True,
                                    transform=train_transform())
    val_set   = TinyImageNetDataset(root=data_root, train=False,
                                    transform=test_transform())

    print(f">>> Train samples: {len(train_set)}")
    print(f">>> Val   samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet50TinyImageNet(num_classes=200).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Parameters : {total_params:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr_initial,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=5e-4,
        amsgrad=True,
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        epoch_start    = time.time()
        last_100_start = time.time()

        # ---- Training ----
        model.train()
        running_loss    = 0.0
        running_correct = 0
        running_total   = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss    += loss.item() * inputs.size(0)
            _, predicted     = outputs.max(1)
            running_total   += targets.size(0)
            running_correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 100 == 0:
                batch_acc   = 100.0 * predicted.eq(targets).sum().item() / targets.size(0)
                elapsed_100 = time.time() - last_100_start
                last_100_start = time.time()
                print(
                    f"[Train Batch {batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}% | "
                    f"100-batch time: {elapsed_100:.2f}s"
                )

        train_loss = running_loss / running_total
        train_acc  = 100.0 * running_correct / running_total

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        val_total    = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, targets)

                val_loss_sum += loss.item() * inputs.size(0)
                _, predicted  = outputs.max(1)
                val_total    += targets.size(0)
                val_correct  += predicted.eq(targets).sum().item()

        val_loss = val_loss_sum / val_total
        val_acc  = 100.0 * val_correct / val_total

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%"
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            "model_snapshots/resnet50_tiny_imagenet.pth",
        )

    print("\n>>> Tiny ImageNet ResNet-50 training completed.")


if __name__ == "__main__":
    main()
