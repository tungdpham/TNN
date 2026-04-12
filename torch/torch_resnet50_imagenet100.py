import csv
import datetime
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

class ImageNet100Dataset(Dataset):
    """
    ImageNet-100 dataset loader.
    Expects directory layout:
      train.X1/<class_id>/*.JPEG
      train.X2/<class_id>/*.JPEG
      train.X3/<class_id>/*.JPEG
      train.X4/<class_id>/*.JPEG
      val.X/<class_id>/*.JPEG
      Labels.json
    """
    def __init__(self, root: str, train: bool = True, transform=None):
        self.transform = transform
        root = Path(root)

        self.samples = []
        self.class_to_idx = {}

        # Build class -> idx mapping from all train directories (canonical ordering)
        all_classes = set()
        for train_subdir in ["train.X1", "train.X2", "train.X3", "train.X4"]:
            train_dir = root / train_subdir
            if train_dir.exists():
                all_classes.update(d.name for d in train_dir.iterdir() if d.is_dir())
        
        classes = sorted(all_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        if train:
            # Load from all train.X1, train.X2, train.X3, train.X4
            for train_subdir in ["train.X1", "train.X2", "train.X3", "train.X4"]:
                data_dir = root / train_subdir
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
            # Load from val.X
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


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

import torchvision.transforms as T

def train_transform():
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def test_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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


class ResNet50ImageNet100(nn.Module):
    """
    ResNet-50 for ImageNet-100 (224x224 inputs, 100 classes).

    Architecture:
      conv1   : 3  -> 64, 7x7, stride 2, pad 3, bias=True
      bn1     : BN + ReLU
      maxpool : 3x3, stride 2, pad 1  (224x224 -> 112x112 -> 56x56)
      layer1  : 3 x bottleneck(64,  64,  256, stride=1)
      layer2  : 4 x bottleneck(256, 128, 512, first stride=2)  (56x56 -> 28x28)
      layer3  : 6 x bottleneck(512, 256, 1024, first stride=2) (28x28 -> 14x14)
      layer4  : 3 x bottleneck(1024, 512, 2048, first stride=2) (14x14 -> 7x7)
      avgpool : 7x7, stride 1 -> 1x1
      flatten
      fc      : 2048 -> 100
    """
    def __init__(self, num_classes: int = 100):
        super().__init__()

        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True)
        self.bn1     = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)  # 224x224 -> 112x112 -> 56x56

        # Layer 1: 3 blocks, all stride=1
        self.layer1 = nn.Sequential(
            BottleneckBlock( 64,  64, 256, stride=1),  # layer1_block1
            BottleneckBlock(256,  64, 256, stride=1),  # layer1_block2
            BottleneckBlock(256,  64, 256, stride=1),  # layer1_block3
        )

        # Layer 2: 4 blocks, first stride=2  (56x56 -> 28x28)
        self.layer2 = nn.Sequential(
            BottleneckBlock(256, 128, 512, stride=2),  # layer2_block1
            BottleneckBlock(512, 128, 512, stride=1),  # layer2_block2
            BottleneckBlock(512, 128, 512, stride=1),  # layer2_block3
            BottleneckBlock(512, 128, 512, stride=1),  # layer2_block4
        )

        # Layer 3: 6 blocks, first stride=2  (28x28 -> 14x14)
        self.layer3 = nn.Sequential(
            BottleneckBlock( 512, 256, 1024, stride=2),  # layer3_block1
            BottleneckBlock(1024, 256, 1024, stride=1),  # layer3_block2
            BottleneckBlock(1024, 256, 1024, stride=1),  # layer3_block3
            BottleneckBlock(1024, 256, 1024, stride=1),  # layer3_block4
            BottleneckBlock(1024, 256, 1024, stride=1),  # layer3_block5
            BottleneckBlock(1024, 256, 1024, stride=1),  # layer3_block6
        )

        # Layer 4: 3 blocks, first stride=2  (14x14 -> 7x7)
        self.layer4 = nn.Sequential(
            BottleneckBlock(1024, 512, 2048, stride=2),  # layer4_block1
            BottleneckBlock(2048, 512, 2048, stride=1),  # layer4_block2
            BottleneckBlock(2048, 512, 2048, stride=1),  # layer4_block3
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)  # 7x7 -> 1x1
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # BN+ReLU
        x = self.maxpool(x)                                 # 56x56

        x = self.layer1(x)                                  # 56x56 x256
        x = self.layer2(x)                                  # 28x28 x512
        x = self.layer3(x)                                  # 14x14 x1024
        x = self.layer4(x)                                  #  7x7  x2048

        x = self.avgpool(x)                                 #  1x1  x2048
        x = self.flatten(x)                                 #  2048
        x = self.fc(x)                                      #  100
        return x


# ======================== Training ========================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(">>> Running on device:", device)

    epochs     = int(os.getenv("EPOCHS",     "90"))
    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    lr_initial = float(os.getenv("LR_INITIAL", "0.001"))
    data_root  = os.getenv("IMAGENET100_ROOT", "data/imagenet-100")

    print(f">>> Data root  : {data_root}")
    print(f">>> Epochs     : {epochs}")
    print(f">>> Batch size : {batch_size}")
    print(f">>> LR initial : {lr_initial}")

    train_set = ImageNet100Dataset(root=data_root, train=True,
                                   transform=train_transform())
    val_set   = ImageNet100Dataset(root=data_root, train=False,
                                   transform=test_transform())

    print(f">>> Train samples: {len(train_set)}")
    print(f">>> Val   samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet50ImageNet100(num_classes=100).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f">>> Parameters : {total_params:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr_initial,
        betas=(0.9, 0.999),
        eps=1e-3,
        weight_decay=3e-4,
        amsgrad=False,
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    batch_csv_path = os.path.join(log_dir, f"torch_imagenet100_resnet50_batch_{ts}.csv")
    epoch_csv_path = os.path.join(log_dir, f"torch_imagenet100_resnet50_epoch_{ts}.csv")
    val_csv_path   = os.path.join(log_dir, f"torch_imagenet100_resnet50_val_{ts}.csv")

    batch_csv_file = open(batch_csv_path, "w", newline="")
    epoch_csv_file = open(epoch_csv_path, "w", newline="")
    val_csv_file   = open(val_csv_path,   "w", newline="")

    batch_writer = csv.writer(batch_csv_file)
    epoch_writer = csv.writer(epoch_csv_file)
    val_writer   = csv.writer(val_csv_file)

    batch_writer.writerow(["epoch", "step", "loss", "accuracy_pct", "time_ms"])
    epoch_writer.writerow(["epoch", "train_loss", "train_accuracy_pct", "val_loss", "val_accuracy_pct"])
    val_writer.writerow(["epoch", "step", "loss", "accuracy_pct"])

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        epoch_start    = time.time()

        # ---- Training ----
        model.train()
        running_loss    = 0.0
        running_correct = 0
        running_total   = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            step_start = time.time()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            step_ms = int((time.time() - step_start) * 1000)

            running_loss    += loss.item() * inputs.size(0)
            _, predicted     = outputs.max(1)
            running_total   += targets.size(0)
            running_correct += predicted.eq(targets).sum().item()

            batch_loss = loss.item()
            batch_acc  = 100.0 * predicted.eq(targets).sum().item() / inputs.size(0)
            batch_writer.writerow([epoch, batch_idx + 1, f"{batch_loss:.6f}", f"{batch_acc:.4f}", step_ms])
            batch_csv_file.flush()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"[Train Batch {batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}% | "
                    f"Step time: {step_ms}ms"
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
            for val_step, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, targets)

                val_loss_sum += loss.item() * inputs.size(0)
                _, predicted  = outputs.max(1)
                val_total    += targets.size(0)
                val_correct  += predicted.eq(targets).sum().item()

                step_loss = loss.item()
                step_acc  = 100.0 * predicted.eq(targets).sum().item() / inputs.size(0)
                val_writer.writerow([epoch, val_step + 1, f"{step_loss:.6f}", f"{step_acc:.4f}"])
            val_csv_file.flush()

        val_loss = val_loss_sum / val_total
        val_acc  = 100.0 * val_correct / val_total

        epoch_time = time.time() - epoch_start

        epoch_writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{val_loss:.6f}", f"{val_acc:.4f}"])
        epoch_csv_file.flush()

        print(
            f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%"
        )

        # Save checkpoint
        os.makedirs("model_snapshots", exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            "model_snapshots/resnet50_imagenet100.pth",
        )

    batch_csv_file.close()
    epoch_csv_file.close()
    val_csv_file.close()
    print(f"\n>>> Logs saved to {log_dir}/torch_imagenet100_resnet50_*_{ts}.csv")
    print("\n>>> ImageNet-100 ResNet-50 training completed.")


if __name__ == "__main__":
    main()
