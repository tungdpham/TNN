import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
load_dotenv()


class CIFAR100Bin(Dataset):
    """
    CIFAR-100 binary format loader.
    Each record: 1 coarse label + 1 fine label + 3072 pixel bytes = 3074 bytes.
    """
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


CIFAR100_MEAN = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(3, 1, 1)
CIFAR100_STD  = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(3, 1, 1)


def normalize(img: torch.Tensor) -> torch.Tensor:
    return (img - CIFAR100_MEAN) / CIFAR100_STD


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
    img = normalize(img)
    return img


def test_transform(img: torch.Tensor) -> torch.Tensor:
    img = normalize(img)
    return img


class WideResidualBlock(nn.Module):
    """
    Pre-activation wide residual block matching the C++ wide_residual_block:
      Main path:  BN+ReLU -> Conv(3x3, stride) -> BN+ReLU -> [Dropout] -> Conv(3x3)
      Shortcut:   Conv(1x1, stride, no bias) when stride != 1 or channels differ
      Post-add:   linear (no activation), matching "linear" activation in C++ ResidualBlock
    """
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout_rate: float = 0.0):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut is taken from raw input (before any pre-activation)
        sc = self.shortcut(x) if self.shortcut is not None else x

        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)

        return out + sc  # no post-addition activation ("linear")


class WRN16_8CIFAR100(nn.Module):
    """
    WRN-16-8 for CIFAR-100, matching create_cifar100_wrn16_8 in example_models.cpp:

      conv1  : 3  -> 16,  3x3, stride 1, pad 1, bias
      group1 : 16 -> 128, stride 1, dropout 0.3  (2 wide_residual_blocks)
      group2 : 128-> 256, stride 2, dropout 0.3  (2 wide_residual_blocks)
      group3 : 256-> 512, stride 2, dropout 0.3  (2 wide_residual_blocks)
      bn_final + ReLU
      avgpool 8x8 -> 1x1
      flatten
      fc 512 -> 100
    """
    def __init__(self, num_classes: int = 100):
        super().__init__()
        width_factor  = 8
        dropout_rate  = 0.3
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
        self.fc      = nn.Linear(c3, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                                   # 32x32 x16

        x = self.group1_block1(x)                           # 32x32 x128
        x = self.group1_block2(x)                           # 32x32 x128

        x = self.group2_block1(x)                           # 16x16 x256
        x = self.group2_block2(x)                           # 16x16 x256

        x = self.group3_block1(x)                           # 8x8   x512
        x = self.group3_block2(x)                           # 8x8   x512

        x = F.relu(self.bn_final(x), inplace=True)          # final BN+ReLU

        x = self.avgpool(x)                                 # 1x1   x512
        x = self.flatten(x)                                 # 512
        x = self.fc(x)                                      # 100
        return x


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(">>> Running on device:", device)

    epochs     = int(os.getenv("EPOCHS",     "50"))
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    lr_initial = float(os.getenv("LR_INITIAL", "0.001"))

    data_root = os.getenv("CIFAR100_BIN_ROOT", "data/cifar-100-binary")

    print(f">>> Using CIFAR-100 bin data at: {data_root}")
    print(f">>> Epochs: {epochs}, Batch size: {batch_size}, LR: {lr_initial}")

    train_set = CIFAR100Bin(root=data_root, train=True,  transform=train_transform)
    test_set  = CIFAR100Bin(root=data_root, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2)

    model = WRN16_8CIFAR100(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr_initial,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=5e-4,
        amsgrad=True,
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        epoch_start      = time.time()
        last_100_start   = time.time()

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
                batch_loss = loss.item()
                batch_acc  = 100.0 * predicted.eq(targets).sum().item() / targets.size(0)
                elapsed_100 = time.time() - last_100_start
                last_100_start = time.time()
                print(
                    f"[Train Batch {batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}% | "
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
            for inputs, targets in test_loader:
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
            f"Epoch {epoch}/{epochs} Completed in {epoch_time:.2f}s\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%"
        )

    print("\n>>> CIFAR-100 WRN-16-8 training completed.")


if __name__ == "__main__":
    main()
