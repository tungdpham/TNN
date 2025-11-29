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
load_dotenv()    # tự tìm và load file .env trong thư mục hiện tại
# =====================================================
#  GIỚI HẠN CPU = 6 THREADS
# =====================================================
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

torch.set_num_threads(6)
torch.set_num_interop_threads(6)

print(">>> Using CPU threads:", torch.get_num_threads())


# =====================================================
#  CIFAR-10 binary loader (data/cifar-10-batches-bin/*.bin)
# =====================================================
class CIFAR10Bin(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
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
                raise FileNotFoundError(f"Không tìm thấy file: {path}")
            with open(path, "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.uint8)
                arr = arr.reshape(-1, 3073)  # 1 label + 3072 pixels

                labels = arr[:, 0]
                images = arr[:, 1:].reshape(-1, 3, 32, 32)  # (N,3,32,32)

                self.data.append(images)
                self.targets.append(labels)

        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 255.0  # [0,1]
        img = torch.from_numpy(img)  # (3,32,32)
        label = int(self.targets[idx])

        if self.transform:
            img = self.transform(img)

        return img, label


# =====================================================
#  Augmentation giống TNN:
#   - random_crop(1.0, 4)
#   - horizontal_flip(0.5)
#   - normalize(mean, std)
# =====================================================
CIFAR10_MEAN = torch.tensor([0.49139968, 0.48215827, 0.44653124]).view(3, 1, 1)
CIFAR10_STD  = torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(3, 1, 1)


def normalize(img: torch.Tensor) -> torch.Tensor:
    return (img - CIFAR10_MEAN) / CIFAR10_STD


def random_horizontal_flip(img: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if random.random() < p:
        img = torch.flip(img, dims=[2])  # flip width
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


# =====================================================
#  Residual Block (basic_residual_block)
# =====================================================
class BasicResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


# =====================================================
#  ResNet-9 CIFAR-10 (full, single model)
# =====================================================
class ResNet9CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2,
                               padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)

        self.res1 = BasicResidualBlock(128)
        self.res2 = BasicResidualBlock(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2,
                               padding=1, bias=True)
        self.bn3   = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)

        self.res3 = BasicResidualBlock(256)
        self.res4 = BasicResidualBlock(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(256, num_classes, bias=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)   # conv1 + bn1 + relu1
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)   # conv2 + bn2 + relu2

        x = self.res1(x)  # res_block1
        x = self.res2(x)  # res_block2

        x = F.relu(self.bn3(self.conv3(x)), inplace=True)   # conv3 + bn3 + relu3

        x = self.res3(x)  # res_block3
        x = self.res4(x)  # res_block4

        x = self.avgpool(x)            # avgpool
        x = torch.flatten(x, 1)        # flatten
        x = self.fc(x)                 # dense(10)
        return x


# =====================================================
#  Train + Validation (có đo thời gian)
# =====================================================
def main():
    device = torch.device("cpu")
    print(">>> Running on device:", device)

    epochs = int(os.getenv("EPOCHS", "10"))
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    lr_initial = float(os.getenv("LR_INITIAL", "0.001"))

    data_root = os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin")

    print(f">>> Using CIFAR-10 bin data at: {data_root}")
    print(f">>> Epochs: {epochs}, Batch size: {batch_size}, LR: {lr_initial}")

    train_set = CIFAR10Bin(root=data_root, train=True,  transform=train_transform)
    test_set  = CIFAR10Bin(root=data_root, train=False, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = ResNet9CIFAR10(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr_initial,
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=1e-3,
        amsgrad=True,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=40, gamma=0.5
    )

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        epoch_start = time.time()
        last_100_start = time.time()

        # ---------------- TRAIN ----------------
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            running_total += targets.size(0)
            running_correct += predicted.eq(targets).sum().item()

            # ---- PRINT EVERY 100 BATCHES + TIME ----
            if (batch_idx + 1) % 100 == 0:
                batch_loss = loss.item()
                batch_acc = 100.0 * predicted.eq(targets).sum().item() / targets.size(0)
                elapsed_100 = time.time() - last_100_start
                last_100_start = time.time()
                print(
                    f"[Train Batch {batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}% | "
                    f"100-batch time: {elapsed_100:.2f}s"
                )

        train_loss = running_loss / running_total
        train_acc = 100.0 * running_correct / running_total

        scheduler.step()

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss = val_loss_sum / val_total
        val_acc = 100.0 * val_correct / val_total

        epoch_time = time.time() - epoch_start  # train + val

        print(
            f"Epoch {epoch}/{epochs} Completed in {epoch_time:.2f}s\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%"
        )

    print("\n>>> CIFAR-10 ResNet-9 (single model) training completed on CPU (6 threads).")


if __name__ == "__main__":
    main()
