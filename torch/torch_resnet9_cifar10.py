import csv
import datetime
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
    img = normalize(img)
    return img

def test_transform(img: torch.Tensor) -> torch.Tensor:
    img = normalize(img)
    return img

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

class ResNet9CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.res1 = BasicResidualBlock(128)
        self.res2 = BasicResidualBlock(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn3   = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.res3 = BasicResidualBlock(256)
        self.res4 = BasicResidualBlock(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn4   = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.res5 = BasicResidualBlock(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(512, num_classes, bias=True)

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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    batch_csv_path = os.path.join(log_dir, f"torch_cifar10_resnet9_batch_{ts}.csv")
    epoch_csv_path = os.path.join(log_dir, f"torch_cifar10_resnet9_epoch_{ts}.csv")
    val_csv_path   = os.path.join(log_dir, f"torch_cifar10_resnet9_val_{ts}.csv")

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
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            step_start = time.time()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            step_ms = int((time.time() - step_start) * 1000)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            running_total += targets.size(0)
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
        train_acc = 100.0 * running_correct / running_total

        scheduler.step()

        
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_step, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                step_loss = loss.item()
                step_acc  = 100.0 * predicted.eq(targets).sum().item() / inputs.size(0)
                val_writer.writerow([epoch, val_step + 1, f"{step_loss:.6f}", f"{step_acc:.4f}"])
            val_csv_file.flush()

        val_loss = val_loss_sum / val_total
        val_acc = 100.0 * val_correct / val_total

        epoch_time = time.time() - epoch_start  

        epoch_writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{val_loss:.6f}", f"{val_acc:.4f}"])
        epoch_csv_file.flush()

        print(
            f"Epoch {epoch}/{epochs} Completed in {epoch_time:.2f}s\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%"
        )

    batch_csv_file.close()
    epoch_csv_file.close()
    val_csv_file.close()
    print(f"\n>>> Logs saved to {log_dir}/torch_cifar10_resnet9_*_{ts}.csv")
    print("\n>>> CIFAR-10 ResNet-9 (single model) training completed on CPU (6 threads).")


if __name__ == "__main__":
    main()
