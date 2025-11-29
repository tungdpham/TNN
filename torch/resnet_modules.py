import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import os

# Load .env variables (e.g., CIFAR10_BIN_ROOT)
load_dotenv()

# =====================================================
# Augmentation and Normalization (Copied for consistency)
# =====================================================
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

# =====================================================
# CIFAR-10 .bin loader (Copied for consistency)
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
                # Note: File not found error assumes CIFAR-10 data is correctly placed.
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

# =====================================================
# ResNet-9 Modules (Updated with MaxPool layers)
# =====================================================
class BasicResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class ResNet9Part1(nn.Module):
    """
    Part 1 of ResNet-9 (Coordinator Node): 3x32x32 -> 128x16x16
    """
    def __init__(self):
        super().__init__()
        # 3x32x32 -> 64x32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)

        # 64x32x32 -> 128x32x32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)

        # MaxPool 1: 128x32x32 -> 128x16x16 (Downsampling)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Residual Blocks (128 channels, 16x16)
        self.res1 = BasicResidualBlock(128)
        self.res2 = BasicResidualBlock(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool1(x) 
        x = self.res1(x)
        x = self.res2(x)
        return x


class ResNet9Part2(nn.Module):
    """
    Part 2 of ResNet-9 (Worker Node): 128x16x16 -> 10 classes
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # --- LAYER 2: 128 -> 256 Channels (16x16 -> 8x8) ---
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3   = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        
        # MaxPool 2: 256x16x16 -> 256x8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res3 = BasicResidualBlock(256)
        self.res4 = BasicResidualBlock(256)

        # --- LAYER 3: 256 -> 512 Channels (8x8 -> 4x4) ---
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4   = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        
        # MaxPool 3: 512x8x8 -> 512x4x4
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res5 = BasicResidualBlock(512)

        # --- CLASSIFICATION HEAD ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        # Layer 2
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.pool2(x) 
        x = self.res3(x)
        x = self.res4(x)
        
        # Layer 3
        x = F.relu(self.bn4(self.conv4(x)), inplace=True) 
        x = self.pool3(x) 
        x = self.res5(x) 

        # Final Layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x