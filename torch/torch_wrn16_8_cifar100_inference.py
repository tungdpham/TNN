import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def test_transform(img: torch.Tensor) -> torch.Tensor:
    img = normalize(img)
    return img


class WideResidualBlock(nn.Module):
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
        sc = self.shortcut(x) if self.shortcut is not None else x

        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)

        return out + sc


class WRN16_8CIFAR100(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        width_factor  = 8
        dropout_rate  = 0.3
        c1 = 16 * width_factor   # 128
        c2 = 32 * width_factor   # 256
        c3 = 64 * width_factor   # 512

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=True)

        self.group1_block1 = WideResidualBlock(16, c1, stride=1, dropout_rate=dropout_rate)
        self.group1_block2 = WideResidualBlock(c1, c1, stride=1, dropout_rate=dropout_rate)

        self.group2_block1 = WideResidualBlock(c1, c2, stride=2, dropout_rate=dropout_rate)
        self.group2_block2 = WideResidualBlock(c2, c2, stride=1, dropout_rate=dropout_rate)

        self.group3_block1 = WideResidualBlock(c2, c3, stride=2, dropout_rate=dropout_rate)
        self.group3_block2 = WideResidualBlock(c3, c3, stride=1, dropout_rate=dropout_rate)

        self.bn_final = nn.BatchNorm2d(c3, eps=1e-5, momentum=0.1)

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(c3, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)

        x = self.group1_block1(x)
        x = self.group1_block2(x)

        x = self.group2_block1(x)
        x = self.group2_block2(x)

        x = self.group3_block1(x)
        x = self.group3_block2(x)

        x = F.relu(self.bn_final(x), inplace=True)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(">>> Running inference on device:", device)

    batch_size = 256
    data_root  = os.getenv("CIFAR100_BIN_ROOT", "data/cifar-100-binary")
    model_path = os.getenv("MODEL_PATH", "model_snapshots/wrn16_8_cifar100.pth")

    print(f">>> Using CIFAR-100 bin data at: {data_root}")
    print(f">>> Batch size: {batch_size}")
    print(f">>> Model path: {model_path}")

    test_set = CIFAR100Bin(root=data_root, train=False, transform=test_transform)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = WRN16_8CIFAR100(num_classes=100).to(device)

    if os.path.isfile(model_path):
        print(f">>> Loading model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(">>> Model weights loaded successfully")
    else:
        print(f">>> Warning: Model file not found at {model_path}")
        print(">>> Running inference with randomly initialized weights")

    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss_sum = 0.0
    test_correct  = 0
    test_total    = 0

    inference_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            test_loss_sum += loss.item() * inputs.size(0)
            _, predicted   = outputs.max(1)
            test_total    += targets.size(0)
            test_correct  += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 10 == 0:
                batch_acc = 100.0 * predicted.eq(targets).sum().item() / targets.size(0)
                print(
                    f"[Inference Batch {batch_idx+1}/{len(test_loader)}] "
                    f"Batch Acc: {batch_acc:.2f}%"
                )

    inference_time = time.time() - inference_start

    test_loss = test_loss_sum / test_total
    test_acc  = 100.0 * test_correct / test_total

    print(f"\n>>> Inference completed in {inference_time:.2f}s")
    print(f">>> Test Loss: {test_loss:.4f}")
    print(f">>> Test Accuracy: {test_acc:.2f}%")
    print(f">>> Throughput: {test_total / inference_time:.2f} images/sec")


if __name__ == "__main__":
    main()
