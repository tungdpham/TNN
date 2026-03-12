import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        x = self.res3(x)  
        x = self.res4(x)
        x = self.maxpool2(x)

        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.res5(x)
        x = self.maxpool3(x)

        x = self.avgpool(x)            
        x = self.flatten(x)

        x = self.fc(x)                 
        return x

def main():
    device = torch.device("cuda:0")
    print(">>> Running inference on device:", device)

    batch_size = 256
    data_root = os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin")
    model_path = os.getenv("MODEL_PATH", "model_snapshots/resnet9_cifar10.pth")

    print(f">>> Using CIFAR-10 bin data at: {data_root}")
    print(f">>> Batch size: {batch_size}")
    print(f">>> Model path: {model_path}")

    # Load train dataset
    test_set = CIFAR10Bin(root=data_root, train=True, transform=test_transform)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Initialize model
    model = ResNet9CIFAR10(num_classes=10).to(device)
    
    # Load trained weights if available
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

    # Run inference
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0
    
    inference_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                batch_acc = 100.0 * predicted.eq(targets).sum().item() / targets.size(0)
                print(
                    f"[Inference Batch {batch_idx+1}/{len(test_loader)}] "
                    f"Batch Acc: {batch_acc:.2f}%"
                )

    inference_time = time.time() - inference_start
    
    test_loss = test_loss_sum / test_total
    test_acc = 100.0 * test_correct / test_total
    
    print(f"\n>>> Inference completed in {inference_time:.2f}s")
    print(f">>> Test Loss: {test_loss:.4f}")
    print(f">>> Test Accuracy: {test_acc:.2f}%")
    print(f">>> Throughput: {test_total / inference_time:.2f} images/sec")


if __name__ == "__main__":
    main()
