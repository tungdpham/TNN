import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from dotenv import load_dotenv

load_dotenv()


# ======================== Dataset ========================

class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet dataset loader (validation split).
    Expects the standard tiny-imagenet-200 directory layout:
      train/<wnid>/images/*.JPEG  (used to build canonical class->idx mapping)
      val/images/*.JPEG  +  val/val_annotations.txt
    """
    def __init__(self, root: str, train: bool = False, transform=None):
        self.transform = transform
        root = Path(root)

        self.samples = []
        self.class_to_idx = {}

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


TINY_MEAN = [0.4802, 0.4481, 0.3975]
TINY_STD  = [0.2770, 0.2691, 0.2821]

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
               -> Conv1x1(no bias) -> BN+ReLU
    Shortcut:  Conv1x1(stride, no bias) -> BN (no ReLU) when shapes differ; else identity
    Post-add:  ReLU
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
        out = F.relu(self.bn3(self.conv3(out)),  inplace=True)

        out = out + sc
        out = F.relu(out, inplace=True)
        return out


class ResNet50TinyImageNet(nn.Module):
    """
    ResNet-50 for Tiny ImageNet (64x64 inputs, 200 classes).
    Matches create_tiny_imagenet_resnet50() in example_models.cpp.
    """
    def __init__(self, num_classes: int = 200):
        super().__init__()

        self.conv1   = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.bn1     = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BottleneckBlock( 64,  64, 256, stride=1),
            BottleneckBlock(256,  64, 256, stride=1),
            BottleneckBlock(256,  64, 256, stride=1),
        )

        self.layer2 = nn.Sequential(
            BottleneckBlock(256, 128, 512, stride=2),
            BottleneckBlock(512, 128, 512, stride=1),
            BottleneckBlock(512, 128, 512, stride=1),
            BottleneckBlock(512, 128, 512, stride=1),
        )

        self.layer3 = nn.Sequential(
            BottleneckBlock( 512, 256, 1024, stride=2),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
        )

        self.layer4 = nn.Sequential(
            BottleneckBlock(1024, 512, 2048, stride=2),
            BottleneckBlock(2048, 512, 2048, stride=1),
            BottleneckBlock(2048, 512, 2048, stride=1),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ======================== Inference ========================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(">>> Running inference on device:", device)

    batch_size = int(os.getenv("BATCH_SIZE", "256"))
    data_root  = os.getenv("TINY_IMAGENET_ROOT", "data/tiny-imagenet-200")
    model_path = os.getenv("MODEL_PATH", "model_snapshots/resnet50_tiny_imagenet.pth")

    print(f">>> Data root  : {data_root}")
    print(f">>> Batch size : {batch_size}")
    print(f">>> Model path : {model_path}")

    val_set = TinyImageNetDataset(root=data_root, train=False,
                                  transform=test_transform())
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f">>> Val samples: {len(val_set)}")

    model = ResNet50TinyImageNet(num_classes=200).to(device)

    if os.path.isfile(model_path):
        print(f">>> Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(">>> Weights loaded successfully")
    else:
        print(f">>> Warning: model file not found at {model_path}")
        print(">>> Running with randomly initialised weights")

    model.eval()
    criterion = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct  = 0
    total    = 0

    inference_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 10 == 0:
                batch_acc = 100.0 * predicted.eq(targets).sum().item() / targets.size(0)
                print(
                    f"[Batch {batch_idx+1}/{len(val_loader)}] "
                    f"Batch Acc: {batch_acc:.2f}%"
                )

    inference_time = time.time() - inference_start

    test_loss = loss_sum / total
    test_acc  = 100.0 * correct / total

    print(f"\n>>> Inference completed in {inference_time:.2f}s")
    print(f">>> Test Loss    : {test_loss:.4f}")
    print(f">>> Test Accuracy: {test_acc:.2f}%")
    print(f">>> Throughput   : {total / inference_time:.2f} images/sec")


if __name__ == "__main__":
    main()
