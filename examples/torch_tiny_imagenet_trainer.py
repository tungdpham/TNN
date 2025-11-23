#!/usr/bin/env python3
"""
PyTorch implementation of Tiny ImageNet trainer for comparison with TNN.
This implementation mirrors the TNN configuration as closely as possible.
"""

import os
import time
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# ======================== Configuration ========================
class TrainingConfig:
    """Training configuration matching TNN setup"""
    def __init__(self):
        # Training parameters from .env
        self.epochs = int(os.getenv('EPOCHS', '5'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '128'))
        self.lr_initial = float(os.getenv('LR_INITIAL', '0.1'))
        self.lr_decay_factor = float(os.getenv('LR_DECAY_FACTOR', '0.9'))
        self.lr_decay_interval = int(os.getenv('LR_DECAY_INTERVAL', '5'))
        self.progress_print_interval = int(os.getenv('PROGRESS_PRINT_INTERVAL', '10'))
        self.device_type = os.getenv('DEVICE_TYPE', 'CPU')
        
        # Adam optimizer params (matching TNN: lr, beta1=0.9, beta2=0.999, eps=1e-7)
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-7
        
        # Dataset
        self.dataset_path = 'data/tiny-imagenet-200'
        self.num_classes = 200
        self.image_size = 64
        
        # Workers for data loading
        self.num_workers = 4
        
    def print_config(self):
        print("\nTraining Configuration:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Initial Learning Rate: {self.lr_initial}")
        print(f"  LR Decay Factor: {self.lr_decay_factor}")
        print(f"  LR Decay Interval (epochs): {self.lr_decay_interval}")
        print(f"  Progress Print Interval: {self.progress_print_interval}")
        print(f"  Device Type: {self.device_type}")
        print(f"  Adam Beta1: {self.adam_beta1}")
        print(f"  Adam Beta2: {self.adam_beta2}")
        print(f"  Adam Epsilon: {self.adam_eps}")
        print(f"  Dataset Path: {self.dataset_path}")
        print(f"  Number of Classes: {self.num_classes}")
        print()


# ======================== ResNet-18 Model ========================
class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.1)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.1)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class ResNet18TinyImageNet(nn.Module):
    """
    ResNet-18 for Tiny ImageNet (64x64 images, 200 classes)
    Architecture matches TNN's create_resnet18_tiny_imagenet()
    """
    def __init__(self, num_classes=200):
        super(ResNet18TinyImageNet, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual layers matching TNN architecture
        # Layer 1: 32 -> 64 channels
        self.layer1 = nn.Sequential(
            BasicBlock(32, 64, stride=1),  # layer1_block1
            BasicBlock(64, 64, stride=1)   # layer1_block2
        )
        
        # Layer 2: 64 -> 128 channels with stride 2
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),   # layer2_block1
            BasicBlock(128, 128, stride=1)   # layer2_block2
        )
        
        # Layer 3: 128 -> 256 channels with stride 2
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),  # layer3_block1
            BasicBlock(256, 256, stride=1)   # layer3_block2
        )
        
        # Layer 4: 256 -> 512 channels with stride 2
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),  # layer4_block1
            BasicBlock(512, 512, stride=1)   # layer4_block2
        )
        
        # Global average pooling and classifier
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x


# ======================== Data Loading ========================
class TinyImageNetDataset(Dataset):
    """Custom dataset for Tiny ImageNet"""
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Args:
            root_dir: Path to tiny-imagenet-200 directory
            split: 'train' or 'val'
            transform: torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {}
        
        if split == 'train':
            self._load_train()
        else:
            self._load_val()
    
    def _load_train(self):
        """Load training data"""
        train_dir = self.root_dir / 'train'
        
        # Build class to index mapping
        classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Load samples
        for class_name in classes:
            class_dir = train_dir / class_name / 'images'
            if not class_dir.exists():
                continue
            
            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.glob('*.JPEG'):
                self.samples.append((str(img_path), class_idx))
    
    def _load_val(self):
        """Load validation data"""
        val_dir = self.root_dir / 'val'
        val_annotations = val_dir / 'val_annotations.txt'
        
        # Build class to index mapping (same as train)
        train_dir = self.root_dir / 'train'
        classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Parse validation annotations
        with open(val_annotations, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_name = parts[1]
                
                img_path = val_dir / 'images' / img_name
                if img_path.exists() and class_name in self.class_to_idx:
                    class_idx = self.class_to_idx[class_name]
                    self.samples.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_augmentation():
    """
    Data augmentation matching TNN setup:
    - horizontal_flip(0.2)
    - rotation(0.2, 5.0)
    - brightness(0.1, 0.15)
    - contrast(0.1, 0.15)
    - gaussian_noise(0.1, 0.05)
    - random_crop(0.25, 4)
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.2),
        # transforms.RandomRotation(degrees=5.0),
        # transforms.ColorJitter(brightness=0.15, contrast=0.15),
        # transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        # # Add Gaussian noise with probability 0.1
        # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05 if torch.rand(1).item() < 0.1 else x),
        # # Normalize (using ImageNet stats as common practice)
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform():
    """Validation transform (no augmentation)"""
    return transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ======================== Training Functions ========================
def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                config: TrainingConfig) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % config.progress_print_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            
            print(f"Epoch {epoch+1}/{config.epochs} | "
                  f"Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Acc: {accuracy:.2f}% | "
                  f"Speed: {batches_per_sec:.2f} batches/s")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# ======================== Main Training Loop ========================
def main():
    # Load configuration
    config = TrainingConfig()
    config.print_config()
    
    # Set device
    if config.device_type == 'GPU' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        device = torch.device('cpu')
        print("Using CPU\n")
    
    # Create datasets
    print("Loading training data...")
    train_transform = get_data_augmentation()
    train_dataset = TinyImageNetDataset(
        config.dataset_path, 
        split='train', 
        transform=train_transform
    )
    print(f"Successfully loaded training data: {len(train_dataset)} samples")
    
    print("\nLoading validation data...")
    val_transform = get_val_transform()
    val_dataset = TinyImageNetDataset(
        config.dataset_path,
        split='val',
        transform=val_transform
    )
    print(f"Successfully loaded validation data: {len(val_dataset)} samples\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # Create model
    print("Building ResNet-18 model architecture for Tiny ImageNet...")
    model = ResNet18TinyImageNet(num_classes=config.num_classes)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Loss function (CrossEntropyLoss combines softmax + NLLLoss)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (matching TNN Adam params)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr_initial,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_decay_interval,
        gamma=config.lr_decay_factor
    )
    
    # Training loop
    print("Starting Tiny ImageNet CNN training...")
    print("=" * 80)
    
    best_val_acc = 0.0
    training_history = []
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print("=" * 80)
        print(f"Epoch {epoch+1}/{config.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print("=" * 80)
        print()
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr,
            'time': epoch_time
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'model_snapshots/torch_resnet18_tiny_imagenet_best.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%\n")
    
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': training_history,
        'config': vars(config)
    }, 'model_snapshots/torch_resnet18_tiny_imagenet_final.pth')
    print("Saved final model checkpoint.")
    
    # Write training log
    with open('torch_tiny_imagenet_log.txt', 'w') as f:
        f.write("PyTorch Tiny ImageNet Training Log\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Configuration:\n")
        for key, value in vars(config).items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("Training History:\n")
        for record in training_history:
            f.write(f"Epoch {record['epoch']}: "
                   f"Train Loss={record['train_loss']:.4f}, "
                   f"Train Acc={record['train_acc']:.2f}%, "
                   f"Val Loss={record['val_loss']:.4f}, "
                   f"Val Acc={record['val_acc']:.2f}%, "
                   f"LR={record['lr']:.6f}, "
                   f"Time={record['time']:.2f}s\n")
        f.write(f"\nBest Validation Accuracy: {best_val_acc:.2f}%\n")
    
    print("Training log saved to 'torch_tiny_imagenet_log.txt'")


if __name__ == '__main__':
    main()
