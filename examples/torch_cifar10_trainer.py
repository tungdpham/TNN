import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import time
import os
import struct
from typing import Tuple, List

# CIFAR-10 Constants (matching C++ implementation)
class CIFAR10Constants:
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    EPSILON = 1e-15
    PROGRESS_PRINT_INTERVAL = 50
    EPOCHS = 20
    BATCH_SIZE = 32
    LR_DECAY_INTERVAL = 10
    LR_DECAY_FACTOR = 0.85
    LR_INITIAL = 0.005

class CIFAR10Dataset(Dataset):
    """Custom CIFAR-10 Dataset to load from binary files (matching C++ data loader)"""
    
    def __init__(self, file_paths: List[str], transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        for file_path in file_paths:
            self._load_batch_file(file_path)
        
        # Convert to numpy arrays
        self.images = np.array(self.images, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # Normalize to [0, 1] range (matching C++ normalization)
        self.images = self.images / 255.0
        
        print(f"Loaded {len(self.images)} samples from {len(file_paths)} file(s)")
        
    def _load_batch_file(self, file_path: str):
        """Load a single CIFAR-10 batch file"""
        try:
            with open(file_path, 'rb') as f:
                while True:
                    # Read label (1 byte)
                    label_data = f.read(1)
                    if not label_data:
                        break
                    label = struct.unpack('B', label_data)[0]
                    
                    # Read image data (3072 bytes = 32*32*3)
                    image_data = f.read(3072)
                    if len(image_data) != 3072:
                        break
                    
                    # Convert to numpy array and reshape
                    image = np.frombuffer(image_data, dtype=np.uint8)
                    # CIFAR-10 format: R channel, G channel, B channel (each 32x32)
                    image = image.reshape(3, 32, 32)
                    # Convert to HWC format (Height, Width, Channels)
                    image = np.transpose(image, (1, 2, 0))
                    
                    self.images.append(image)
                    self.labels.append(label)
                    
        except FileNotFoundError:
            print(f"Warning: Could not find file {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor and change from HWC to CHW format
        image = torch.tensor(image).permute(2, 0, 1)  # Shape: (3, 32, 32)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class OptimizedCIFAR10CNN(nn.Module):
    """CNN architecture matching the C++ implementation exactly"""
    
    def __init__(self, enable_profiling=False):
        super(OptimizedCIFAR10CNN, self).__init__()
        
        # Architecture matching C++ model:
        # .input({3, 32, 32})
        # .conv2d(16, 3, 3, 1, 1, 0, 0, "relu", true, "conv1")
        # .maxpool2d(3, 3, 3, 3, 0, 0, "maxpool1")
        # .conv2d(64, 3, 3, 1, 1, 0, 0, "relu", true, "conv2")
        # .maxpool2d(4, 4, 4, 4, 0, 0, "maxpool2")
        # .flatten("flatten")
        # .dense(10, "linear", true, "fc1")
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        
        # Calculate the flattened size after convolutions and pooling
        # Input: 32x32
        # After conv1 (3x3, no padding): 30x30
        # After maxpool1 (3x3, stride 3): 10x10
        # After conv2 (3x3, no padding): 8x8
        # After maxpool2 (4x4, stride 4): 2x2
        # So flattened size = 64 * 2 * 2 = 256
        
        self.fc1 = nn.Linear(64 * 2 * 2, CIFAR10Constants.NUM_CLASSES, bias=True)
        
        # Profiling setup
        self.enable_profiling = enable_profiling
        self.layer_times = {}
        self.profile_count = 0
        
    def forward(self, x):
        if self.enable_profiling:
            return self._forward_with_profiling(x)
        else:
            return self._forward_normal(x)
    
    def _forward_normal(self, x):
        # Conv1 + ReLU
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        
        # Conv2 + ReLU
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Output layer (linear activation)
        x = self.fc1(x)
        
        return x
    
    def _forward_with_profiling(self, x):
        layer_times = {}
        
        # Conv1 + ReLU
        start_time = time.time()
        x = F.relu(self.conv1(x))
        layer_times['conv1'] = (time.time() - start_time) * 1000  # Convert to ms
        
        # MaxPool1
        start_time = time.time()
        x = self.maxpool1(x)
        layer_times['maxpool1'] = (time.time() - start_time) * 1000
        
        # Conv2 + ReLU
        start_time = time.time()
        x = F.relu(self.conv2(x))
        layer_times['conv2'] = (time.time() - start_time) * 1000
        
        # MaxPool2
        start_time = time.time()
        x = self.maxpool2(x)
        layer_times['maxpool2'] = (time.time() - start_time) * 1000
        
        # Flatten
        start_time = time.time()
        x = x.view(x.size(0), -1)
        layer_times['flatten'] = (time.time() - start_time) * 1000
        
        # Output layer (linear activation)
        start_time = time.time()
        x = self.fc1(x)
        layer_times['fc1'] = (time.time() - start_time) * 1000
        
        # Accumulate timing statistics
        for layer_name, layer_time in layer_times.items():
            if layer_name not in self.layer_times:
                self.layer_times[layer_name] = []
            self.layer_times[layer_name].append(layer_time)
        
        return x
    
    def print_performance_profile(self):
        """Print performance profile similar to C++ implementation"""
        if not self.enable_profiling or not self.layer_times:
            print("Profiling not enabled or no timing data available")
            return
        
        print("=" * 60)
        print("Performance Profile: PyTorch CIFAR-10 CNN")
        print("=" * 60)
        print(f"{'Layer':<15} {'Forward (ms)':<15} {'Total (ms)':<15}")
        print("-" * 60)
        
        total_time = 0.0
        for layer_name in ['conv1', 'maxpool1', 'conv2', 'maxpool2', 'flatten', 'fc1']:
            if layer_name in self.layer_times:
                times = self.layer_times[layer_name]
                avg_time = sum(times) / len(times)
                print(f"{layer_name:<15} {avg_time:<15.3f} {avg_time:<15.3f}")
                total_time += avg_time
        
        print("-" * 60)
        print(f"{'TOTAL':<15} {total_time:<15.3f} {total_time:<15.3f}")
        print("=" * 60)
    
    def reset_profiling_stats(self):
        """Reset profiling statistics"""
        self.layer_times = {}
        self.profile_count = 0

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device, epoch: int) -> Tuple[float, float]:
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % CIFAR10Constants.PROGRESS_PRINT_INTERVAL == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.6f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')
            
            # Print performance profile if profiling is enabled
            if hasattr(model, 'enable_profiling') and model.enable_profiling:
                model.print_performance_profile()
                model.reset_profiling_stats()  # Reset for next interval
    
    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    print(f'Epoch {epoch} completed in {epoch_time:.2f}s - '
          f'Average Loss: {avg_loss:.6f}, Training Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy

def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                   criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Test Loss: {avg_loss:.6f}, Test Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy

def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...]):
    """Print model architecture summary"""
    print("\nModel Architecture Summary:")
    print("=" * 50)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if params > 0:
                print(f"{name:15} {str(module):30} Params: {params:8}")
                total_params += params
                trainable_params += trainable
    
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50)
    
    # Test forward pass with dummy input
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_shape)
        try:
            output = model(dummy_input)
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error in forward pass: {e}")

def save_model(model: nn.Module, filepath: str):
    """Save model state dict"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
    }, filepath)
    print(f"Model saved to: {filepath}")

def main():
    print("PyTorch CIFAR-10 CNN Trainer (CPU Only)")
    print("=" * 50)
    
    # Force CPU usage for fair comparison with C++ implementation
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Set number of threads for CPU computation (matching C++ OpenMP threads)
    torch.set_num_threads(8)
    print(f"PyTorch CPU threads: {torch.get_num_threads()}")
    
    if torch.cuda.is_available():
        print(f"Note: GPU available but using CPU for fair comparison with C++ implementation")
    
    # Load datasets
    print("\nLoading CIFAR-10 data...")
    
    # Training files (matching C++ data loader)
    train_files = []
    for i in range(1, 6):
        train_files.append(f"./data/cifar-10-batches-bin/data_batch_{i}.bin")
    
    test_files = ["./data/cifar-10-batches-bin/test_batch.bin"]
    
    train_dataset = CIFAR10Dataset(train_files)
    test_dataset = CIFAR10Dataset(test_files)
    
    print(f"Successfully loaded training data: {len(train_dataset)} samples")
    print(f"Successfully loaded test data: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CIFAR10Constants.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # Keep at 0 for CPU-only consistent performance
        pin_memory=False  # No GPU, so no need for pinned memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CIFAR10Constants.BATCH_SIZE, 
        shuffle=False,
        num_workers=0,  # Keep at 0 for CPU-only consistent performance
        pin_memory=False  # No GPU, so no need for pinned memory
    )
    
    # Create model
    print("\nBuilding CNN model architecture for CIFAR-10...")
    model = OptimizedCIFAR10CNN(enable_profiling=True).to(device)  # Enable profiling
    
    # Print model summary
    print_model_summary(model, (CIFAR10Constants.BATCH_SIZE, 3, 
                               CIFAR10Constants.IMAGE_HEIGHT, CIFAR10Constants.IMAGE_WIDTH))
    
    # Create optimizer (SGD with momentum to match C++ implementation)
    optimizer = optim.SGD(model.parameters(), lr=CIFAR10Constants.LR_INITIAL, 
                         momentum=0.9, weight_decay=0.0)
    
    # Create loss function (CrossEntropy to match C++ epsilon)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)  # No smoothing initially
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=CIFAR10Constants.LR_DECAY_INTERVAL, 
                                         gamma=CIFAR10Constants.LR_DECAY_FACTOR)
    
    print(f"\nStarting CIFAR-10 CNN training...")
    print(f"Epochs: {CIFAR10Constants.EPOCHS}")
    print(f"Batch size: {CIFAR10Constants.BATCH_SIZE}")
    print(f"Initial learning rate: {CIFAR10Constants.LR_INITIAL}")
    print(f"LR decay factor: {CIFAR10Constants.LR_DECAY_FACTOR} every {CIFAR10Constants.LR_DECAY_INTERVAL} epochs")
    
    # Training loop
    training_start_time = time.time()
    best_test_accuracy = 0.0
    
    for epoch in range(1, CIFAR10Constants.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{CIFAR10Constants.EPOCHS}")
        print("-" * 30)
        
        # Train for one epoch
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Evaluate on test set
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate updated to: {current_lr:.6f}")
        
        # Track best test accuracy
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            print(f"New best test accuracy: {best_test_accuracy:.2f}%")
    
    total_training_time = time.time() - training_start_time
    
    print("\n" + "="*60)
    print("CIFAR-10 CNN PyTorch model training completed successfully!")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Best test accuracy: {best_test_accuracy:.2f}%")
    print("="*60)
    
    # Save the model
    try:
        os.makedirs('./model_snapshots', exist_ok=True)
        save_model(model, './model_snapshots/pytorch_cifar10_cnn_model.pth')
    except Exception as e:
        print(f"Warning: Failed to save model: {e}")

if __name__ == "__main__":
    main()
