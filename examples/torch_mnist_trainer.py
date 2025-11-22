import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import time
import os
from typing import Tuple, List

# MNIST Constants (matching C++ implementation)
class MNISTConstants:
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    NUM_CLASSES = 10
    NUM_CHANNELS = 1
    EPSILON = 1e-15
    PROGRESS_PRINT_INTERVAL = 100
    EPOCHS = 3
    BATCH_SIZE = 64
    LR_DECAY_INTERVAL = 2
    LR_DECAY_FACTOR = 0.8
    LR_INITIAL = 0.01

class MNISTDataset(Dataset):
    """Custom MNIST Dataset to load from CSV files (matching C++ data loader)"""
    
    def __init__(self, csv_file: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Extract labels and pixel data
        if 'label' in self.data.columns:
            # Training data format: label, pixel0, pixel1, ..., pixel783
            self.labels = self.data['label'].values
            self.images = self.data.drop('label', axis=1).values
        else:
            # Test data format: pixel0, pixel1, ..., pixel783 (no labels)
            self.labels = None
            self.images = self.data.values
            
        # Reshape images to 28x28
        self.images = self.images.reshape(-1, MNISTConstants.IMAGE_HEIGHT, MNISTConstants.IMAGE_WIDTH)
        # Normalize to [0, 1] range (matching C++ normalization)
        self.images = self.images.astype(np.float32) / 255.0
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Convert to tensor and add channel dimension
        image = torch.tensor(image).unsqueeze(0)  # Shape: (1, 28, 28)
        
        if self.transform:
            image = self.transform(image)
            
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            return image

class OptimizedMNISTCNN(nn.Module):
    """CNN architecture matching the C++ implementation exactly"""
    
    def __init__(self, enable_profiling=False):
        super(OptimizedMNISTCNN, self).__init__()
        
        # Architecture matching C++ model:
        # .input({1, 28, 28})
        # .conv2d(8, 5, 5, 1, 1, 0, 0, "elu", true, "conv1")
        # .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")
        # .conv2d(16, 1, 1, 1, 1, 0, 0, "elu", true, "conv2_1x1")
        # .conv2d(48, 5, 5, 1, 1, 0, 0, "elu", true, "conv3")
        # .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
        # .flatten("flatten")
        # .dense(10, true, "output")
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        
        self.conv2_1x1 = nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.conv3 = nn.Conv2d(16, 48, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the flattened size after convolutions and pooling
        # Input: 28x28
        # After conv1 (5x5, no padding): 24x24
        # After pool1 (3x3, stride 3): 8x8
        # After conv2_1x1 (1x1): 8x8
        # After conv3 (5x5, no padding): 4x4
        # After pool2 (2x2, stride 2): 2x2
        # So flattened size = 48 * 2 * 2 = 192
        
        self.fc = nn.Linear(48 * 2 * 2, MNISTConstants.NUM_CLASSES, bias=True)
        
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
        # Conv1 + ELU
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Conv2 1x1 + ELU
        x = F.relu(self.conv2_1x1(x))
        
        # Conv3 + ELU
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Output layer (linear activation)
        x = self.fc(x)
        
        return x
    
    def _forward_with_profiling(self, x):
        """Forward pass with layer-wise timing using high-precision perf_counter()"""
        layer_times = {}
        
        # Conv1 + ELU
        start_time = time.perf_counter()
        x = F.relu(self.conv1(x))
        layer_times['conv1'] = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Pool1
        start_time = time.perf_counter()
        x = self.pool1(x)
        layer_times['pool1'] = (time.perf_counter() - start_time) * 1000
        
        # Conv2 1x1 + ELU
        start_time = time.perf_counter()
        x = F.relu(self.conv2_1x1(x))
        layer_times['conv2_1x1'] = (time.perf_counter() - start_time) * 1000
        
        # Conv3 + ELU
        start_time = time.perf_counter()
        x = F.relu(self.conv3(x))
        layer_times['conv3'] = (time.perf_counter() - start_time) * 1000
        
        # Pool2
        start_time = time.perf_counter()
        x = self.pool2(x)
        layer_times['pool2'] = (time.perf_counter() - start_time) * 1000
        
        # Flatten
        start_time = time.perf_counter()
        x = x.view(x.size(0), -1)
        layer_times['flatten'] = (time.perf_counter() - start_time) * 1000
        
        # Output layer (linear activation)
        start_time = time.perf_counter()
        x = self.fc(x)
        layer_times['output'] = (time.perf_counter() - start_time) * 1000
        
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
        print("Performance Profile: PyTorch MNIST CNN")
        print("=" * 60)
        print(f"{'Layer':<15} {'Forward (ms)':<15} {'Total (ms)':<15}")
        print("-" * 60)
        
        total_time = 0.0
        for layer_name in ['conv1', 'pool1', 'conv2_1x1', 'conv3', 'pool2', 'flatten', 'output']:
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
    
    start_time = time.perf_counter()
    
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
        
        if batch_idx % MNISTConstants.PROGRESS_PRINT_INTERVAL == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.6f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')
            
            # Print performance profile if profiling is enabled
            if hasattr(model, 'enable_profiling') and model.enable_profiling:
                model.print_performance_profile()
                model.reset_profiling_stats()  # Reset for next interval
    
    epoch_time = time.perf_counter() - start_time
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
    print("PyTorch MNIST CNN Trainer (CPU Only)")
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
    print("\nLoading MNIST data...")
    train_dataset = MNISTDataset('./data/mnist/train.csv')
    test_dataset = MNISTDataset('./data/mnist/test.csv')
    
    print(f"Successfully loaded training data: {len(train_dataset)} samples")
    print(f"Successfully loaded test data: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=MNISTConstants.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # Keep at 0 for CPU-only consistent performance
        pin_memory=False  # No GPU, so no need for pinned memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=MNISTConstants.BATCH_SIZE, 
        shuffle=False,
        num_workers=0,  # Keep at 0 for CPU-only consistent performance
        pin_memory=False  # No GPU, so no need for pinned memory
    )
    
    # Create model
    print("\nBuilding CNN model architecture with automatic shape inference...")
    model = OptimizedMNISTCNN(enable_profiling=True).to(device)  # Enable profiling
    
    # Print model summary
    print_model_summary(model, (MNISTConstants.BATCH_SIZE, 1, 
                               MNISTConstants.IMAGE_HEIGHT, MNISTConstants.IMAGE_WIDTH))
    
    # Create optimizer (Adam to match C++ implementation)
    optimizer = optim.Adam(model.parameters(), lr=MNISTConstants.LR_INITIAL, 
                          betas=(0.9, 0.999), eps=1e-8)
    
    # Create loss function (CrossEntropy with label smoothing to match C++ epsilon)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)  # No smoothing initially
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=MNISTConstants.LR_DECAY_INTERVAL, 
                                         gamma=MNISTConstants.LR_DECAY_FACTOR)
    
    print(f"\nStarting training for {MNISTConstants.EPOCHS} epochs...")
    print(f"Batch size: {MNISTConstants.BATCH_SIZE}")
    print(f"Initial learning rate: {MNISTConstants.LR_INITIAL}")
    print(f"LR decay factor: {MNISTConstants.LR_DECAY_FACTOR} every {MNISTConstants.LR_DECAY_INTERVAL} epochs")
    
    # Training loop
    training_start_time = time.perf_counter()
    best_test_accuracy = 0.0
    
    for epoch in range(1, MNISTConstants.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{MNISTConstants.EPOCHS}")
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
    
    total_training_time = time.perf_counter() - training_start_time
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Best test accuracy: {best_test_accuracy:.2f}%")
    print("="*60)
    
    # Save the model
    try:
        os.makedirs('./model_snapshots', exist_ok=True)
        save_model(model, './model_snapshots/pytorch_mnist_cnn_model.pth')
    except Exception as e:
        print(f"Warning: Failed to save model: {e}")

if __name__ == "__main__":
    main()