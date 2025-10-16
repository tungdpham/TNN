import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
import struct
from typing import Tuple, List
from fvcore.nn import FlopCountAnalysis
os.environ['PYTORCH_JIT'] = '0'

# CIFAR-10 Constants (matching C++ implementation)
class CIFAR10Constants:
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    NORMALIZATION_FACTOR = 255.0
    RECORD_SIZE = 1 + IMAGE_SIZE
    EPSILON = 1e-15
    PROGRESS_PRINT_INTERVAL = 100
    EPOCHS = 40
    BATCH_SIZE = 32
    LR_DECAY_INTERVAL = 5
    LR_DECAY_FACTOR = 0.85
    LR_INITIAL = 0.001

class CIFAR10Dataset(Dataset):
    """Custom CIFAR-10 Dataset to load from binary files (matching C++ data loader)"""
    
    def __init__(self, file_paths: List[str], transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        
        # Load data from binary files
        for file_path in file_paths:
            self._load_binary_file(file_path)
        
        # Convert to numpy arrays
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # Normalize to [0, 1] range (matching C++ normalization)
        self.data = self.data / CIFAR10Constants.NORMALIZATION_FACTOR
        
        print(f"Loaded {len(self.data)} CIFAR-10 samples from {len(file_paths)} files")
        
    def _load_binary_file(self, file_path: str):
        """Load CIFAR-10 binary file format"""
        try:
            with open(file_path, 'rb') as f:
                while True:
                    # Read one record (1 byte label + 3072 bytes image data)
                    record = f.read(CIFAR10Constants.RECORD_SIZE)
                    if len(record) != CIFAR10Constants.RECORD_SIZE:
                        break
                    
                    # First byte is the label
                    label = record[0]
                    
                    # Remaining 3072 bytes are the image data (32x32x3)
                    # Data format: R channel (1024 bytes), G channel (1024 bytes), B channel (1024 bytes)
                    image_data = np.frombuffer(record[1:], dtype=np.uint8)
                    
                    # Reshape to (3, 32, 32) - channels first format
                    image = image_data.reshape(3, 32, 32)
                    
                    self.data.append(image)
                    self.labels.append(label)
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"CIFAR-10 binary file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading CIFAR-10 binary file {file_path}: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Convert to tensor (already in channels-first format)
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class OptimizedCIFAR10CNN(nn.Module):
    """CNN architecture matching the C++ implementation exactly"""
    
    def __init__(self, enable_profiling=False):
        super(OptimizedCIFAR10CNN, self).__init__()
        
        # Block 1: 64 channels
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        
        # Block 2: 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        
        # Block 3: 256 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        
        # Block 4: 512 channels
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        
        # Fully connected layers
        # After 4 pooling operations: 32 -> 16 -> 8 -> 4 -> 2
        # So flattened size = 512 * 2 * 2 = 2048
        self.fc0 = nn.Linear(512 * 2 * 2, 512, bias=True)
        self.bn5 = nn.BatchNorm1d(512, eps=1e-5, momentum=0.1)
        self.fc1 = nn.Linear(512, CIFAR10Constants.NUM_CLASSES, bias=True)
        
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
        # Block 1
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = self.pool0(x)
        x = self.bn1(x)
        
        # Block 2
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        x = self.bn2(x)
        
        # Block 3
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(x)
        x = self.bn3(x)
        
        # Block 4
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool3(x)
        x = self.bn4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc0(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.fc1(x)
        
        return x
    
    def _forward_with_profiling(self, x):
        """Forward pass with layer-wise timing using high-precision perf_counter()"""
        layer_times = {}
        
        # Block 1
        start_time = time.perf_counter()
        x = F.relu(self.conv0(x))
        layer_times['conv0'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = F.relu(self.conv1(x))
        layer_times['conv1'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.pool0(x)
        layer_times['pool0'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.bn1(x)
        layer_times['bn1'] = (time.perf_counter() - start_time) * 1000
        
        # Block 2
        start_time = time.perf_counter()
        x = F.relu(self.conv2(x))
        layer_times['conv2'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = F.relu(self.conv3(x))
        layer_times['conv3'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.pool1(x)
        layer_times['pool1'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.bn2(x)
        layer_times['bn2'] = (time.perf_counter() - start_time) * 1000
        
        # Block 3
        start_time = time.perf_counter()
        x = F.relu(self.conv4(x))
        layer_times['conv4'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = F.relu(self.conv5(x))
        layer_times['conv5'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = F.relu(self.conv6(x))
        layer_times['conv6'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.pool2(x)
        layer_times['pool2'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.bn3(x)
        layer_times['bn3'] = (time.perf_counter() - start_time) * 1000
        
        # Block 4
        start_time = time.perf_counter()
        x = F.relu(self.conv7(x))
        layer_times['conv7'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = F.relu(self.conv8(x))
        layer_times['conv8'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = F.relu(self.conv9(x))
        layer_times['conv9'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.pool3(x)
        layer_times['pool3'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.bn4(x)
        layer_times['bn4'] = (time.perf_counter() - start_time) * 1000
        
        # Flatten
        start_time = time.perf_counter()
        x = x.view(x.size(0), -1)
        layer_times['flatten'] = (time.perf_counter() - start_time) * 1000
        
        # Fully connected layers
        start_time = time.perf_counter()
        x = self.fc0(x)
        layer_times['fc0'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.bn5(x)
        layer_times['bn5'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = F.relu(x)
        layer_times['relu3'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.fc1(x)
        layer_times['fc1'] = (time.perf_counter() - start_time) * 1000
        
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
        layer_order = ['conv0', 'conv1', 'pool0', 'bn1', 
                      'conv2', 'conv3', 'pool1', 'bn2',
                      'conv4', 'conv5', 'conv6', 'pool2', 'bn3',
                      'conv7', 'conv8', 'conv9', 'pool3', 'bn4',
                      'flatten', 'fc0', 'bn5', 'relu3', 'fc1']
        
        for layer_name in layer_order:
            if layer_name in self.layer_times:
                times = self.layer_times[layer_name]
                avg_time = sum(times) / len(times)
                print(f"{layer_name:<15} {avg_time:<15.3f} {avg_time:<15.3f}")
                total_time += avg_time
        
        print("-" * 60)
        print(f"{'TOTAL':<15} {total_time:<15.3f} {total_time:<15.3f}")
        print("=" * 60)
    
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
        
        if batch_idx % CIFAR10Constants.PROGRESS_PRINT_INTERVAL == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.6f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')
            
            # Print performance profile if profiling is enabled
            if hasattr(model, 'enable_profiling') and model.enable_profiling:
                model.print_performance_profile()
    
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
    torch.jit.enable_onednn_fusion(False)
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
    
    # Training data (data_batch_1.bin to data_batch_5.bin)
    train_files = []
    for i in range(1, 6):
        train_files.append(f'./data/cifar-10-batches-bin/data_batch_{i}.bin')
    
    # Test data
    test_files = ['./data/cifar-10-batches-bin/test_batch.bin']
    
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

    print(f"Total FLOPs: {FlopCountAnalysis(model, torch.randn(64, 3, 32, 32)).total():,}")
    # Print model summary
    print_model_summary(model, (CIFAR10Constants.BATCH_SIZE, 3, 
                               CIFAR10Constants.IMAGE_HEIGHT, CIFAR10Constants.IMAGE_WIDTH))
    
    # Create optimizer (Adam to match C++ implementation)
    optimizer = optim.Adam(model.parameters(), lr=CIFAR10Constants.LR_INITIAL, 
                          betas=(0.9, 0.999), eps=1e-8)
    
    # Create loss function (CrossEntropy with epsilon matching C++ implementation)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)  # No smoothing initially
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=CIFAR10Constants.LR_DECAY_INTERVAL, 
                                         gamma=CIFAR10Constants.LR_DECAY_FACTOR)
    
    print(f"\nStarting CIFAR-10 CNN training for {CIFAR10Constants.EPOCHS} epochs...")
    print(f"Batch size: {CIFAR10Constants.BATCH_SIZE}")
    print(f"Initial learning rate: {CIFAR10Constants.LR_INITIAL}")
    print(f"LR decay factor: {CIFAR10Constants.LR_DECAY_FACTOR} every {CIFAR10Constants.LR_DECAY_INTERVAL} epochs")
    
    # Training loop
    training_start_time = time.perf_counter()
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
    
    total_training_time = time.perf_counter() - training_start_time
    
    print("\n" + "="*60)
    print("CIFAR-10 CNN Tensor<float> model training completed successfully!")
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
