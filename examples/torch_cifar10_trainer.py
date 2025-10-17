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

# CIFAR-10 Constants (matching C++ implementation exactly)
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
    BATCH_SIZE = 64
    LR_DECAY_INTERVAL = 3  # Matching C++ version (was 5)
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
        
        # Profiling setup
        self.enable_profiling = enable_profiling
        self.layer_times = {}
        self.profile_count = 0
        
        # Build the model using Sequential for optimized forward pass
        # This matches the C++ architecture exactly: Conv→ReLU→Conv→ReLU→Pool→BatchNorm pattern
        
        # Block 1: 64 channels (conv0→relu0→conv1→relu1→pool0→bn0)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),      # conv0
            nn.ReLU(inplace=True),                                                # relu0
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),    # conv1  
            nn.ReLU(inplace=True),                                                # relu1
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                    # pool0
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)                          # bn0
        )
        
        # Block 2: 128 channels (conv2→relu2→conv3→relu3→pool1→bn1) 
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),   # conv2
            nn.ReLU(inplace=True),                                                # relu2
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),  # conv3
            nn.ReLU(inplace=True),                                                # relu3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                    # pool1
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)                         # bn1
        )
        
        # Block 3: 256 channels (conv4→relu5→conv5→relu6→conv6→relu6→pool2→bn6)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),  # conv4
            nn.ReLU(inplace=True),                                                # relu5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),  # conv5
            nn.ReLU(inplace=True),                                                # relu6
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),  # conv6
            nn.ReLU(inplace=True),                                                # relu6 (reused name in C++)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                    # pool2
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)                         # bn6
        )
        
        # Block 4: 512 channels (conv7→relu7→conv8→relu8→conv9→relu9→pool3→bn10)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),  # conv7
            nn.ReLU(inplace=True),                                                # relu7
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),  # conv8
            nn.ReLU(inplace=True),                                                # relu8
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),  # conv9
            nn.ReLU(inplace=True),                                                # relu9
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                    # pool3
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)                         # bn10
        )
        
        # Fully connected layers (flatten→fc0→relu10→fc1)
        # After 4 pooling operations: 32 -> 16 -> 8 -> 4 -> 2
        # So flattened size = 512 * 2 * 2 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),                                                         # flatten
            nn.Linear(512 * 2 * 2, 512, bias=True),                             # fc0
            nn.ReLU(inplace=True),                                                # relu10
            nn.Linear(512, CIFAR10Constants.NUM_CLASSES, bias=True)             # fc1
        )
        
        # For profiling, we need to access individual layers
        self.conv0 = self.block1[0]  # Conv2d
        self.relu0 = self.block1[1]  # ReLU
        self.conv1 = self.block1[2]  # Conv2d  
        self.relu1 = self.block1[3]  # ReLU
        self.pool0 = self.block1[4]  # MaxPool2d
        self.bn0 = self.block1[5]    # BatchNorm2d
        
        self.conv2 = self.block2[0]  # Conv2d
        self.relu2 = self.block2[1]  # ReLU
        self.conv3 = self.block2[2]  # Conv2d
        self.relu3 = self.block2[3]  # ReLU
        self.pool1 = self.block2[4]  # MaxPool2d
        self.bn1 = self.block2[5]    # BatchNorm2d
        
        self.conv4 = self.block3[0]  # Conv2d
        self.relu5 = self.block3[1]  # ReLU
        self.conv5 = self.block3[2]  # Conv2d
        self.relu6a = self.block3[3] # ReLU
        self.conv6 = self.block3[4]  # Conv2d
        self.relu6b = self.block3[5] # ReLU (second relu6)
        self.pool2 = self.block3[6]  # MaxPool2d
        self.bn6 = self.block3[7]    # BatchNorm2d
        
        self.conv7 = self.block4[0]  # Conv2d
        self.relu7 = self.block4[1]  # ReLU
        self.conv8 = self.block4[2]  # Conv2d
        self.relu8 = self.block4[3]  # ReLU
        self.conv9 = self.block4[4]  # Conv2d
        self.relu9 = self.block4[5]  # ReLU
        self.pool3 = self.block4[6]  # MaxPool2d
        self.bn10 = self.block4[7]   # BatchNorm2d
        
        self.flatten = self.classifier[0]  # Flatten
        self.fc0 = self.classifier[1]      # Linear
        self.relu10 = self.classifier[2]   # ReLU
        self.fc1 = self.classifier[3]      # Linear
        
    def forward(self, x):
        if self.enable_profiling:
            return self._forward_with_profiling(x)
        else:
            # Optimized forward pass using Sequential blocks for better performance
            x = self.block1(x)  # conv0→relu0→conv1→relu1→pool0→bn0
            x = self.block2(x)  # conv2→relu2→conv3→relu3→pool1→bn1
            x = self.block3(x)  # conv4→relu5→conv5→relu6→conv6→relu6→pool2→bn6
            x = self.block4(x)  # conv7→relu7→conv8→relu8→conv9→relu9→pool3→bn10
            x = self.classifier(x)  # flatten→fc0→relu10→fc1
            return x
    
    def _forward_normal(self, x):
        # This method is no longer used but kept for compatibility
        return self.forward(x)
    
    def _forward_with_profiling(self, x):
        """Forward pass with layer-wise timing using high-precision perf_counter()"""
        layer_times = {}
        
        # Block 1: conv0→relu0→conv1→relu1→pool0→bn0
        start_time = time.perf_counter()
        x = self.conv0(x)
        layer_times['conv0'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu0(x)
        layer_times['relu0'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.conv1(x)
        layer_times['conv1'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu1(x)
        layer_times['relu1'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.pool0(x)
        layer_times['pool0'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.bn0(x)
        layer_times['bn0'] = (time.perf_counter() - start_time) * 1000
        
        # Block 2: conv2→relu2→conv3→relu3→pool1→bn1
        start_time = time.perf_counter()
        x = self.conv2(x)
        layer_times['conv2'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu2(x)
        layer_times['relu2'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.conv3(x)
        layer_times['conv3'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu3(x)
        layer_times['relu3'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.pool1(x)
        layer_times['pool1'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.bn1(x)
        layer_times['bn1'] = (time.perf_counter() - start_time) * 1000
        
        # Block 3: conv4→relu5→conv5→relu6→conv6→relu6→pool2→bn6
        start_time = time.perf_counter()
        x = self.conv4(x)
        layer_times['conv4'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu5(x)
        layer_times['relu5'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.conv5(x)
        layer_times['conv5'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu6a(x)
        layer_times['relu6'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.conv6(x)
        layer_times['conv6'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu6b(x)  # Second relu6 in C++ 
        layer_times['relu6b'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.pool2(x)
        layer_times['pool2'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.bn6(x)
        layer_times['bn6'] = (time.perf_counter() - start_time) * 1000
        
        # Block 4: conv7→relu7→conv8→relu8→conv9→relu9→pool3→bn10
        start_time = time.perf_counter()
        x = self.conv7(x)
        layer_times['conv7'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu7(x)
        layer_times['relu7'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.conv8(x)
        layer_times['conv8'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu8(x)
        layer_times['relu8'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.conv9(x)
        layer_times['conv9'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu9(x)
        layer_times['relu9'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.pool3(x)
        layer_times['pool3'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.bn10(x)
        layer_times['bn10'] = (time.perf_counter() - start_time) * 1000
        
        # Classifier: flatten→fc0→relu10→fc1
        start_time = time.perf_counter()
        x = self.flatten(x)
        layer_times['flatten'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.fc0(x)
        layer_times['fc0'] = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        x = self.relu10(x)
        layer_times['relu10'] = (time.perf_counter() - start_time) * 1000
        
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
        # Layer order matching C++ implementation exactly
        layer_order = ['conv0', 'relu0', 'conv1', 'relu1', 'pool0', 'bn0',
                      'conv2', 'relu2', 'conv3', 'relu3', 'pool1', 'bn1',
                      'conv4', 'relu5', 'conv5', 'relu6', 'conv6', 'relu6b', 'pool2', 'bn6',
                      'conv7', 'relu7', 'conv8', 'relu8', 'conv9', 'relu9', 'pool3', 'bn10',
                      'flatten', 'fc0', 'relu10', 'fc1']
        
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
    print("=" * 70)
    
    total_params = 0
    trainable_params = 0
    
    print(f"{'Block/Layer':<20} {'Module':<35} {'Params':>8}")
    print("-" * 70)
    
    # Print block-wise summary for better readability
    blocks = [
        ('Block1', model.block1),
        ('Block2', model.block2), 
        ('Block3', model.block3),
        ('Block4', model.block4),
        ('Classifier', model.classifier)
    ]
    
    for block_name, block in blocks:
        print(f"{block_name}:")
        for name, module in block.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if params > 0:
                module_str = str(module).replace('\n', ' ')[:30] + ('...' if len(str(module)) > 30 else '')
                print(f"  {name:<18} {module_str:<33} {params:>8,}")
                total_params += params
                trainable_params += trainable
        print()
    
    print("=" * 70)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)
    
    # Test forward pass with dummy input
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_shape)
        try:
            output = model(dummy_input)
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output.shape}")
            print("Forward pass successful!")
        except Exception as e:
            print(f"Error in forward pass: {e}")
    print("=" * 70)

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
