# Getting Started

## Dependencies
You should have these dependencies for the main programs installed before building. Other dependencies and open-source frameworks are fetched directly from their repository for proper licensing and up-to-date builds.

### Install Required Packages
```bash
sudo apt install build-essential g++ make cmake git libtbb-dev wget libnuma-dev
```

### Install Intel MKL (Recommended for CPU usage)
```bash
# 1. Add oneAPI repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
# 2. Install MKL
sudo apt install intel-oneapi-mkl-devel
# 3. Source environment variables
source /opt/intel/oneapi/setvars.sh
```

### Install CUDA and cuDNN
For installing these two dependencies, you need to follow the guide from NVIDIA page.

## Build Instructions
### Option 1: Using the build script (Recommended)
```bash
# Add executable permission to build script
chmod +x ./build.sh

# Simple build with default settings
./build.sh

# Clean build (removes previous build artifacts)
./build.sh --clean

# Debug build with sanitizers
./build.sh --debug

# Enable Intel MKL
./build.sh --mkl

# Enable CUDA
./build.sh --cuda

# Verbose build output
./build.sh --verbose
```

### Option 2: Manual CMake commands
```bash
# Create and enter build directory
mkdir build && cd build

# Configure (basic build)
cmake ..

# Configure with options
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_MKL=ON \ 
         -DENABLE_TBB=OFF \

# Build with maximum number of cores
cmake --build . -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `Enable_MKL` | OFF | ENABLE Intel Math Kernel Library |
| `ENABLE_TBB` | ON | Enable Intel Threading Building Blocks |
| `ENABLE_DEBUG` | OFF | Enable debug build with AddressSanitizer |
| `ENABLE_CUDA` | OFF | Enable CUDA support for GPUs |

## Prepraring Data
Download the dataset needed before running the examples.

- For MNIST dataset, download from [kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).
- For CIFAR10 and CIFAR100, download from
[here](https://www.cs.toronto.edu/~kriz/cifar.html)
- For UJI and UTS indoor positioning dataset, download from their paper.

The structure of your data directory should look like this.

```
data/
  mnist/
    train.csv
    test.csv
  cifar-10-batches-bin/ (default extract)
  cifar-100-binary/ (default extract)
  uji/ (default extract)
  uts/
    train.csv
    test.csv
```

Alternatively, you change the path to data in the examples' code.

# Running the examples
There are two different ways to run the examples. For detailed instructions on how to run them see README in examples directory.

## Directly running them
There are several preconfigured trainers for MNIST, CIFAR10, CIFAR100, and UJI IPS datasets. You should see them in bin/ after building successfully. 

For Linux with GCC
```bash
# To run any of them
./bin/{executable_name}

# Example: 
./bin/mnist_cnn_trainer
```

For Windows with MSVC, you should see a Release/Debug folder inside bin/. if you are building optimized build, or Debug/ if you want to debug or profile the code.
```bash
# Example:
./bin/Release/mnist_cnn_trainer.exe
```