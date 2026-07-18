!!! NOTICE: This repository now serve as an archive for the original TNN project. The project has been renamed to tunx. Please visit [here](https://github.com/tungdpham/tunx) to see the latest releases and updates.

# Getting Started

## Dependencies

You should have these dependencies for the main programs installed before building. Other dependencies and open-source frameworks are fetched directly from their repository for proper licensing and up-to-date builds.

### Install Required Packages

#### For Ubuntu
```bash
sudo apt install build-essential g++ make cmake git libtbb-dev wget libnuma-dev libibverbs-dev libfmt-dev
```

#### For Fedora
```bash
sudo dnf install cmake gcc-c++ make git tbb-devel wget numactl-devel libibverbs-devel fmt-devel
```

### Install Intel MKL

#### For Ubuntu
```bash
# 1. Add oneAPI repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update

# 2. Install MKL
sudo apt install intel-oneapi-mkl-devel intel-oneapi-dnnl-devel

# 3. Source environment variables
source /opt/intel/oneapi/setvars.sh
```

#### Fedora
```bash
# 1. Add oneAPI repository
wget -O- https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://yum.repos.intel.com/oneapi all main" | sudo tee /etc/yum.repos.d/oneAPI.repo
sudo dnf update

# 2. Install MKL
sudo dnf install intel-oneapi-mkl-devel intel-oneapi-dnnl-devel

# 3. Source environment variables
source /opt/intel/oneapi/setvars.sh
```

### Install CUDA Toolkit and cuDNN (9.17+)

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

# Enable DNNL
./build.sh --dnnl

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

Note: You can change the path to data in json config.

# Running the examples

There are two different ways to run the examples. For detailed instructions on how to run them see README in examples directory.

## Directly running them

For Linux with GCC
```bash
cd build/

# To run any of them
./build/bin/{executable_name}

# For example (for single-model trainer):
./build/bin/trainer --config ./configs/default_config.json
```

For Windows with MSVC, you should see a Release/Debug folder inside bin/. if you are building optimized build, or Debug/ if you want to debug or profile the code.
```bash
# Example:
./build/bin/Release/mnist_cnn_trainer.exe
```
