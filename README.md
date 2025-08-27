# Getting Started

## Getting the datasets
Download the MNIST dataset from kaggle (preferred) and put it in data/mnist. Name the training and validation data set train.csv and test.csv respectively.

# Build Instructions

## Dependencies
You should have these dependencies installed before building:

```bash
# Install OpenMP (usually comes with GCC)
sudo apt install libomp-dev

# Install Intel TBB
sudo apt install libtbb-dev

# Install CUDA (follow NVIDIA's installation guide)
```

## Prepraring Data
Download the dataset needed before running the examples.
the structure should look like this. Alternatively, you change the path to data in the example code.

- For MNIST dataset, use kaggle.
- For CIFAR10 and CIFAR100, download from
[here](https://www.cs.toronto.edu/~kriz/cifar.html)
- For UJI and UTS indoor positioning dataset, download from their paper.

```
data/
  mnist/
    train.csv
    test.csv
  cifar-10-batches-bin/ (default extract)
  cifar-100-binary/ (default extract)
  uji/
    train.csv
    validation.csv
  uts/
    train.csv
    test.csv
```
## Quick Start

### Option 1: Using the build script (Recommended)
```bash
# Simple build with default settings
./build.sh

# Clean build (removes previous build artifacts)
./build.sh --clean

# Debug build with sanitizers
./build.sh --debug

# Enable CUDA support
./build.sh --cuda

# Enable Intel TBB support
./build.sh --tbb

# Disable OpenMP (enabled by default)
./build.sh --no-openmp

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
         -DENABLE_OPENMP=ON \
         -DENABLE_CUDA=OFF \
         -DENABLE_TBB=OFF

# Build
cmake --build . -j$(nproc)
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_OPENMP` | ON | Enable OpenMP parallel processing |
| `ENABLE_CUDA` | OFF | Enable CUDA GPU acceleration |
| `ENABLE_TBB` | OFF | Enable Intel Threading Building Blocks |
| `ENABLE_DEBUG` | OFF | Enable debug build with AddressSanitizer |

# Running the examples
There are two different ways to run the examples

## Directly running them
There are several preconfigured trainers for MNIST, CIFAR10, CIFAR100, and UJI IPS datasets. You should see them in bin/ after building successfully. 

```bash
# To run any of them
./bin/{executable_name}

# Example: 
./bin/mnist_cnn_trainer
```

## Containerized run

```bash
# First build the docker images
docker compose build

# Run the profile you want

## For single model
docker compose --profile single-model up -d

## For sync
docker compose --profile sync up -d

## For semi async
docker compose --profile semi_async up -d
```
