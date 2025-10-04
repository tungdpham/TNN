# Getting Started

## Build Instructions

## Dependencies
You should have these dependencies for the main programs installed before building. Other dependencies and open-source frameworks are fetched directly from their repository for proper licensing and up-to-date builds.

```bash
# Install Intel TBB (Required)
sudo apt install libtbb-dev

# Install Intel MKL (Recommended)
# 1. Add oneAPI repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
# 2. Install MKL
sudo apt install intel-oneapi-mkl-devel
# 3. Source environment variables
source /opt/intel/oneapi/setvars.sh

# For Python scripts (Optional, not related to our library), install the dependencies from requirements.txt
pip install -r requirements.txt
```

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

# Enable OpenMP support
./build.sh --openmp

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

# Build with maximum number of cores
cmake --build . -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `Enable_MKL` | OFF | ENABLE Intel Math Kernel Library |
| `ENABLE_TBB` | ON | Enable Intel Threading Building Blocks |
| `ENABLE_DEBUG` | OFF | Enable debug build with AddressSanitizer |

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
There are two different ways to run the examples

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

## Containerized run

```bash
# First build the docker images
docker compose build

# Run the profile you want
## Using the shell script (Example)
./docker_start.sh -p semi-async

## Manually
## For single model
docker compose --profile single-model up -d

## For sync
docker compose --profile sync up -d

## For semi async
docker compose --profile semi-async up -d
```