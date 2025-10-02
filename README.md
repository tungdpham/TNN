# Getting Started

## Getting the datasets
Download the MNIST dataset from kaggle (preferred) and put it in data/mnist. Name the training and validation data set train.csv and test.csv respectively.

# Build Instructions

## Dependencies
You should have these dependencies for the main programs installed before building:

```bash
# Install OpenMP (usually comes with GCC)
sudo apt install libomp-dev

# Install Intel TBB
sudo apt install libtbb-dev

# Install Intel MKL
# add oneAPI repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
# instal mkl
sudo apt install intel-oneapi-mkl-devel
# source env vars
source /opt/intel/oneapi/setvars.sh

# Install CUDA (follow NVIDIA's installation guide)

# For python scripts, install the dependencies from requirements.txt
pip install -r requirements.txt
```

## Prepraring Data
Download the dataset needed before running the examples.
the structure should look like this. Alternatively, you change the path to data in the examples' code.

- For MNIST dataset, download from [kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).
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
  uji/ (default extract)
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

# Build
cmake --build . -j$(nproc)
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_OPENMP` | OFF | Enable OpenMP parallel processing |
| `ENABLE_CUDA` | OFF | Enable CUDA GPU acceleration |
| `ENABLE_TBB` | ON | Enable Intel Threading Building Blocks |
| `ENABLE_DEBUG` | OFF | Enable debug build with AddressSanitizer |

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

For Windows with MSVC, you should see a Release folder inside bin/ if you are building optimized build, or Debug/ if you want to debug or profile the code.
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

## Other utilities for debugging/monitoring
### CPU Monitor uses:
If (profile is semi_async for example) run the following command to draw cpu load:

```bash
python3 plot_cpu_range.py --logs ./logs --out cpu_usage_sync_0_25.png --tmin 0 --tmax 25
```