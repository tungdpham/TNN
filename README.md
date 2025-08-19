# Getting Started

## Getting the datasets
Download the MNIST dataset from kaggle (preferred) and put it in data/mnist. Name the training and validation data set train.csv and test.csv respectively.

# Build Instructions

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

## Available Executables

After building, you'll find the following executables in the `build/` directory:

- `mnist_cnn_trainer` - MNIST CNN neural network trainer
- `cifar10_cnn_trainer` - CIFAR-10 CNN trainer  
- `cifar100_cnn_trainer` - CIFAR-100 CNN trainer
- `uji_ips_trainer` - UJI indoor positioning system trainer
- `mnist_cnn_test` - MNIST CNN test program
- `pipeline_test` - Pipeline functionality test
- `network_worker` - Network pipeline stage worker
- `distributed_pipeline_docker` - Distributed pipeline example

## Running Programs

```bash
# Run from build directory
cd build
./mnist_cnn_trainer

# Or run from project root
./build/mnist_cnn_trainer
```

## Running Tests

```bash
cd build
make run_tests
```

## Clean Build

To completely clean your build:
```bash
rm -rf build/
./build.sh
```

## IDE Integration

### VS Code
1. Install the CMake Tools extension
2. Open the project folder
3. Press `Ctrl+Shift+P` and run "CMake: Configure"
4. Press `F7` to build or use the status bar

### CLion
Just open the project folder - CLion will automatically detect the CMakeLists.txt file.

## Troubleshooting

### CUDA Issues
If you have CUDA enabled but get errors:
- Ensure CUDA toolkit is installed
- Check that your GPU supports the architecture (currently set to sm_89)
- Adjust `CMAKE_CUDA_FLAGS` in CMakeLists.txt if needed

### Missing Dependencies
```bash
# Install OpenMP (usually comes with GCC)
sudo apt install libomp-dev

# Install Intel TBB
sudo apt install libtbb-dev

# Install CUDA (follow NVIDIA's installation guide)
```
