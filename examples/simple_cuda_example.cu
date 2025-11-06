#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef USE_CUDA
#include "cuda/error_handler.hpp"
#include "device/device_manager.hpp"
#include <cuda_runtime.h>

// Simple vector addition kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// Vector multiplication kernel
__global__ void vectorMul(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
}

// Vector scaling kernel
__global__ void vectorScale(const float *a, float *b, float scale, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    b[idx] = a[idx] * scale;
  }
}

// Mathematical operations kernel
__global__ void mathOps(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = input[idx];
    // Perform intensive mathematical operations
    x = sinf(x) + cosf(x) + expf(x * 0.1f) + logf(fabsf(x) + 1.0f);
    output[idx] = x;
  }
}

// Simple matrix multiplication kernel (naive)
__global__ void matrixMul(const float *a, const float *b, float *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
      sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
  }
}

// Optimized matrix multiplication with shared memory
__global__ void matrixMulShared(const float *a, const float *b, float *c, int n) {
  const int TILE_SIZE = 16;
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
    // Load tiles into shared memory
    if (row < n && t * TILE_SIZE + threadIdx.x < n) {
      As[threadIdx.y][threadIdx.x] = a[row * n + t * TILE_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (col < n && t * TILE_SIZE + threadIdx.y < n) {
      Bs[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * n + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute partial dot product
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = sum;
  }
}

// Memory copy kernel for bandwidth testing
__global__ void memcopy(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx];
  }
}

// Reduction kernel
__global__ void reduce(const float *input, float *output, int n) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

class SimpleCUDABenchmark {
private:
  int device_id_;
  cudaStream_t stream_;
  const tdevice::Device *device_; // Reference to our device management system

public:
  SimpleCUDABenchmark(int device_id = 0) : device_id_(device_id), device_(nullptr) {
    // Initialize device manager and get the device
    tdevice::DeviceManager &manager = tdevice::DeviceManager::getInstance();

    // Try to find a GPU device with the specified CUDA device ID
    std::vector<int> device_ids = manager.getAvailableDeviceIDs();
    for (int id : device_ids) {
      const tdevice::Device &dev = manager.getDevice(id);
      if (dev.getDeviceType() == tdevice::DeviceType::GPU) {
        // For simplicity, we'll use the first available GPU
        // In a real implementation, you might want to match the CUDA device ID
        device_ = &dev;
        break;
      }
    }

    if (!device_) {
      throw std::runtime_error("No GPU device found in device manager");
    }

    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaStreamCreate(&stream_));

    std::cout << "Using device: " << device_->getName() << " (ID: " << device_->getID() << ")"
              << std::endl;
  }

  ~SimpleCUDABenchmark() { cudaStreamDestroy(stream_); }

  void printDeviceInfo() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));

    std::cout << "=== GPU Device Information ===" << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << std::endl;
  }

  void benchmarkVectorOperations(int size, int iterations = 1000) {
    std::cout << "=== Vector Operations Benchmark (Using Device Manager) ===" << std::endl;
    std::cout << "Vector size: " << size << " elements" << std::endl;
    std::cout << "Device: " << device_->getName() << std::endl;
    std::cout << "Device Memory - Total: " << device_->getTotalMemory() / (1024 * 1024) << " MB, "
              << "Available: " << device_->getAvailableMemory() / (1024 * 1024) << " MB"
              << std::endl;

    size_t bytes = size * sizeof(float);

    // Allocate host memory
    std::vector<float> h_a(size, 1.5f);
    std::vector<float> h_b(size, 2.5f);
    std::vector<float> h_c(size);

    // Allocate device memory using our device manager
    float *d_a, *d_b, *d_c;
    try {
      d_a = static_cast<float *>(device_->allocateMemory(bytes));
      d_b = static_cast<float *>(device_->allocateMemory(bytes));
      d_c = static_cast<float *>(device_->allocateMemory(bytes));

      std::cout << "Successfully allocated device memory using DeviceManager" << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Failed to allocate memory using DeviceManager: " << e.what() << std::endl;
      return;
    }

    // Copy data to device using our device manager
    try {
      device_->copyToDevice(d_a, h_a.data(), bytes);
      device_->copyToDevice(d_b, h_b.data(), bytes);
      std::cout << "Successfully copied data to device using DeviceManager" << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Failed to copy data to device: " << e.what() << std::endl;
      device_->deallocateMemory(d_a);
      device_->deallocateMemory(d_b);
      device_->deallocateMemory(d_c);
      return;
    }

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Benchmark vector addition
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      vectorAdd<<<gridSize, blockSize, 0, stream_>>>(d_a, d_b, d_c, size);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    double bandwidth = (3.0 * bytes * iterations) / (duration.count() * 1e-3); // GB/s

    std::cout << "Vector Addition:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_ms << " ms"
              << std::endl;
    std::cout << "  Avg per kernel: " << std::fixed << std::setprecision(3) << time_ms / iterations
              << " ms" << std::endl;
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
              << std::endl;

    // Benchmark vector multiplication
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      vectorMul<<<gridSize, blockSize, 0, stream_>>>(d_a, d_b, d_c, size);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    time_ms = duration.count() / 1000.0;
    bandwidth = (3.0 * bytes * iterations) / (duration.count() * 1e-3); // GB/s

    std::cout << "Vector Multiplication:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_ms << " ms"
              << std::endl;
    std::cout << "  Avg per kernel: " << std::fixed << std::setprecision(3) << time_ms / iterations
              << " ms" << std::endl;
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
              << std::endl;

    // Verify results using our device manager
    try {
      device_->copyToHost(h_c.data(), d_c, bytes);
      std::cout << "Successfully copied results back to host using DeviceManager" << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Failed to copy results back to host: " << e.what() << std::endl;
    }

    bool correct = true;
    for (int i = 0; i < std::min(10, size); i++) {
      if (std::abs(h_c[i] - (h_a[i] * h_b[i])) > 1e-5) {
        correct = false;
        break;
      }
    }
    std::cout << "Results: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;

    // Cleanup using our device manager
    try {
      device_->deallocateMemory(d_a);
      device_->deallocateMemory(d_b);
      device_->deallocateMemory(d_c);
      std::cout << "Successfully deallocated device memory using DeviceManager" << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Failed to deallocate memory: " << e.what() << std::endl;
    }
    std::cout << std::endl;
  }

  void benchmarkMathOperations(int size, int iterations = 100) {
    std::cout << "=== Mathematical Operations Benchmark (Using Device Manager) ===" << std::endl;
    std::cout << "Array size: " << size << " elements" << std::endl;
    std::cout << "Device: " << device_->getName() << std::endl;

    size_t bytes = size * sizeof(float);

    // Allocate and initialize host memory
    std::vector<float> h_input(size);
    for (int i = 0; i < size; i++) {
      h_input[i] = static_cast<float>(i % 1000) / 1000.0f;
    }

    // Allocate device memory using our device manager
    float *d_input, *d_output;
    try {
      d_input = static_cast<float *>(device_->allocateMemory(bytes));
      d_output = static_cast<float *>(device_->allocateMemory(bytes));
    } catch (const std::exception &e) {
      std::cerr << "Failed to allocate memory: " << e.what() << std::endl;
      return;
    }

    // Copy data to device using our device manager
    try {
      device_->copyToDevice(d_input, h_input.data(), bytes);
    } catch (const std::exception &e) {
      std::cerr << "Failed to copy data to device: " << e.what() << std::endl;
      device_->deallocateMemory(d_input);
      device_->deallocateMemory(d_output);
      return;
    }

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Benchmark mathematical operations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      mathOps<<<gridSize, blockSize, 0, stream_>>>(d_input, d_output, size);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    double throughput = (size * iterations) / (duration.count() * 1e-3); // ops/second

    std::cout << "Mathematical Operations:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_ms << " ms"
              << std::endl;
    std::cout << "  Avg per kernel: " << std::fixed << std::setprecision(3) << time_ms / iterations
              << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << throughput / 1e9
              << " GOp/s" << std::endl;

    // Cleanup using our device manager
    try {
      device_->deallocateMemory(d_input);
      device_->deallocateMemory(d_output);
    } catch (const std::exception &e) {
      std::cerr << "Failed to deallocate memory: " << e.what() << std::endl;
    }
    std::cout << std::endl;
  }

  void benchmarkMatrixMultiplication(int n, int iterations = 5) {
    std::cout << "=== Matrix Multiplication Benchmark (Using Device Manager) ===" << std::endl;
    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Device: " << device_->getName() << std::endl;

    size_t bytes = n * n * sizeof(float);

    // Allocate host memory
    std::vector<float> h_a(n * n, 1.0f);
    std::vector<float> h_b(n * n, 2.0f);
    std::vector<float> h_c(n * n);

    // Allocate device memory using our device manager
    float *d_a, *d_b, *d_c;
    try {
      d_a = static_cast<float *>(device_->allocateMemory(bytes));
      d_b = static_cast<float *>(device_->allocateMemory(bytes));
      d_c = static_cast<float *>(device_->allocateMemory(bytes));
    } catch (const std::exception &e) {
      std::cerr << "Failed to allocate memory: " << e.what() << std::endl;
      return;
    }

    // Copy data to device using our device manager
    try {
      device_->copyToDevice(d_a, h_a.data(), bytes);
      device_->copyToDevice(d_b, h_b.data(), bytes);
    } catch (const std::exception &e) {
      std::cerr << "Failed to copy data to device: " << e.what() << std::endl;
      device_->deallocateMemory(d_a);
      device_->deallocateMemory(d_b);
      device_->deallocateMemory(d_c);
      return;
    }

    // Test naive implementation
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      matrixMul<<<gridDim, blockDim, 0, stream_>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    double gflops = (2.0 * n * n * n * iterations) / (duration.count() * 1e-3);

    std::cout << "Naive Implementation:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_ms << " ms"
              << std::endl;
    std::cout << "  Avg per kernel: " << std::fixed << std::setprecision(3) << time_ms / iterations
              << " ms" << std::endl;
    std::cout << "  Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s"
              << std::endl;

    // Test shared memory implementation
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      matrixMulShared<<<gridDim, blockDim, 0, stream_>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    time_ms = duration.count() / 1000.0;
    gflops = (2.0 * n * n * n * iterations) / (duration.count() * 1e-3);

    std::cout << "Shared Memory Implementation:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_ms << " ms"
              << std::endl;
    std::cout << "  Avg per kernel: " << std::fixed << std::setprecision(3) << time_ms / iterations
              << " ms" << std::endl;
    std::cout << "  Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s"
              << std::endl;

    // Verify results using our device manager
    try {
      device_->copyToHost(h_c.data(), d_c, bytes);
    } catch (const std::exception &e) {
      std::cerr << "Failed to copy results back to host: " << e.what() << std::endl;
    }

    bool correct = true;
    for (int i = 0; i < std::min(10, n * n); i++) {
      if (std::abs(h_c[i] - (2.0f * n)) > 1e-3) {
        correct = false;
        break;
      }
    }
    std::cout << "Results: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;

    // Cleanup using our device manager
    try {
      device_->deallocateMemory(d_a);
      device_->deallocateMemory(d_b);
      device_->deallocateMemory(d_c);
    } catch (const std::exception &e) {
      std::cerr << "Failed to deallocate memory: " << e.what() << std::endl;
    }
    std::cout << std::endl;
  }

  void benchmarkMemoryBandwidth(int size, int iterations = 100) {
    std::cout << "=== Memory Bandwidth Benchmark (Using Device Manager) ===" << std::endl;
    std::cout << "Array size: " << size << " elements" << std::endl;
    std::cout << "Device: " << device_->getName() << std::endl;

    size_t bytes = size * sizeof(float);

    // Allocate device memory using our device manager
    float *d_input, *d_output;
    try {
      d_input = static_cast<float *>(device_->allocateMemory(bytes));
      d_output = static_cast<float *>(device_->allocateMemory(bytes));
    } catch (const std::exception &e) {
      std::cerr << "Failed to allocate memory: " << e.what() << std::endl;
      return;
    }

    // Initialize device memory (we'll use CUDA directly for memset since it's not in our Device
    // interface)
    CUDA_CHECK(cudaMemset(d_input, 0x42, bytes));

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Benchmark memory copy
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      memcopy<<<gridSize, blockSize, 0, stream_>>>(d_input, d_output, size);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    double bandwidth =
        (2.0 * bytes * iterations) / (duration.count() * 1e-3); // GB/s (read + write)

    std::cout << "Kernel Memory Copy:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << time_ms << " ms"
              << std::endl;
    std::cout << "  Avg per kernel: " << std::fixed << std::setprecision(3) << time_ms / iterations
              << " ms" << std::endl;
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
              << std::endl;

    // Cleanup using our device manager
    try {
      device_->deallocateMemory(d_input);
      device_->deallocateMemory(d_output);
    } catch (const std::exception &e) {
      std::cerr << "Failed to deallocate memory: " << e.what() << std::endl;
    }
    std::cout << std::endl;
  }

  void runAllBenchmarks() {
    printDeviceInfo();

    // Test different vector sizes
    std::vector<int> vector_sizes = {1024, 1024 * 1024, 16 * 1024 * 1024};
    for (int size : vector_sizes) {
      benchmarkVectorOperations(size);
      benchmarkMathOperations(size);
      benchmarkMemoryBandwidth(size);
    }

    // Test different matrix sizes
    std::vector<int> matrix_sizes = {64, 128, 256, 512};
    for (int size : matrix_sizes) {
      benchmarkMatrixMultiplication(size);
    }
  }
};

#endif // USE_CUDA

int main() {
  std::cout << "=== Simple CUDA Performance Test with Device Manager ===" << std::endl;

#ifdef USE_CUDA
  // Initialize the device manager first
  std::cout << "Initializing device manager..." << std::endl;
  try {
    tdevice::initializeDefaultDevices();
  } catch (const std::exception &e) {
    std::cerr << "Failed to initialize device manager: " << e.what() << std::endl;
    return 1;
  }

  tdevice::DeviceManager &manager = tdevice::DeviceManager::getInstance();
  std::vector<int> device_ids = manager.getAvailableDeviceIDs();

  std::cout << "Found " << device_ids.size() << " device(s) in device manager" << std::endl;

  // List all devices
  for (int id : device_ids) {
    const tdevice::Device &device = manager.getDevice(id);
    std::cout << "  Device " << id << ": " << device.getName()
              << " (Type: " << (device.getDeviceType() == tdevice::DeviceType::CPU ? "CPU" : "GPU")
              << ")"
              << " - Total Memory: " << device.getTotalMemory() / (1024 * 1024) << " MB"
              << std::endl;
  }

  // Find GPU devices and test them
  bool found_gpu = false;
  for (int id : device_ids) {
    const tdevice::Device &device = manager.getDevice(id);
    if (device.getDeviceType() == tdevice::DeviceType::GPU) {
      std::cout << "\n" << std::string(60, '=') << std::endl;
      std::cout << "Testing GPU Device " << id << " (" << device.getName() << ")" << std::endl;
      std::cout << std::string(60, '=') << std::endl;

      try {
        // Note: We're still using CUDA device ID 0 for the actual CUDA context
        // In a more sophisticated implementation, you'd map device manager IDs to CUDA device IDs
        SimpleCUDABenchmark benchmark(0);
        benchmark.runAllBenchmarks();
        found_gpu = true;
      } catch (const std::exception &e) {
        std::cerr << "Error testing device " << id << ": " << e.what() << std::endl;
      }
      break; // Test only the first GPU for now
    }
  }

  if (!found_gpu) {
    std::cout << "No GPU devices found in device manager." << std::endl;

    // Fallback to direct CUDA detection
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      std::cout << "No CUDA devices found or CUDA not available." << std::endl;
      return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s) directly, but not in device manager"
              << std::endl;
    std::cout << "This suggests the device manager initialization might need debugging."
              << std::endl;
    return 1;
  }

  std::cout << "\n=== All CUDA performance tests with Device Manager completed ===" << std::endl;

#else
  std::cout << "CUDA support not compiled. Please rebuild with -DENABLE_CUDA=ON" << std::endl;
  return 1;
#endif

  return 0;
}