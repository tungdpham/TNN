#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef USE_CUDA
#include "cuda/error_handler.hpp"
#include "device/device.hpp"
#include "device/device_manager.hpp"
#include <cuda_runtime.h>

using namespace tnn;
using namespace std;

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
  const Device *device_; // Reference to our device management system

public:
  SimpleCUDABenchmark(int device_id = 0) : device_id_(device_id), device_(nullptr) {
    // Initialize device manager and get the device
    DeviceManager &manager = DeviceManager::getInstance();

    // Try to find a GPU device with the specified CUDA device ID
    vector<string> device_ids = manager.getAvailableDeviceIDs();
    for (const string &id : device_ids) {
      const Device &dev = manager.getDevice(id);
      if (dev.getDeviceType() == DeviceType::GPU) {
        // For simplicity, we'll use the first available GPU
        // In a real implementation, you might want to match the CUDA device ID
        device_ = &dev;
        break;
      }
    }

    if (!device_) {
      throw runtime_error("No GPU device found in device manager");
    }

    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaStreamCreate(&stream_));

    cout << "Using device: " << device_->getName() << " (ID: " << device_->getID() << ")" << endl;
  }

  ~SimpleCUDABenchmark() { cudaStreamDestroy(stream_); }

  void printDeviceInfo() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));

    cout << "=== GPU Device Information ===" << endl;
    cout << "Device Name: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
    cout << "Multiprocessors: " << prop.multiProcessorCount << endl;
    cout << "Warp Size: " << prop.warpSize << endl;
    cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << endl;
    cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << endl;
    cout << endl;
  }

  void benchmarkVectorOperations(int size, int iterations = 1000) {
    cout << "=== Vector Operations Benchmark (Using Device Manager) ===" << endl;
    cout << "Vector size: " << size << " elements" << endl;
    cout << "Device: " << device_->getName() << endl;
    cout << "Device Memory - Total: " << device_->getTotalMemory() / (1024 * 1024) << " MB, "
         << "Available: " << device_->getAvailableMemory() / (1024 * 1024) << " MB" << endl;

    size_t bytes = size * sizeof(float);

    // Allocate host memory
    vector<float> h_a(size, 1.5f);
    vector<float> h_b(size, 2.5f);
    vector<float> h_c(size);

    // Allocate device memory using our device manager
    float *d_a, *d_b, *d_c;
    try {
      d_a = static_cast<float *>(device_->allocateMemory(bytes));
      d_b = static_cast<float *>(device_->allocateMemory(bytes));
      d_c = static_cast<float *>(device_->allocateMemory(bytes));

      cout << "Successfully allocated device memory using DeviceManager" << endl;
    } catch (const exception &e) {
      cerr << "Failed to allocate memory using DeviceManager: " << e.what() << endl;
      return;
    }

    // Copy data to device using our device manager
    try {
      device_->copyToDevice(d_a, h_a.data(), bytes);
      device_->copyToDevice(d_b, h_b.data(), bytes);
      cout << "Successfully copied data to device using DeviceManager" << endl;
    } catch (const exception &e) {
      cerr << "Failed to copy data to device: " << e.what() << endl;
      device_->deallocateMemory(d_a);
      device_->deallocateMemory(d_b);
      device_->deallocateMemory(d_c);
      return;
    }

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Benchmark vector addition
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      vectorAdd<<<gridSize, blockSize, 0, stream_>>>(d_a, d_b, d_c, size);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    double bandwidth = (3.0 * bytes * iterations) / (duration.count() * 1e-3); // GB/s

    cout << "Vector Addition:" << endl;
    cout << "  Total time: " << fixed << setprecision(2) << time_ms << " ms" << endl;
    cout << "  Avg per kernel: " << fixed << setprecision(3) << time_ms / iterations << " ms"
         << endl;
    cout << "  Bandwidth: " << fixed << setprecision(2) << bandwidth << " GB/s" << endl;

    // Benchmark vector multiplication
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      vectorMul<<<gridSize, blockSize, 0, stream_>>>(d_a, d_b, d_c, size);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    end = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    time_ms = duration.count() / 1000.0;
    bandwidth = (3.0 * bytes * iterations) / (duration.count() * 1e-3); // GB/s

    cout << "Vector Multiplication:" << endl;
    cout << "  Total time: " << fixed << setprecision(2) << time_ms << " ms" << endl;
    cout << "  Avg per kernel: " << fixed << setprecision(3) << time_ms / iterations << " ms"
         << endl;
    cout << "  Bandwidth: " << fixed << setprecision(2) << bandwidth << " GB/s" << endl;

    // Verify results using our device manager
    try {
      device_->copyToHost(h_c.data(), d_c, bytes);
      cout << "Successfully copied results back to host using DeviceManager" << endl;
    } catch (const exception &e) {
      cerr << "Failed to copy results back to host: " << e.what() << endl;
    }

    bool correct = true;
    for (int i = 0; i < min(10, size); i++) {
      if (abs(h_c[i] - (h_a[i] * h_b[i])) > 1e-5) {
        correct = false;
        break;
      }
    }
    cout << "Results: " << (correct ? "CORRECT" : "INCORRECT") << endl;

    // Cleanup using our device manager
    try {
      device_->deallocateMemory(d_a);
      device_->deallocateMemory(d_b);
      device_->deallocateMemory(d_c);
      cout << "Successfully deallocated device memory using DeviceManager" << endl;
    } catch (const exception &e) {
      cerr << "Failed to deallocate memory: " << e.what() << endl;
    }
    cout << endl;
  }

  void benchmarkMathOperations(int size, int iterations = 100) {
    cout << "=== Mathematical Operations Benchmark (Using Device Manager) ===" << endl;
    cout << "Array size: " << size << " elements" << endl;
    cout << "Device: " << device_->getName() << endl;

    size_t bytes = size * sizeof(float);

    // Allocate and initialize host memory
    vector<float> h_input(size);
    for (int i = 0; i < size; i++) {
      h_input[i] = static_cast<float>(i % 1000) / 1000.0f;
    }

    // Allocate device memory using our device manager
    float *d_input, *d_output;
    try {
      d_input = static_cast<float *>(device_->allocateMemory(bytes));
      d_output = static_cast<float *>(device_->allocateMemory(bytes));
    } catch (const exception &e) {
      cerr << "Failed to allocate memory: " << e.what() << endl;
      return;
    }

    // Copy data to device using our device manager
    try {
      device_->copyToDevice(d_input, h_input.data(), bytes);
    } catch (const exception &e) {
      cerr << "Failed to copy data to device: " << e.what() << endl;
      device_->deallocateMemory(d_input);
      device_->deallocateMemory(d_output);
      return;
    }

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Benchmark mathematical operations
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      mathOps<<<gridSize, blockSize, 0, stream_>>>(d_input, d_output, size);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    double throughput = (size * iterations) / (duration.count() * 1e-3); // ops/second

    cout << "Mathematical Operations:" << endl;
    cout << "  Total time: " << fixed << setprecision(2) << time_ms << " ms" << endl;
    cout << "  Avg per kernel: " << fixed << setprecision(3) << time_ms / iterations << " ms"
         << endl;
    cout << "  Throughput: " << fixed << setprecision(2) << throughput / 1e9 << " GOp/s" << endl;

    // Cleanup using our device manager
    try {
      device_->deallocateMemory(d_input);
      device_->deallocateMemory(d_output);
    } catch (const exception &e) {
      cerr << "Failed to deallocate memory: " << e.what() << endl;
    }
    cout << endl;
  }

  void benchmarkMatrixMultiplication(int n, int iterations = 5) {
    cout << "=== Matrix Multiplication Benchmark (Using Device Manager) ===" << endl;
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Device: " << device_->getName() << endl;

    size_t bytes = n * n * sizeof(float);

    // Allocate host memory
    vector<float> h_a(n * n, 1.0f);
    vector<float> h_b(n * n, 2.0f);
    vector<float> h_c(n * n);

    // Allocate device memory using our device manager
    float *d_a, *d_b, *d_c;
    try {
      d_a = static_cast<float *>(device_->allocateMemory(bytes));
      d_b = static_cast<float *>(device_->allocateMemory(bytes));
      d_c = static_cast<float *>(device_->allocateMemory(bytes));
    } catch (const exception &e) {
      cerr << "Failed to allocate memory: " << e.what() << endl;
      return;
    }

    // Copy data to device using our device manager
    try {
      device_->copyToDevice(d_a, h_a.data(), bytes);
      device_->copyToDevice(d_b, h_b.data(), bytes);
    } catch (const exception &e) {
      cerr << "Failed to copy data to device: " << e.what() << endl;
      device_->deallocateMemory(d_a);
      device_->deallocateMemory(d_b);
      device_->deallocateMemory(d_c);
      return;
    }

    // Test naive implementation
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      matrixMul<<<gridDim, blockDim, 0, stream_>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    double gflops = (2.0 * n * n * n * iterations) / (duration.count() * 1e-3);

    cout << "Naive Implementation:" << endl;
    cout << "  Total time: " << fixed << setprecision(2) << time_ms << " ms" << endl;
    cout << "  Avg per kernel: " << fixed << setprecision(3) << time_ms / iterations << " ms"
         << endl;
    cout << "  Performance: " << fixed << setprecision(2) << gflops << " GFLOP/s" << endl;

    // Test shared memory implementation
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      matrixMulShared<<<gridDim, blockDim, 0, stream_>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    end = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    time_ms = duration.count() / 1000.0;
    gflops = (2.0 * n * n * n * iterations) / (duration.count() * 1e-3);

    cout << "Shared Memory Implementation:" << endl;
    cout << "  Total time: " << fixed << setprecision(2) << time_ms << " ms" << endl;
    cout << "  Avg per kernel: " << fixed << setprecision(3) << time_ms / iterations << " ms"
         << endl;
    cout << "  Performance: " << fixed << setprecision(2) << gflops << " GFLOP/s" << endl;

    // Verify results using our device manager
    try {
      device_->copyToHost(h_c.data(), d_c, bytes);
    } catch (const exception &e) {
      cerr << "Failed to copy results back to host: " << e.what() << endl;
    }

    bool correct = true;
    for (int i = 0; i < min(10, n * n); i++) {
      if (abs(h_c[i] - (2.0f * n)) > 1e-3) {
        correct = false;
        break;
      }
    }
    cout << "Results: " << (correct ? "CORRECT" : "INCORRECT") << endl;

    // Cleanup using our device manager
    try {
      device_->deallocateMemory(d_a);
      device_->deallocateMemory(d_b);
      device_->deallocateMemory(d_c);
    } catch (const exception &e) {
      cerr << "Failed to deallocate memory: " << e.what() << endl;
    }
    cout << endl;
  }

  void benchmarkMemoryBandwidth(int size, int iterations = 100) {
    cout << "=== Memory Bandwidth Benchmark (Using Device Manager) ===" << endl;
    cout << "Array size: " << size << " elements" << endl;
    cout << "Device: " << device_->getName() << endl;

    size_t bytes = size * sizeof(float);

    // Allocate device memory using our device manager
    float *d_input, *d_output;
    try {
      d_input = static_cast<float *>(device_->allocateMemory(bytes));
      d_output = static_cast<float *>(device_->allocateMemory(bytes));
    } catch (const exception &e) {
      cerr << "Failed to allocate memory: " << e.what() << endl;
      return;
    }

    // Initialize device memory (we'll use CUDA directly for memset since it's not in our Device
    // interface)
    CUDA_CHECK(cudaMemset(d_input, 0x42, bytes));

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Benchmark memory copy
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      memcopy<<<gridSize, blockSize, 0, stream_>>>(d_input, d_output, size);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    double bandwidth =
        (2.0 * bytes * iterations) / (duration.count() * 1e-3); // GB/s (read + write)

    cout << "Kernel Memory Copy:" << endl;
    cout << "  Total time: " << fixed << setprecision(2) << time_ms << " ms" << endl;
    cout << "  Avg per kernel: " << fixed << setprecision(3) << time_ms / iterations << " ms"
         << endl;
    cout << "  Bandwidth: " << fixed << setprecision(2) << bandwidth << " GB/s" << endl;

    // Cleanup using our device manager
    try {
      device_->deallocateMemory(d_input);
      device_->deallocateMemory(d_output);
    } catch (const exception &e) {
      cerr << "Failed to deallocate memory: " << e.what() << endl;
    }
    cout << endl;
  }

  void runAllBenchmarks() {
    printDeviceInfo();

    // Test different vector sizes
    vector<int> vector_sizes = {1024, 1024 * 1024, 16 * 1024 * 1024};
    for (int size : vector_sizes) {
      benchmarkVectorOperations(size);
      benchmarkMathOperations(size);
      benchmarkMemoryBandwidth(size);
    }

    // Test different matrix sizes
    vector<int> matrix_sizes = {64, 128, 256, 512};
    for (int size : matrix_sizes) {
      benchmarkMatrixMultiplication(size);
    }
  }
};

#endif // USE_CUDA

int main() {
  cout << "=== Simple CUDA Performance Test with Device Manager ===" << endl;

#ifdef USE_CUDA
  // Initialize the device manager first
  cout << "Initializing device manager..." << endl;
  try {
    initializeDefaultDevices();
  } catch (const exception &e) {
    cerr << "Failed to initialize device manager: " << e.what() << endl;
    return 1;
  }

  DeviceManager &manager = DeviceManager::getInstance();
  vector<string> device_ids = manager.getAvailableDeviceIDs();

  cout << "Found " << device_ids.size() << " device(s) in device manager" << endl;

  // List all devices
  for (const string &id : device_ids) {
    const Device &device = manager.getDevice(id);
    cout << "  Device " << id << ": " << device.getName()
         << " (Type: " << (device.getDeviceType() == DeviceType::CPU ? "CPU" : "GPU") << ")"
         << " - Total Memory: " << device.getTotalMemory() / (1024 * 1024) << " MB" << endl;
  }

  // Find GPU devices and test them
  bool found_gpu = false;
  for (const string &id : device_ids) {
    const Device &device = manager.getDevice(id);
    if (device.getDeviceType() == DeviceType::GPU) {
      cout << "\n" << string(60, '=') << endl;
      cout << "Testing GPU Device " << id << " (" << device.getName() << ")" << endl;
      cout << string(60, '=') << endl;

      try {
        // Note: We're still using CUDA device ID 0 for the actual CUDA context
        // In a more sophisticated implementation, you'd map device manager IDs to CUDA device IDs
        SimpleCUDABenchmark benchmark(0);
        benchmark.runAllBenchmarks();
        found_gpu = true;
      } catch (const exception &e) {
        cerr << "Error testing device " << id << ": " << e.what() << endl;
      }
      break; // Test only the first GPU for now
    }
  }

  if (!found_gpu) {
    cout << "No GPU devices found in device manager." << endl;

    // Fallback to direct CUDA detection
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      cout << "No CUDA devices found or CUDA not available." << endl;
      return 0;
    }

    cout << "Found " << deviceCount << " CUDA device(s) directly, but not in device manager"
         << endl;
    cout << "This suggests the device manager initialization might need debugging." << endl;
    return 1;
  }

  cout << "\n=== All CUDA performance tests with Device Manager completed ===" << endl;

#else
  cout << "CUDA support not compiled. Please rebuild with -DENABLE_CUDA=ON" << endl;
  return 1;
#endif

  return 0;
}