#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"
#include "ops/cpu/kernels.hpp"
#include "ops/cuda/kernels.hpp"
#include "threading/thread_handler.hpp"
#include "threading/thread_wrapper.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

using namespace tnn;

template <typename Func> double timeFunction(Func &&func, const std::string &name) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double timeMs = duration.count() / 1000.0;
  std::cout << name << " took: " << std::fixed << std::setprecision(3) << timeMs << " ms"
            << std::endl;
  return timeMs;
}

// Multithreaded AVX2 implementation
void avx2_add_multithreaded(const float *a, const float *b, float *c, size_t size) {
  size_t num_threads = tbb::this_task_arena::max_concurrency();
  size_t chunk_size = size / num_threads;

  parallel_for<size_t>(0, num_threads, [&](size_t i) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    ops::cpu::add(a + start, b + start, c + start, end - start);
  });
}

void avx2_mul_multithreaded(const float *a, const float *b, float *c, size_t size) {
  size_t num_threads = tbb::this_task_arena::max_concurrency();
  size_t chunk_size = size / num_threads;

  parallel_for<size_t>(0, num_threads, [&](size_t i) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    ops::cpu::mul(a + start, b + start, c + start, end - start);
  });
}

void avx2_add_scalar_multithreaded(const float *a, float scalar, float *c, size_t size) {
  size_t num_threads = tbb::this_task_arena::max_concurrency();
  size_t chunk_size = size / num_threads;

  parallel_for<size_t>(0, num_threads, [&](size_t i) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    ops::cpu::add_scalar(a + start, scalar, c + start, end - start);
  });
}

void avx2_sqrt_multithreaded(const float *a, float *c, size_t size) {
  size_t num_threads = tbb::this_task_arena::max_concurrency();
  size_t chunk_size = size / num_threads;

  parallel_for<size_t>(0, num_threads, [&](size_t i) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    ops::cpu::sqrt(a + start, c + start, end - start);
  });
}

// Function to measure memory bandwidth
double measureMemoryBandwidth(size_t size_mb = 1024) {
  size_t size = size_mb * 1024 * 1024 / sizeof(float);
  std::vector<float> src(size, 1.0f);
  std::vector<float> dst(size);

  auto start = std::chrono::high_resolution_clock::now();
  std::memcpy(dst.data(), src.data(), size * sizeof(float));
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double timeSeconds = duration.count() / 1e6;
  double bandwidthGBps = (size * sizeof(float) * 2) / (timeSeconds * 1e9); // Read + Write

  return bandwidthGBps;
}

int main() {
  initializeDefaultDevices();

  // Test parameters
  const size_t SIZE = 100000000; // Start with smaller size for more detailed analysis
  const int ITERATIONS = 5;
  const int num_cpu_threads = std::thread::hardware_concurrency();

  std::cout << "Available CPU threads: " << num_cpu_threads << std::endl;

  // Measure memory bandwidth
  double memBandwidth = measureMemoryBandwidth();
  std::cout << "System memory bandwidth: " << std::fixed << std::setprecision(1) << memBandwidth
            << " GB/s" << std::endl;

  // Get devices
  const Device &cpuDevice = getCPU();
  const Device &gpuDevice = getGPU(0);

  std::cout << "Using CPU device: " << cpuDevice.getName() << std::endl;
  std::cout << "Using GPU device: " << gpuDevice.getName() << std::endl;

  // Timing variables
  double avx2SingleTime = 0.0;
  double avx2MultiTime = 0.0;
  double cudaTime = 0.0;

  // Create device pointers for arrays
  auto cpu_a = make_array_ptr<float[]>(const_cast<Device *>(&cpuDevice), SIZE);
  auto cpu_b = make_array_ptr<float[]>(const_cast<Device *>(&cpuDevice), SIZE);
  auto cpu_c_avx2_single = make_array_ptr<float[]>(const_cast<Device *>(&cpuDevice), SIZE);
  auto cpu_c_avx2_multi = make_array_ptr<float[]>(const_cast<Device *>(&cpuDevice), SIZE);
  auto cpu_c_cuda = make_array_ptr<float[]>(const_cast<Device *>(&cpuDevice), SIZE);

  auto gpu_a = make_array_ptr<float[]>(const_cast<Device *>(&gpuDevice), SIZE);
  auto gpu_b = make_array_ptr<float[]>(const_cast<Device *>(&gpuDevice), SIZE);
  auto gpu_c = make_array_ptr<float[]>(const_cast<Device *>(&gpuDevice), SIZE);

  // Initialize test data on CPU
  for (size_t i = 0; i < SIZE; ++i) {
    cpu_a.get()[i] = static_cast<float>(i * 0.001f);
    cpu_b.get()[i] = static_cast<float>((i + 1) * 0.002f);
  }

  std::cout << "Test size: " << SIZE << " elements (" << (SIZE * sizeof(float) / 1024 / 1024)
            << " MB per array)" << std::endl;
  std::cout << "Total memory for 3 arrays: " << (SIZE * sizeof(float) * 3 / 1024 / 1024) << " MB"
            << std::endl;
  std::cout << "Iterations: " << ITERATIONS << std::endl << std::endl;

  gpuDevice.copyToDevice(gpu_a.get(), cpu_a.get(), SIZE * sizeof(float));
  gpuDevice.copyToDevice(gpu_b.get(), cpu_b.get(), SIZE * sizeof(float));

  std::cout << "=== CUDA Tests (with memory transfer overhead) ===" << std::endl;

  // CUDA Addition (including memory transfers)
  cudaTime = timeFunction(
      [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
          gpuDevice.copyToDevice(gpu_a.get(), cpu_a.get(), SIZE * sizeof(float));
          gpuDevice.copyToDevice(gpu_b.get(), cpu_b.get(), SIZE * sizeof(float));
          cuda::cuda_add(gpu_a.get(), gpu_b.get(), gpu_c.get(), SIZE, 0);
          gpuDevice.copyToHost(cpu_c_cuda.get(), gpu_c.get(), SIZE * sizeof(float));
        }
        cudaDeviceSynchronize();
      },
      "CUDA Addition (with memory transfer)");

  std::cout << std::endl << "=== CUDA Tests (compute only) ===" << std::endl;

  // CUDA Addition (compute only)
  double cudaComputeTime = timeFunction(
      [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
          cuda::cuda_add(gpu_a.get(), gpu_b.get(), gpu_c.get(), SIZE, 0);
        }
        cudaDeviceSynchronize();
      },
      "CUDA Addition (compute only)");

  // More CUDA compute-only tests
  timeFunction(
      [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
          cuda::cuda_mul(gpu_a.get(), gpu_b.get(), gpu_c.get(), SIZE, 0);
        }
        cudaDeviceSynchronize();
      },
      "CUDA Multiplication (compute only)");

  timeFunction(
      [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
          cuda::cuda_add_scalar(gpu_a.get(), 3.14f, gpu_c.get(), SIZE, 0);
        }
        cudaDeviceSynchronize();
      },
      "CUDA Add Scalar (compute only)");

  timeFunction(
      [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
          cuda::cuda_sqrt(gpu_a.get(), gpu_c.get(), SIZE, 0);
        }
        cudaDeviceSynchronize();
      },
      "CUDA Square Root (compute only)");

  ThreadWrapper threadWrapper({8});

  threadWrapper.execute([&]() {
#ifdef __AVX2__
    std::cout << std::endl << "=== AVX2 Tests (Single-threaded) ===" << std::endl;

    // AVX2 Addition (single-threaded)
    avx2SingleTime = timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            ops::cpu::add(cpu_a.get(), cpu_b.get(), cpu_c_avx2_single.get(), SIZE);
          }
        },
        "AVX2 Addition (single-threaded)");

    // More AVX2 single-threaded tests
    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            ops::cpu::mul(cpu_a.get(), cpu_b.get(), cpu_c_avx2_single.get(), SIZE);
          }
        },
        "AVX2 Multiplication (single-threaded)");

    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            ops::cpu::add_scalar(cpu_a.get(), 3.14f, cpu_c_avx2_single.get(), SIZE);
          }
        },
        "AVX2 Add Scalar (single-threaded)");

    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            ops::cpu::sqrt(cpu_a.get(), cpu_c_avx2_single.get(), SIZE);
          }
        },
        "AVX2 Square Root (single-threaded)");

    std::cout << std::endl << "=== AVX2 Tests (Multi-threaded) ===" << std::endl;

    // AVX2 Addition (multi-threaded)
    avx2MultiTime = timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            avx2_add_multithreaded(cpu_a.get(), cpu_b.get(), cpu_c_avx2_multi.get(), SIZE);
          }
        },
        "AVX2 Addition (multi-threaded)");

    // More AVX2 multi-threaded tests
    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            avx2_mul_multithreaded(cpu_a.get(), cpu_b.get(), cpu_c_avx2_multi.get(), SIZE);
          }
        },
        "AVX2 Multiplication (multi-threaded)");

    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            avx2_add_scalar_multithreaded(cpu_a.get(), 3.14f, cpu_c_avx2_multi.get(), SIZE);
          }
        },
        "AVX2 Add Scalar (multi-threaded)");

    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            avx2_sqrt_multithreaded(cpu_a.get(), cpu_c_avx2_multi.get(), SIZE);
          }
        },
        "AVX2 Square Root (multi-threaded)");
#endif
  });

  // Performance analysis
  std::cout << std::endl << "=== Performance Analysis ===" << std::endl;

  // Calculate theoretical memory bandwidth usage
  double dataPerIteration = SIZE * sizeof(float) * 3; // Read A, Read B, Write C
  double totalData = dataPerIteration * ITERATIONS;
  double totalDataGB = totalData / (1024.0 * 1024.0 * 1024.0);

  std::cout << "Data per iteration: " << (dataPerIteration / 1024 / 1024) << " MB" << std::endl;
  std::cout << "Total data transferred: " << std::fixed << std::setprecision(2) << totalDataGB
            << " GB" << std::endl;

#ifdef __AVX2__
  if (avx2SingleTime > 0) {
    double avx2SingleBandwidth = totalDataGB / (avx2SingleTime / 1000.0);
    std::cout << "AVX2 single-threaded effective bandwidth: " << std::fixed << std::setprecision(1)
              << avx2SingleBandwidth << " GB/s" << std::endl;
  }

  if (avx2MultiTime > 0) {
    double avx2MultiBandwidth = totalDataGB / (avx2MultiTime / 1000.0);
    std::cout << "AVX2 multi-threaded effective bandwidth: " << std::fixed << std::setprecision(1)
              << avx2MultiBandwidth << " GB/s" << std::endl;
  }
#endif

  if (cudaComputeTime > 0) {
    double cudaComputeBandwidth = totalDataGB / (cudaComputeTime / 1000.0);
    std::cout << "CUDA compute-only effective bandwidth: " << std::fixed << std::setprecision(1)
              << cudaComputeBandwidth << " GB/s" << std::endl;
  }

  std::cout << std::endl << "=== Speedup Comparison ===" << std::endl;
#ifdef __AVX2__
  if (cudaComputeTime > 0 && avx2SingleTime > 0) {
    double speedupVsSingle = avx2SingleTime / cudaComputeTime;
    std::cout << "CUDA vs AVX2 single-threaded speedup: " << std::fixed << std::setprecision(2)
              << speedupVsSingle << "x" << std::endl;
  }

  if (cudaComputeTime > 0 && avx2MultiTime > 0) {
    double speedupVsMulti = avx2MultiTime / cudaComputeTime;
    std::cout << "CUDA vs AVX2 multi-threaded speedup: " << std::fixed << std::setprecision(2)
              << speedupVsMulti << "x" << std::endl;
  }

  if (avx2SingleTime > 0 && avx2MultiTime > 0) {
    double multithreadSpeedup = avx2SingleTime / avx2MultiTime;
    std::cout << "AVX2 multi-threaded vs single-threaded speedup: " << std::fixed
              << std::setprecision(2) << multithreadSpeedup << "x" << std::endl;
  }
#endif

  return 0;
}