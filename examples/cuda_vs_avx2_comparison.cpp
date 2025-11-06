#include "cuda/kernels.hpp"
#include "threading/thread_handler.hpp"
#include "threading/thread_wrapper.hpp"
#include "utils/avx2.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

using namespace tthreads;

void printVector(const std::vector<float> &vec, const std::string &name, size_t maxElements = 10) {
  std::cout << name << ": [";
  size_t printCount = std::min(maxElements, vec.size());
  for (size_t i = 0; i < printCount; ++i) {
    std::cout << std::fixed << std::setprecision(3) << vec[i];
    if (i < printCount - 1)
      std::cout << ", ";
  }
  if (vec.size() > maxElements) {
    std::cout << ", ... (" << vec.size() << " total)";
  }
  std::cout << "]" << std::endl;
}

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
void avx2_add_multithreaded(const float *a, const float *b, float *c, size_t size,
                            int num_threads = 0) {
  if (num_threads <= 0) {
    num_threads = std::thread::hardware_concurrency();
  }

  size_t chunk_size = size / num_threads;

  tthreads::parallel_for<size_t>(0, num_threads, [&](size_t i) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    utils::avx2_add(a + start, b + start, c + start, end - start);
  });
}

void avx2_mul_multithreaded(const float *a, const float *b, float *c, size_t size,
                            int num_threads = 0) {
  if (num_threads <= 0) {
    num_threads = std::thread::hardware_concurrency();
  }

  size_t chunk_size = size / num_threads;

  tthreads::parallel_for<size_t>(0, num_threads, [&](size_t i) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    utils::avx2_mul(a + start, b + start, c + start, end - start);
  });
}

void avx2_add_scalar_multithreaded(const float *a, float scalar, float *c, size_t size,
                                   int num_threads = 0) {
  if (num_threads <= 0) {
    num_threads = std::thread::hardware_concurrency();
  }

  size_t chunk_size = size / num_threads;

  tthreads::parallel_for<size_t>(0, num_threads, [&](size_t i) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    utils::avx2_add_scalar(a + start, scalar, c + start, end - start);
  });
}

void avx2_sqrt_multithreaded(const float *a, float *c, size_t size, int num_threads = 0) {
  if (num_threads <= 0) {
    num_threads = std::thread::hardware_concurrency();
  }

  size_t chunk_size = size / num_threads;

  tthreads::parallel_for<size_t>(0, num_threads, [&](size_t i) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    utils::avx2_sqrt(a + start, c + start, end - start);
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
  std::cout << "=== Optimized CUDA Kernels vs AVX2 Performance Comparison ===" << std::endl;

  // Test parameters
  const size_t SIZE = 100000000; // Start with smaller size for more detailed analysis
  const int ITERATIONS = 5;
  const int num_cpu_threads = std::thread::hardware_concurrency();

  std::cout << "Available CPU threads: " << num_cpu_threads << std::endl;

  // Measure memory bandwidth
  double memBandwidth = measureMemoryBandwidth();
  std::cout << "System memory bandwidth: " << std::fixed << std::setprecision(1) << memBandwidth
            << " GB/s" << std::endl;

  // Timing variables
  double avx2SingleTime = 0.0;
  double avx2MultiTime = 0.0;
  double cudaTime = 0.0;

  // Host data
  std::vector<float> a(SIZE), b(SIZE), c_cuda(SIZE), c_avx2_single(SIZE), c_avx2_multi(SIZE);

  // Initialize test data
  for (size_t i = 0; i < SIZE; ++i) {
    a[i] = static_cast<float>(i * 0.001f);
    b[i] = static_cast<float>((i + 1) * 0.002f);
  }

  std::cout << "Test size: " << SIZE << " elements (" << (SIZE * sizeof(float) / 1024 / 1024)
            << " MB per array)" << std::endl;
  std::cout << "Total memory for 3 arrays: " << (SIZE * sizeof(float) * 3 / 1024 / 1024) << " MB"
            << std::endl;
  std::cout << "Iterations: " << ITERATIONS << std::endl << std::endl;

  // CUDA setup
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, SIZE * sizeof(float));
  cudaMalloc(&d_b, SIZE * sizeof(float));
  cudaMalloc(&d_c, SIZE * sizeof(float));

  cudaMemcpy(d_a, a.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);

  std::cout << "=== CUDA Tests (with memory transfer overhead) ===" << std::endl;

  // CUDA Addition (including memory transfers)
  cudaTime = timeFunction(
      [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
          cudaMemcpy(d_a, a.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(d_b, b.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
          cuda::cuda_add(d_a, d_b, d_c, SIZE);
          cudaMemcpy(c_cuda.data(), d_c, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
      },
      "CUDA Addition (with memory transfer)");

  std::cout << std::endl << "=== CUDA Tests (compute only) ===" << std::endl;

  // CUDA Addition (compute only)
  double cudaComputeTime = timeFunction(
      [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
          cuda::cuda_add(d_a, d_b, d_c, SIZE);
        }
        cudaDeviceSynchronize();
      },
      "CUDA Addition (compute only)");

  // More CUDA compute-only tests
  timeFunction(
      [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
          cuda::cuda_mul(d_a, d_b, d_c, SIZE);
        }
        cudaDeviceSynchronize();
      },
      "CUDA Multiplication (compute only)");

  timeFunction(
      [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
          cuda::cuda_add_scalar(d_a, 3.14f, d_c, SIZE);
        }
        cudaDeviceSynchronize();
      },
      "CUDA Add Scalar (compute only)");

  timeFunction(
      [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
          cuda::cuda_sqrt(d_a, d_c, SIZE);
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
            utils::avx2_add(a.data(), b.data(), c_avx2_single.data(), SIZE);
          }
        },
        "AVX2 Addition (single-threaded)");

    // More AVX2 single-threaded tests
    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            utils::avx2_mul(a.data(), b.data(), c_avx2_single.data(), SIZE);
          }
        },
        "AVX2 Multiplication (single-threaded)");

    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            utils::avx2_add_scalar(a.data(), 3.14f, c_avx2_single.data(), SIZE);
          }
        },
        "AVX2 Add Scalar (single-threaded)");

    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            utils::avx2_sqrt(a.data(), c_avx2_single.data(), SIZE);
          }
        },
        "AVX2 Square Root (single-threaded)");

    std::cout << std::endl << "=== AVX2 Tests (Multi-threaded) ===" << std::endl;

    // AVX2 Addition (multi-threaded)
    avx2MultiTime = timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            avx2_add_multithreaded(a.data(), b.data(), c_avx2_multi.data(), SIZE);
          }
        },
        "AVX2 Addition (multi-threaded)");

    // More AVX2 multi-threaded tests
    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            avx2_mul_multithreaded(a.data(), b.data(), c_avx2_multi.data(), SIZE);
          }
        },
        "AVX2 Multiplication (multi-threaded)");

    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            avx2_add_scalar_multithreaded(a.data(), 3.14f, c_avx2_multi.data(), SIZE);
          }
        },
        "AVX2 Add Scalar (multi-threaded)");

    timeFunction(
        [&]() {
          for (int i = 0; i < ITERATIONS; ++i) {
            avx2_sqrt_multithreaded(a.data(), c_avx2_multi.data(), SIZE);
          }
        },
        "AVX2 Square Root (multi-threaded)");
#endif
  });

  // Cleanup CUDA memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

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