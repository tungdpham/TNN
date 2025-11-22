/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers_impl/cpu/conv2d_ops.hpp"
#include "ops/ops.hpp"
#include "tensor/cpu/tensor_ops.hpp"
#include "tensor/tensor.hpp"

#ifdef USE_CUDA
#include "cuda/error_handler.hpp"
#include "device/device_manager.hpp"
#include "nn/layers_impl/cuda/conv2d_ops.hpp"
#include "tensor/cuda/tensor_ops.hpp"
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace tnn;

// Test configurations with increasing complexity
struct BenchmarkConfig {
  size_t batch_size;
  size_t channels;
  size_t height;
  size_t width;
  size_t kernel_h;
  size_t kernel_w;
  size_t stride_h;
  size_t stride_w;
  size_t pad_h;
  size_t pad_w;
  std::string description;
};

std::vector<BenchmarkConfig> get_benchmark_configs() {
  return {
      // Small input (typical early conv layers)
      {32, 64, 56, 56, 3, 3, 1, 1, 1, 1, "Small (32x64x56x56, 3x3 kernel, pad=1, stride=1)"},

      // Medium input (middle conv layers)
      {32, 128, 28, 28, 3, 3, 1, 1, 1, 1, "Medium (32x128x28x28, 3x3 kernel, pad=1, stride=1)"},

      // Large input (input layer)
      {64, 256, 112, 112, 3, 3, 2, 2, 1, 1, "Large (64x256x112x112, 3x3 kernel, pad=1, stride=2)"},

      // Very large input (high-res processing) - reduced to avoid GPU memory overflow
      {8, 256, 224, 224, 3, 3, 1, 1, 1, 1,
       "Very Large (8x256x224x224, 3x3 kernel, pad=1, stride=1)"},

      // 5x5 kernel
      {32, 64, 56, 56, 5, 5, 1, 1, 2, 2, "5x5 kernel (32x64x56x56, 5x5 kernel, pad=2, stride=1)"},

      // 7x7 kernel (like ResNet first layer)
      {16, 64, 224, 224, 7, 7, 2, 2, 3, 3,
       "7x7 kernel (16x64x224x224, 7x7 kernel, pad=3, stride=2)"},
  };
}

void print_separator() { std::cout << std::string(80, '=') << std::endl; }

void print_config(const BenchmarkConfig &config) {
  std::cout << "\nConfiguration: " << config.description << std::endl;
  std::cout << "  Input: " << config.batch_size << "x" << config.channels << "x" << config.height
            << "x" << config.width << std::endl;
  std::cout << "  Kernel: " << config.kernel_h << "x" << config.kernel_w << std::endl;
  std::cout << "  Stride: " << config.stride_h << "x" << config.stride_w << std::endl;
  std::cout << "  Padding: " << config.pad_h << "x" << config.pad_w << std::endl;

  // Calculate output dimensions
  size_t padded_h = config.height + 2 * config.pad_h;
  size_t padded_w = config.width + 2 * config.pad_w;
  size_t out_h = (padded_h - config.kernel_h) / config.stride_h + 1;
  size_t out_w = (padded_w - config.kernel_w) / config.stride_w + 1;

  std::cout << "  Output spatial: " << out_h << "x" << out_w << std::endl;

  // Calculate memory sizes
  size_t input_size = config.batch_size * config.channels * config.height * config.width;
  size_t col_size =
      config.batch_size * config.channels * config.kernel_h * config.kernel_w * out_h * out_w;

  std::cout << "  Input memory: " << (input_size * sizeof(float) / 1024.0 / 1024.0) << " MB"
            << std::endl;
  std::cout << "  Col buffer memory: " << (col_size * sizeof(float) / 1024.0 / 1024.0) << " MB"
            << std::endl;
}

void benchmark_cpu_im2col_col2im(const BenchmarkConfig &config, int warmup_runs = 2,
                                 int bench_runs = 5) {
  std::cout << "\n--- CPU Benchmark ---" << std::endl;

  // Create input tensor on CPU
  Tensor<float, NCHW> input({config.batch_size, config.channels, config.height, config.width});
  input.fill_random_normal(0.0f, 1.0f);

  // Calculate output dimensions
  size_t padded_h = config.height + 2 * config.pad_h;
  size_t padded_w = config.width + 2 * config.pad_w;
  size_t out_h = (padded_h - config.kernel_h) / config.stride_h + 1;
  size_t out_w = (padded_w - config.kernel_w) / config.stride_w + 1;

  // Allocate col buffer
  size_t col_size =
      config.batch_size * config.channels * config.kernel_h * config.kernel_w * out_h * out_w;
  std::vector<float> col_buffer(col_size);

  // Allocate output buffer for col2im
  std::vector<float> output_buffer(config.batch_size * config.channels * config.height *
                                   config.width);

  // Warmup
  for (int i = 0; i < warmup_runs; ++i) {
    cpu::im2col(input, col_buffer.data(), config.kernel_h, config.kernel_w, config.stride_h,
                config.stride_w, config.pad_h, config.pad_w);
    cpu::col2im(col_buffer.data(), output_buffer.data(), config.batch_size, config.channels,
                config.height, config.width, config.kernel_h, config.kernel_w, config.stride_h,
                config.stride_w, config.pad_h, config.pad_w);
  }

  // Benchmark im2col
  std::vector<double> im2col_times;
  for (int i = 0; i < bench_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    cpu::im2col(input, col_buffer.data(), config.kernel_h, config.kernel_w, config.stride_h,
                config.stride_w, config.pad_h, config.pad_w);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    im2col_times.push_back(duration.count());
  }

  // Benchmark col2im
  std::vector<double> col2im_times;
  for (int i = 0; i < bench_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    cpu::col2im(col_buffer.data(), output_buffer.data(), config.batch_size, config.channels,
                config.height, config.width, config.kernel_h, config.kernel_w, config.stride_h,
                config.stride_w, config.pad_h, config.pad_w);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    col2im_times.push_back(duration.count());
  }

  double im2col_avg =
      std::accumulate(im2col_times.begin(), im2col_times.end(), 0.0) / im2col_times.size();
  double col2im_avg =
      std::accumulate(col2im_times.begin(), col2im_times.end(), 0.0) / col2im_times.size();

  std::cout << "CPU im2col average: " << im2col_avg << " ms" << std::endl;
  std::cout << "CPU col2im average: " << col2im_avg << " ms" << std::endl;
  std::cout << "CPU total average:  " << (im2col_avg + col2im_avg) << " ms" << std::endl;
}

#ifdef USE_CUDA
void benchmark_cuda_im2col_col2im(const BenchmarkConfig &config, int warmup_runs = 2,
                                  int bench_runs = 5) {
  std::cout << "\n--- CUDA Benchmark ---" << std::endl;

  // Get CUDA device
  const Device &gpu_device = getGPU(0);

  // Create input tensor on GPU
  Tensor<float, NCHW> input({config.batch_size, config.channels, config.height, config.width},
                            &gpu_device);
  input.fill_random_normal(0.0f, 1.0f);

  // Calculate output dimensions
  size_t padded_h = config.height + 2 * config.pad_h;
  size_t padded_w = config.width + 2 * config.pad_w;
  size_t out_h = (padded_h - config.kernel_h) / config.stride_h + 1;
  size_t out_w = (padded_w - config.kernel_w) / config.stride_w + 1;

  // Allocate col buffer on GPU
  size_t col_size =
      config.batch_size * config.channels * config.kernel_h * config.kernel_w * out_h * out_w;
  float *col_buffer;
  CUDA_CHECK(cudaMalloc(&col_buffer, col_size * sizeof(float)));

  // Allocate output buffer on GPU
  float *output_buffer;
  size_t output_size = config.batch_size * config.channels * config.height * config.width;
  CUDA_CHECK(cudaMalloc(&output_buffer, output_size * sizeof(float)));

  // Warmup
  for (int i = 0; i < warmup_runs; ++i) {
    cuda::im2col(input, col_buffer, config.kernel_h, config.kernel_w, config.stride_h,
                 config.stride_w, config.pad_h, config.pad_w);
    cuda::col2im(col_buffer, output_buffer, config.batch_size, config.channels, config.height,
                 config.width, config.kernel_h, config.kernel_w, config.stride_h, config.stride_w,
                 config.pad_h, config.pad_w);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark im2col
  std::vector<double> im2col_times;
  for (int i = 0; i < bench_runs; ++i) {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    cuda::im2col(input, col_buffer, config.kernel_h, config.kernel_w, config.stride_h,
                 config.stride_w, config.pad_h, config.pad_w);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    im2col_times.push_back(duration.count());
  }

  // Benchmark col2im
  std::vector<double> col2im_times;
  for (int i = 0; i < bench_runs; ++i) {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    cuda::col2im(col_buffer, output_buffer, config.batch_size, config.channels, config.height,
                 config.width, config.kernel_h, config.kernel_w, config.stride_h, config.stride_w,
                 config.pad_h, config.pad_w);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    col2im_times.push_back(duration.count());
  }

  double im2col_avg =
      std::accumulate(im2col_times.begin(), im2col_times.end(), 0.0) / im2col_times.size();
  double col2im_avg =
      std::accumulate(col2im_times.begin(), col2im_times.end(), 0.0) / col2im_times.size();

  std::cout << "CUDA im2col average: " << im2col_avg << " ms" << std::endl;
  std::cout << "CUDA col2im average: " << col2im_avg << " ms" << std::endl;
  std::cout << "CUDA total average:  " << (im2col_avg + col2im_avg) << " ms" << std::endl;

  // Cleanup
  CUDA_CHECK(cudaFree(col_buffer));
  CUDA_CHECK(cudaFree(output_buffer));
}

void run_comparison_benchmark(const BenchmarkConfig &config, int warmup_runs = 2,
                              int bench_runs = 5) {
  print_config(config);

  // Run CPU benchmark
  benchmark_cpu_im2col_col2im(config, warmup_runs, bench_runs);

  // Run CUDA benchmark
  benchmark_cuda_im2col_col2im(config, warmup_runs, bench_runs);

  std::cout << std::endl;
}
#else
void run_comparison_benchmark(const BenchmarkConfig &config, int warmup_runs = 2,
                              int bench_runs = 5) {
  print_config(config);

  // Run CPU benchmark only
  benchmark_cpu_im2col_col2im(config, warmup_runs, bench_runs);

  std::cout << "\nNote: CUDA benchmarks skipped (built without CUDA support)" << std::endl;
  std::cout << std::endl;
}
#endif

void benchmark_cpu_layout_transform(const BenchmarkConfig &config, int warmup_runs = 2,
                                    int bench_runs = 5) {
  std::cout << "\n--- CPU Layout Transform Benchmark ---" << std::endl;

  // Create input tensor on CPU
  Tensor<float, NCHW> input({config.batch_size, config.channels, config.height, config.width});
  input.fill_random_normal(0.0f, 1.0f);

  size_t total_size = config.batch_size * config.channels * config.height * config.width;

  // Allocate output buffers
  auto output_cnhw = make_array_ptr<float[]>(&getCPU(), total_size);
  auto output_nchw = make_array_ptr<float[]>(&getCPU(), total_size);

  // Warmup
  for (int i = 0; i < warmup_runs; ++i) {
    [[maybe_unused]] auto status1 =
        ops::nchw_to_cnhw(input.data_ptr(), output_cnhw, config.batch_size, config.channels,
                          config.height, config.width)
            ->sync();
    [[maybe_unused]] auto status2 = ops::cnhw_to_nchw(output_cnhw, output_nchw, config.batch_size,
                                                      config.channels, config.height, config.width)
                                        ->sync();
  }

  // Benchmark nchw_to_cnhw
  std::vector<double> nchw_to_cnhw_times;
  for (int i = 0; i < bench_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto task = ops::nchw_to_cnhw(input.data_ptr(), output_cnhw, config.batch_size, config.channels,
                                  config.height, config.width);
    [[maybe_unused]] auto status = task->sync();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    nchw_to_cnhw_times.push_back(duration.count());
  }

  // Benchmark cnhw_to_nchw
  std::vector<double> cnhw_to_nchw_times;
  for (int i = 0; i < bench_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto task = ops::cnhw_to_nchw(output_cnhw, output_nchw, config.batch_size, config.channels,
                                  config.height, config.width);
    [[maybe_unused]] auto status = task->sync();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    cnhw_to_nchw_times.push_back(duration.count());
  }

  double nchw_to_cnhw_avg =
      std::accumulate(nchw_to_cnhw_times.begin(), nchw_to_cnhw_times.end(), 0.0) /
      nchw_to_cnhw_times.size();
  double cnhw_to_nchw_avg =
      std::accumulate(cnhw_to_nchw_times.begin(), cnhw_to_nchw_times.end(), 0.0) /
      cnhw_to_nchw_times.size();

  std::cout << "CPU nchw_to_cnhw average: " << nchw_to_cnhw_avg << " ms" << std::endl;
  std::cout << "CPU cnhw_to_nchw average: " << cnhw_to_nchw_avg << " ms" << std::endl;
  std::cout << "CPU total average:        " << (nchw_to_cnhw_avg + cnhw_to_nchw_avg) << " ms"
            << std::endl;
}

#ifdef USE_CUDA
void benchmark_cuda_layout_transform(const BenchmarkConfig &config, int warmup_runs = 2,
                                     int bench_runs = 5) {
  std::cout << "\n--- CUDA Layout Transform Benchmark ---" << std::endl;

  // Get CUDA device
  const Device &gpu_device = getGPU(0);

  // Create input tensor on GPU
  Tensor<float, NCHW> input({config.batch_size, config.channels, config.height, config.width},
                            &gpu_device);
  input.fill_random_normal(0.0f, 1.0f);

  size_t total_size = config.batch_size * config.channels * config.height * config.width;

  // Allocate output buffers on GPU
  auto output_cnhw = make_array_ptr<float[]>(&gpu_device, total_size);
  auto output_nchw = make_array_ptr<float[]>(&gpu_device, total_size);

  // Warmup
  for (int i = 0; i < warmup_runs; ++i) {
    auto task1 = ops::nchw_to_cnhw(input.data_ptr(), output_cnhw, config.batch_size,
                                   config.channels, config.height, config.width);
    [[maybe_unused]] auto status1 = task1->sync();
    auto task2 = ops::cnhw_to_nchw(output_cnhw, output_nchw, config.batch_size, config.channels,
                                   config.height, config.width);
    [[maybe_unused]] auto status2 = task2->sync();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark nchw_to_cnhw
  std::vector<double> nchw_to_cnhw_times;
  for (int i = 0; i < bench_runs; ++i) {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    auto task = ops::nchw_to_cnhw(input.data_ptr(), output_cnhw, config.batch_size, config.channels,
                                  config.height, config.width);
    [[maybe_unused]] auto status = task->sync();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    nchw_to_cnhw_times.push_back(duration.count());
  }

  // Benchmark cnhw_to_nchw
  std::vector<double> cnhw_to_nchw_times;
  for (int i = 0; i < bench_runs; ++i) {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    auto task = ops::cnhw_to_nchw(output_cnhw, output_nchw, config.batch_size, config.channels,
                                  config.height, config.width);
    [[maybe_unused]] auto status = task->sync();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    cnhw_to_nchw_times.push_back(duration.count());
  }

  double nchw_to_cnhw_avg =
      std::accumulate(nchw_to_cnhw_times.begin(), nchw_to_cnhw_times.end(), 0.0) /
      nchw_to_cnhw_times.size();
  double cnhw_to_nchw_avg =
      std::accumulate(cnhw_to_nchw_times.begin(), cnhw_to_nchw_times.end(), 0.0) /
      cnhw_to_nchw_times.size();

  std::cout << "CUDA nchw_to_cnhw average: " << nchw_to_cnhw_avg << " ms" << std::endl;
  std::cout << "CUDA cnhw_to_nchw average: " << cnhw_to_nchw_avg << " ms" << std::endl;
  std::cout << "CUDA total average:        " << (nchw_to_cnhw_avg + cnhw_to_nchw_avg) << " ms"
            << std::endl;
}

void run_layout_transform_benchmark(const BenchmarkConfig &config, int warmup_runs = 2,
                                    int bench_runs = 5) {
  std::cout << "\n=== Layout Transform Benchmark ===" << std::endl;
  std::cout << "Configuration: " << config.batch_size << "x" << config.channels << "x"
            << config.height << "x" << config.width << std::endl;

  size_t total_size = config.batch_size * config.channels * config.height * config.width;
  std::cout << "Total elements: " << total_size << std::endl;
  std::cout << "Memory size: " << (total_size * sizeof(float) / 1024.0 / 1024.0) << " MB"
            << std::endl;

  // Run CPU benchmark
  benchmark_cpu_layout_transform(config, warmup_runs, bench_runs);

  // Run CUDA benchmark
  benchmark_cuda_layout_transform(config, warmup_runs, bench_runs);

  std::cout << std::endl;
}
#else
void run_layout_transform_benchmark(const BenchmarkConfig &config, int warmup_runs = 2,
                                    int bench_runs = 5) {
  std::cout << "\n=== Layout Transform Benchmark ===" << std::endl;
  std::cout << "Configuration: " << config.batch_size << "x" << config.channels << "x"
            << config.height << "x" << config.width << std::endl;

  size_t total_size = config.batch_size * config.channels * config.height * config.width;
  std::cout << "Total elements: " << total_size << std::endl;
  std::cout << "Memory size: " << (total_size * sizeof(float) / 1024.0 / 1024.0) << " MB"
            << std::endl;

  // Run CPU benchmark only
  benchmark_cpu_layout_transform(config, warmup_runs, bench_runs);

  std::cout << "\nNote: CUDA benchmarks skipped (built without CUDA support)" << std::endl;
  std::cout << std::endl;
}
#endif

void benchmark_cpu_conv2d_gradients(const BenchmarkConfig &config, int warmup_runs = 2,
                                    int bench_runs = 5) {
  std::cout << "\n--- CPU Conv2D Gradients Benchmark ---" << std::endl;

  const size_t batch_size = config.batch_size;
  const size_t in_channels = config.channels;
  const size_t out_channels = config.channels * 2; // Double channels for output
  const size_t input_h = config.height;
  const size_t input_w = config.width;

  const size_t output_h = (input_h + 2 * config.pad_h - config.kernel_h) / config.stride_h + 1;
  const size_t output_w = (input_w + 2 * config.pad_w - config.kernel_w) / config.stride_w + 1;

  const size_t kernel_size = in_channels * config.kernel_h * config.kernel_w;
  const size_t output_size = batch_size * output_h * output_w;

  // Allocate buffers
  std::vector<float> col_data(kernel_size * output_size);
  std::vector<float> gradient_data(out_channels * output_size);
  std::vector<float> weight_data(out_channels * kernel_size);
  std::vector<float> weight_grad_data(out_channels * kernel_size);
  std::vector<float> col_grad_data(kernel_size * output_size);

  // Fill with random data
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &val : col_data)
    val = dist(gen);
  for (auto &val : gradient_data)
    val = dist(gen);
  for (auto &val : weight_data)
    val = dist(gen);

  // Warmup
  for (int i = 0; i < warmup_runs; ++i) {
    auto weight_task = create_cpu_task(
        "benchmark_weight_grad", cpu::conv2d::compute_weight_gradients<float>, col_data.data(),
        gradient_data.data(), weight_grad_data.data(), output_size, kernel_size, out_channels);
    weight_task->sync();
    auto input_task = create_cpu_task(
        "benchmark_input_grad", cpu::conv2d::compute_input_gradients<float>, gradient_data.data(),
        weight_data.data(), col_grad_data.data(), output_size, kernel_size, out_channels);
    input_task->sync();
  }

  // Benchmark compute_weight_gradients
  std::vector<double> weight_grad_times;
  for (int i = 0; i < bench_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto weight_task = create_cpu_task(
        "benchmark_weight_grad", cpu::conv2d::compute_weight_gradients<float>, col_data.data(),
        gradient_data.data(), weight_grad_data.data(), output_size, kernel_size, out_channels);
    weight_task->sync();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    weight_grad_times.push_back(duration.count());
  }

  // Benchmark compute_input_gradients
  std::vector<double> input_grad_times;
  for (int i = 0; i < bench_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto input_task = create_cpu_task(
        "benchmark_input_grad", cpu::conv2d::compute_input_gradients<float>, gradient_data.data(),
        weight_data.data(), col_grad_data.data(), output_size, kernel_size, out_channels);
    input_task->sync();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    input_grad_times.push_back(duration.count());
  }

  double weight_grad_avg =
      std::accumulate(weight_grad_times.begin(), weight_grad_times.end(), 0.0) /
      weight_grad_times.size();
  double input_grad_avg = std::accumulate(input_grad_times.begin(), input_grad_times.end(), 0.0) /
                          input_grad_times.size();

  std::cout << "CPU weight_gradients average: " << weight_grad_avg << " ms" << std::endl;
  std::cout << "CPU input_gradients average:  " << input_grad_avg << " ms" << std::endl;
  std::cout << "CPU total average:            " << (weight_grad_avg + input_grad_avg) << " ms"
            << std::endl;
}

#ifdef USE_CUDA
void benchmark_cuda_conv2d_gradients(const BenchmarkConfig &config, int warmup_runs = 2,
                                     int bench_runs = 5) {
  std::cout << "\n--- CUDA Conv2D Gradients Benchmark ---" << std::endl;

  const size_t batch_size = config.batch_size;
  const size_t in_channels = config.channels;
  const size_t out_channels = config.channels * 2; // Double channels for output
  const size_t input_h = config.height;
  const size_t input_w = config.width;

  const size_t output_h = (input_h + 2 * config.pad_h - config.kernel_h) / config.stride_h + 1;
  const size_t output_w = (input_w + 2 * config.pad_w - config.kernel_w) / config.stride_w + 1;

  const size_t kernel_size = in_channels * config.kernel_h * config.kernel_w;
  const size_t output_size = batch_size * output_h * output_w;

  // Allocate GPU buffers
  float *col_data, *gradient_data, *weight_data, *weight_grad_data, *col_grad_data;
  CUDA_CHECK(cudaMalloc(&col_data, kernel_size * output_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&gradient_data, out_channels * output_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&weight_data, out_channels * kernel_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&weight_grad_data, out_channels * kernel_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&col_grad_data, kernel_size * output_size * sizeof(float)));

  // Fill with random data (using CPU temporary buffers)
  std::vector<float> tmp_col(kernel_size * output_size);
  std::vector<float> tmp_grad(out_channels * output_size);
  std::vector<float> tmp_weight(out_channels * kernel_size);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &val : tmp_col)
    val = dist(gen);
  for (auto &val : tmp_grad)
    val = dist(gen);
  for (auto &val : tmp_weight)
    val = dist(gen);

  CUDA_CHECK(
      cudaMemcpy(col_data, tmp_col.data(), tmp_col.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gradient_data, tmp_grad.data(), tmp_grad.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(weight_data, tmp_weight.data(), tmp_weight.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Warmup
  for (int i = 0; i < warmup_runs; ++i) {
    auto weight_task = create_gpu_task(
        "benchmark_weight_grad", cuda::conv2d::compute_weight_gradients<float>, col_data,
        gradient_data, weight_grad_data, output_size, kernel_size, out_channels);
    weight_task->sync();
    auto input_task = create_gpu_task(
        "benchmark_input_grad", cuda::conv2d::compute_input_gradients<float>, gradient_data,
        weight_data, col_grad_data, output_size, kernel_size, out_channels);
    input_task->sync();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark compute_weight_gradients
  std::vector<double> weight_grad_times;
  for (int i = 0; i < bench_runs; ++i) {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    auto weight_task = create_gpu_task(
        "benchmark_weight_grad", cuda::conv2d::compute_weight_gradients<float>, col_data,
        gradient_data, weight_grad_data, output_size, kernel_size, out_channels);
    weight_task->sync();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    weight_grad_times.push_back(duration.count());
  }

  // Benchmark compute_input_gradients
  std::vector<double> input_grad_times;
  for (int i = 0; i < bench_runs; ++i) {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    auto input_task = create_gpu_task(
        "benchmark_input_grad", cuda::conv2d::compute_input_gradients<float>, gradient_data,
        weight_data, col_grad_data, output_size, kernel_size, out_channels);
    input_task->sync();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    input_grad_times.push_back(duration.count());
  }

  double weight_grad_avg =
      std::accumulate(weight_grad_times.begin(), weight_grad_times.end(), 0.0) /
      weight_grad_times.size();
  double input_grad_avg = std::accumulate(input_grad_times.begin(), input_grad_times.end(), 0.0) /
                          input_grad_times.size();

  std::cout << "CUDA weight_gradients average: " << weight_grad_avg << " ms" << std::endl;
  std::cout << "CUDA input_gradients average:  " << input_grad_avg << " ms" << std::endl;
  std::cout << "CUDA total average:            " << (weight_grad_avg + input_grad_avg) << " ms"
            << std::endl;

  // Cleanup
  CUDA_CHECK(cudaFree(col_data));
  CUDA_CHECK(cudaFree(gradient_data));
  CUDA_CHECK(cudaFree(weight_data));
  CUDA_CHECK(cudaFree(weight_grad_data));
  CUDA_CHECK(cudaFree(col_grad_data));
}

void run_conv2d_gradients_benchmark(const BenchmarkConfig &config, int warmup_runs = 2,
                                    int bench_runs = 5) {
  std::cout << "\n=== Conv2D Gradients Benchmark ===" << std::endl;
  print_config(config);

  const size_t out_channels = config.channels * 2;
  const size_t output_h =
      (config.height + 2 * config.pad_h - config.kernel_h) / config.stride_h + 1;
  const size_t output_w = (config.width + 2 * config.pad_w - config.kernel_w) / config.stride_w + 1;

  std::cout << "  Output channels: " << out_channels << std::endl;
  std::cout << "  Matrix dims: gradient(" << out_channels << "x"
            << (config.batch_size * output_h * output_w) << ") × col_data("
            << (config.channels * config.kernel_h * config.kernel_w) << "x"
            << (config.batch_size * output_h * output_w) << ")^T" << std::endl;

  // Run CPU benchmark
  benchmark_cpu_conv2d_gradients(config, warmup_runs, bench_runs);

  // Run CUDA benchmark
  benchmark_cuda_conv2d_gradients(config, warmup_runs, bench_runs);

  std::cout << std::endl;
}
#else
void run_conv2d_gradients_benchmark(const BenchmarkConfig &config, int warmup_runs = 2,
                                    int bench_runs = 5) {
  std::cout << "\n=== Conv2D Gradients Benchmark ===" << std::endl;
  print_config(config);

  const size_t out_channels = config.channels * 2;
  const size_t output_h =
      (config.height + 2 * config.pad_h - config.kernel_h) / config.stride_h + 1;
  const size_t output_w = (config.width + 2 * config.pad_w - config.kernel_w) / config.stride_w + 1;

  std::cout << "  Output channels: " << out_channels << std::endl;
  std::cout << "  Matrix dims: gradient(" << out_channels << "x"
            << (config.batch_size * output_h * output_w) << ") × col_data("
            << (config.channels * config.kernel_h * config.kernel_w) << "x"
            << (config.batch_size * output_h * output_w) << ")^T" << std::endl;

  // Run CPU benchmark only
  benchmark_cpu_conv2d_gradients(config, warmup_runs, bench_runs);

  std::cout << "\nNote: CUDA benchmarks skipped (built without CUDA support)" << std::endl;
  std::cout << std::endl;
}
#endif

int main() {
  std::cout << "TNN Tensor Operations Benchmark: CPU vs CUDA" << std::endl;
  print_separator();

#ifdef USE_CUDA
  cudaDeviceProp prop;
  int device_count;
  cudaGetDeviceCount(&device_count);
  std::cout << "\nCUDA Device Information:" << std::endl;
  std::cout << "Number of GPUs: " << device_count << std::endl;
  if (device_count > 0) {
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU 0: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0)
              << " GB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
  }
#else
  std::cout << "\nBuilt without CUDA support - CPU benchmarks only" << std::endl;
#endif

  print_separator();

  auto configs = get_benchmark_configs();

  std::cout << "\n### Part 1: im2col/col2im Operations ###\n" << std::endl;
  for (const auto &config : configs) {
    run_comparison_benchmark(config, 2, 5);
  }

  print_separator();
  std::cout << "\n### Part 2: Layout Transform Operations (nchw_to_cnhw / cnhw_to_nchw) ###\n"
            << std::endl;

  // Use simpler configs for layout transforms (no kernel/stride/pad params needed)
  std::vector<BenchmarkConfig> layout_configs = {
      {32, 64, 56, 56, 0, 0, 0, 0, 0, 0, "Small (32x64x56x56)"},
      {32, 128, 28, 28, 0, 0, 0, 0, 0, 0, "Medium (32x128x28x28)"},
      {64, 256, 112, 112, 0, 0, 0, 0, 0, 0, "Large (64x256x112x112)"},
      {8, 256, 224, 224, 0, 0, 0, 0, 0, 0, "Very Large (8x256x224x224)"},
  };
  for (const auto &config : layout_configs) {
    run_layout_transform_benchmark(config, 2, 5);
  }

  print_separator();
  std::cout << "\n### Part 3: Conv2D Gradient Operations (compute_weight_gradients / "
               "compute_input_gradients) ###\n"
            << std::endl;

  for (const auto &config : configs) {
    run_conv2d_gradients_benchmark(config, 2, 5);
  }

  print_separator();
  std::cout << "All benchmarks completed!" << std::endl;

  return 0;
}
