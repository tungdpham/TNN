/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "device/device_manager.hpp"
#include "device/dptr.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/cpu/maxpool_nchw_ops.hpp"
#include "nn/layers_impl/cuda/maxpool_nchw_ops.hpp"

using namespace tnn;

#ifdef USE_CUDA

class CUDAMaxPoolOpsTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {
    DeviceManager &manager = DeviceManager::getInstance();
    std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

    has_gpu_ = false;
    for (const std::string &id : device_ids) {
      const Device &device = manager.getDevice(id);
      if (device.device_type() == DeviceType::GPU) {
        has_gpu_ = true;
        break;
      }
    }

    if (!has_gpu_) {
      GTEST_SKIP() << "No GPU device available, skipping CUDA maxpool ops tests";
    }
  }

  void TearDown() override {}

  static void TearDownTestSuite() {}

  void compareArrays(const std::vector<float> &expected, const std::vector<float> &actual,
                     float tolerance = 1e-4f) {
    ASSERT_EQ(expected.size(), actual.size())
        << "Array sizes don't match: expected " << expected.size() << ", got " << actual.size();

    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance)
          << "Mismatch at index " << i << ": expected " << expected[i] << ", got " << actual[i];
    }
  }

  void compareMasks(const dptr &expected, const dptr &actual, size_t size) {
    std::vector<size_t> expected_host(size);
    std::vector<size_t> actual_host(size);

    if (expected.device_type() == DeviceType::CPU) {
      std::memcpy(expected_host.data(), expected.get<size_t>(), size * sizeof(size_t));
    } else {
      expected.getDevice().copyToHost(expected_host.data(), expected.get<size_t>(),
                                      size * sizeof(size_t));
    }

    if (actual.device_type() == DeviceType::CPU) {
      std::memcpy(actual_host.data(), actual.get<size_t>(), size * sizeof(size_t));
    } else {
      actual.getDevice().copyToHost(actual_host.data(), actual.get<size_t>(),
                                    size * sizeof(size_t));
    }

    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(expected_host[i], actual_host[i]) << "Mask mismatch at index " << i << ": expected "
                                                  << expected_host[i] << ", got " << actual_host[i];
    }
  }

  bool has_gpu_;
};

TEST_F(CUDAMaxPoolOpsTest, MaxPoolForwardBasic) {
  const size_t batch_size = 1;
  const size_t channels = 1;
  const size_t input_h = 4;
  const size_t input_w = 4;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Device &cpu_device = getHost();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  dptr cpu_mask = make_dptr_t<size_t>(cpu_device, mask_size);
  cpu::maxpool_nchw::compute_max_pool_forward<float>(
      input_data.data(), cpu_output.data(), batch_size, channels, input_h, input_w, output_h,
      output_w, pool_h, pool_w, stride_h, stride_w, 0, 0, cpu_mask.get<size_t>());

  dptr gpu_input = make_dptr_t<float>(getGPU(), input_data.size());
  dptr gpu_output = make_dptr_t<float>(getGPU(), batch_size * channels * output_h * output_w);

  getGPU().copyToDevice(gpu_input.get<float>(), input_data.data(),
                        input_data.size() * sizeof(float));

  dptr gpu_mask = make_dptr_t<size_t>(getGPU(), mask_size);
  auto gpu_task = create_cuda_task(
      defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_forward<float>,
      gpu_input.get<float>(), gpu_output.get<float>(), batch_size, channels, input_h, input_w,
      output_h, output_w, pool_h, pool_w, stride_h, stride_w, 0, 0, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_task->sync()) << "GPU maxpool forward task failed";

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  getGPU().copyToHost(gpu_output_cpu.data(), gpu_output.get<float>(),
                      (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask, mask_size);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolForwardMultiChannel) {
  const size_t batch_size = 2;
  const size_t channels = 3;
  const size_t input_h = 6;
  const size_t input_w = 6;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) + 1) * 0.1f;
  }

  const Device &cpu_device = getHost();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  dptr cpu_mask = make_dptr_t<size_t>(cpu_device, mask_size);
  cpu::maxpool_nchw::compute_max_pool_forward<float>(
      input_data.data(), cpu_output.data(), batch_size, channels, input_h, input_w, output_h,
      output_w, pool_h, pool_w, stride_h, stride_w, 0, 0, cpu_mask.get<size_t>());

  dptr gpu_input = make_dptr_t<float>(getGPU(), input_data.size());
  dptr gpu_output = make_dptr_t<float>(getGPU(), batch_size * channels * output_h * output_w);

  getGPU().copyToDevice(gpu_input.get<float>(), input_data.data(),
                        input_data.size() * sizeof(float));

  dptr gpu_mask = make_dptr_t<size_t>(getGPU(), mask_size);
  auto gpu_task = create_cuda_task(
      defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_forward<float>,
      gpu_input.get<float>(), gpu_output.get<float>(), batch_size, channels, input_h, input_w,
      output_h, output_w, pool_h, pool_w, stride_h, stride_w, 0, 0, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_task->sync()) << "GPU maxpool forward task failed";

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  getGPU().copyToHost(gpu_output_cpu.data(), gpu_output.get<float>(),
                      (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask, mask_size);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolForwardLargePool) {
  const size_t batch_size = 1;
  const size_t channels = 2;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t pool_h = 3;
  const size_t pool_w = 3;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>((i * 7) % 100) * 0.1f;
  }

  const Device &cpu_device = getHost();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  dptr cpu_mask = make_dptr_t<size_t>(cpu_device, mask_size);
  auto cpu_task = create_cpu_task(
      defaultFlowHandle, cpu::maxpool_nchw::compute_max_pool_forward<float>, input_data.data(),
      cpu_output.data(), batch_size, channels, input_h, input_w, output_h, output_w, pool_h, pool_w,
      stride_h, stride_w, 0, 0, cpu_mask.get<size_t>());
  ASSERT_FALSE(cpu_task->sync()) << "CPU maxpool forward task failed";

  dptr gpu_input = make_dptr_t<float>(getGPU(), input_data.size());
  dptr gpu_output = make_dptr_t<float>(getGPU(), batch_size * channels * output_h * output_w);

  getGPU().copyToDevice(gpu_input.get<float>(), input_data.data(),
                        input_data.size() * sizeof(float));

  dptr gpu_mask = make_dptr_t<size_t>(getGPU(), mask_size);
  auto gpu_task = create_cuda_task(
      defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_forward<float>,
      gpu_input.get<float>(), gpu_output.get<float>(), batch_size, channels, input_h, input_w,
      output_h, output_w, pool_h, pool_w, stride_h, stride_w, 0, 0, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_task->sync()) << "GPU maxpool forward task failed";

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  getGPU().copyToHost(gpu_output_cpu.data(), gpu_output.get<float>(),
                      (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask, mask_size);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolForwardNonSquare) {
  const size_t batch_size = 1;
  const size_t channels = 1;
  const size_t input_h = 8;
  const size_t input_w = 12;
  const size_t pool_h = 2;
  const size_t pool_w = 3;
  const size_t stride_h = 2;
  const size_t stride_w = 3;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Device &cpu_device = getHost();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  std::vector<float> cpu_output(batch_size * channels * output_h * output_w);
  dptr cpu_mask = make_dptr_t<size_t>(cpu_device, mask_size);
  auto cpu_task = create_cpu_task(
      defaultFlowHandle, cpu::maxpool_nchw::compute_max_pool_forward<float>, input_data.data(),
      cpu_output.data(), batch_size, channels, input_h, input_w, output_h, output_w, pool_h, pool_w,
      stride_h, stride_w, 0, 0, cpu_mask.get<size_t>());
  ASSERT_FALSE(cpu_task->sync()) << "CPU maxpool forward task failed";

  dptr gpu_input = make_dptr_t<float>(getGPU(), input_data.size());
  dptr gpu_output = make_dptr_t<float>(getGPU(), batch_size * channels * output_h * output_w);

  getGPU().copyToDevice(gpu_input.get<float>(), input_data.data(),
                        input_data.size() * sizeof(float));

  dptr gpu_mask = make_dptr_t<size_t>(getGPU(), mask_size);
  auto gpu_task = create_cuda_task(
      defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_forward<float>,
      gpu_input.get<float>(), gpu_output.get<float>(), batch_size, channels, input_h, input_w,
      output_h, output_w, pool_h, pool_w, stride_h, stride_w, 0, 0, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_task->sync()) << "GPU maxpool forward task failed";

  std::vector<float> gpu_output_cpu(batch_size * channels * output_h * output_w);
  getGPU().copyToHost(gpu_output_cpu.data(), gpu_output.get<float>(),
                      (batch_size * channels * output_h * output_w) * sizeof(float));

  compareArrays(cpu_output, gpu_output_cpu);
  compareMasks(cpu_mask, gpu_mask, mask_size);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolBackwardBasic) {
  const size_t batch_size = 1;
  const size_t channels = 1;
  const size_t input_h = 4;
  const size_t input_w = 4;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Device &cpu_device = getHost();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  dptr cpu_mask = make_dptr_t<size_t>(cpu_device, mask_size);
  cpu::maxpool_nchw::compute_max_pool_forward<float>(
      input_data.data(), forward_output.data(), batch_size, channels, input_h, input_w, output_h,
      output_w, pool_h, pool_w, stride_h, stride_w, 0, 0, cpu_mask.get<size_t>());

  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i + 1) * 0.1f;
  }

  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  cpu::maxpool_nchw::compute_max_pool_backward<float>(gradient_data.data(), cpu_grad_input.data(),
                                                      batch_size, channels, output_h, output_w,
                                                      cpu_mask.get<size_t>());

  dptr gpu_input = make_dptr_t<float>(getGPU(), input_data.size());
  dptr gpu_forward_output = make_dptr_t<float>(getGPU(), mask_size);
  dptr gpu_mask = make_dptr_t<size_t>(getGPU(), mask_size);

  getGPU().copyToDevice(gpu_input.get<float>(), input_data.data(),
                        input_data.size() * sizeof(float));

  auto gpu_forward_task =
      create_cuda_task(defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_forward<float>,
                       gpu_input.get<float>(), gpu_forward_output.get<float>(), batch_size,
                       channels, input_h, input_w, output_h, output_w, pool_h, pool_w, stride_h,
                       stride_w, 0, 0, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_forward_task->sync()) << "GPU maxpool forward task failed";

  dptr gpu_gradient = make_dptr_t<float>(getGPU(), gradient_data.size());
  dptr gpu_grad_input = make_dptr_t<float>(getGPU(), batch_size * channels * input_h * input_w);

  getGPU().copyToDevice(gpu_gradient.get<float>(), gradient_data.data(),
                        gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  getGPU().copyToDevice(gpu_grad_input.get<float>(), zero_grad.data(),
                        zero_grad.size() * sizeof(float));

  auto gpu_backward_task =
      create_cuda_task(defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_backward<float>,
                       gpu_gradient.get<float>(), gpu_grad_input.get<float>(), batch_size, channels,
                       output_h, output_w, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_backward_task->sync()) << "GPU maxpool backward task failed";

  std::vector<float> gpu_grad_input_cpu(batch_size * channels * input_h * input_w);
  getGPU().copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get<float>(),
                      (batch_size * channels * input_h * input_w) * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolBackwardMultiChannel) {
  const size_t batch_size = 2;
  const size_t channels = 3;
  const size_t input_h = 6;
  const size_t input_w = 6;
  const size_t pool_h = 2;
  const size_t pool_w = 2;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>((i % 100) + 1) * 0.1f;
  }

  const Device &cpu_device = getHost();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  dptr cpu_mask = make_dptr_t<size_t>(cpu_device, mask_size);
  auto cpu_forward_task = create_cpu_task(
      defaultFlowHandle, cpu::maxpool_nchw::compute_max_pool_forward<float>, input_data.data(),
      forward_output.data(), batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
      pool_w, stride_h, stride_w, 0, 0, cpu_mask.get<size_t>());
  ASSERT_FALSE(cpu_forward_task->sync()) << "CPU maxpool forward task failed";

  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>((i % 50) + 1) * 0.05f;
  }

  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  auto cpu_backward_task = create_cpu_task(
      defaultFlowHandle, cpu::maxpool_nchw::compute_max_pool_backward<float>, gradient_data.data(),
      cpu_grad_input.data(), batch_size, channels, output_h, output_w, cpu_mask.get<size_t>());
  ASSERT_FALSE(cpu_backward_task->sync()) << "CPU maxpool backward task failed";

  dptr gpu_input = make_dptr_t<float>(getGPU(), input_data.size());
  dptr gpu_forward_output = make_dptr_t<float>(getGPU(), mask_size);
  dptr gpu_mask = make_dptr_t<size_t>(getGPU(), mask_size);

  getGPU().copyToDevice(gpu_input.get<float>(), input_data.data(),
                        input_data.size() * sizeof(float));

  auto gpu_forward_task =
      create_cuda_task(defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_forward<float>,
                       gpu_input.get<float>(), gpu_forward_output.get<float>(), batch_size,
                       channels, input_h, input_w, output_h, output_w, pool_h, pool_w, stride_h,
                       stride_w, 0, 0, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_forward_task->sync()) << "GPU maxpool forward task failed";

  dptr gpu_gradient = make_dptr_t<float>(getGPU(), gradient_data.size());
  dptr gpu_grad_input = make_dptr_t<float>(getGPU(), batch_size * channels * input_h * input_w);

  getGPU().copyToDevice(gpu_gradient.get<float>(), gradient_data.data(),
                        gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  getGPU().copyToDevice(gpu_grad_input.get<float>(), zero_grad.data(),
                        zero_grad.size() * sizeof(float));

  auto gpu_backward_task =
      create_cuda_task(defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_backward<float>,
                       gpu_gradient.get<float>(), gpu_grad_input.get<float>(), batch_size, channels,
                       output_h, output_w, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_backward_task->sync()) << "GPU maxpool backward task failed";

  std::vector<float> gpu_grad_input_cpu(batch_size * channels * input_h * input_w);
  getGPU().copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get<float>(),
                      (batch_size * channels * input_h * input_w) * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolBackwardLargePool) {
  const size_t batch_size = 1;
  const size_t channels = 2;
  const size_t input_h = 8;
  const size_t input_w = 8;
  const size_t pool_h = 3;
  const size_t pool_w = 3;
  const size_t stride_h = 2;
  const size_t stride_w = 2;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>((i * 7) % 100) * 0.1f;
  }

  const Device &cpu_device = getHost();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  dptr cpu_mask = make_dptr_t<size_t>(cpu_device, mask_size);
  auto cpu_forward_task = create_cpu_task(
      defaultFlowHandle, cpu::maxpool_nchw::compute_max_pool_forward<float>, input_data.data(),
      forward_output.data(), batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
      pool_w, stride_h, stride_w, 0, 0, cpu_mask.get<size_t>());
  ASSERT_FALSE(cpu_forward_task->sync()) << "CPU maxpool forward task failed";

  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>((i + 1)) * 0.2f;
  }

  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  auto cpu_backward_task = create_cpu_task(
      defaultFlowHandle, cpu::maxpool_nchw::compute_max_pool_backward<float>, gradient_data.data(),
      cpu_grad_input.data(), batch_size, channels, output_h, output_w, cpu_mask.get<size_t>());
  ASSERT_FALSE(cpu_backward_task->sync()) << "CPU maxpool backward task failed";

  dptr gpu_input = make_dptr_t<float>(getGPU(), input_data.size());
  dptr gpu_forward_output = make_dptr_t<float>(getGPU(), mask_size);
  dptr gpu_mask = make_dptr_t<size_t>(getGPU(), mask_size);

  getGPU().copyToDevice(gpu_input.get<float>(), input_data.data(),
                        input_data.size() * sizeof(float));

  auto gpu_forward_task =
      create_cuda_task(defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_forward<float>,
                       gpu_input.get<float>(), gpu_forward_output.get<float>(), batch_size,
                       channels, input_h, input_w, output_h, output_w, pool_h, pool_w, stride_h,
                       stride_w, 0, 0, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_forward_task->sync()) << "GPU maxpool forward task failed";

  dptr gpu_gradient = make_dptr_t<float>(getGPU(), gradient_data.size());
  dptr gpu_grad_input = make_dptr_t<float>(getGPU(), batch_size * channels * input_h * input_w);

  getGPU().copyToDevice(gpu_gradient.get<float>(), gradient_data.data(),
                        gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  getGPU().copyToDevice(gpu_grad_input.get<float>(), zero_grad.data(),
                        zero_grad.size() * sizeof(float));

  auto gpu_backward_task =
      create_cuda_task(defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_backward<float>,
                       gpu_gradient.get<float>(), gpu_grad_input.get<float>(), batch_size, channels,
                       output_h, output_w, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_backward_task->sync()) << "GPU maxpool backward task failed";

  std::vector<float> gpu_grad_input_cpu(batch_size * channels * input_h * input_w);
  getGPU().copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get<float>(),
                      (batch_size * channels * input_h * input_w) * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

TEST_F(CUDAMaxPoolOpsTest, MaxPoolBackwardNonSquare) {
  const size_t batch_size = 1;
  const size_t channels = 1;
  const size_t input_h = 8;
  const size_t input_w = 12;
  const size_t pool_h = 2;
  const size_t pool_w = 3;
  const size_t stride_h = 2;
  const size_t stride_w = 3;
  const size_t output_h = (input_h - pool_h) / stride_h + 1;
  const size_t output_w = (input_w - pool_w) / stride_w + 1;

  std::vector<float> input_data(batch_size * channels * input_h * input_w);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = static_cast<float>(i + 1);
  }

  const Device &cpu_device = getHost();
  const size_t mask_size = batch_size * channels * output_h * output_w;

  std::vector<float> forward_output(batch_size * channels * output_h * output_w);
  dptr cpu_mask = make_dptr_t<size_t>(cpu_device, mask_size);
  auto cpu_forward_task = create_cpu_task(
      defaultFlowHandle, cpu::maxpool_nchw::compute_max_pool_forward<float>, input_data.data(),
      forward_output.data(), batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
      pool_w, stride_h, stride_w, 0, 0, cpu_mask.get<size_t>());
  ASSERT_FALSE(cpu_forward_task->sync()) << "CPU maxpool forward task failed";

  std::vector<float> gradient_data(batch_size * channels * output_h * output_w);
  for (size_t i = 0; i < gradient_data.size(); ++i) {
    gradient_data[i] = static_cast<float>(i + 1) * 0.15f;
  }

  std::vector<float> cpu_grad_input(batch_size * channels * input_h * input_w, 0.0f);
  auto cpu_backward_task = create_cpu_task(
      defaultFlowHandle, cpu::maxpool_nchw::compute_max_pool_backward<float>, gradient_data.data(),
      cpu_grad_input.data(), batch_size, channels, output_h, output_w, cpu_mask.get<size_t>());
  ASSERT_FALSE(cpu_backward_task->sync()) << "CPU maxpool backward task failed";

  dptr gpu_input = make_dptr_t<float>(getGPU(), input_data.size());
  dptr gpu_forward_output = make_dptr_t<float>(getGPU(), mask_size);
  dptr gpu_mask = make_dptr_t<size_t>(getGPU(), mask_size);

  getGPU().copyToDevice(gpu_input.get<float>(), input_data.data(),
                        input_data.size() * sizeof(float));

  auto gpu_forward_task =
      create_cuda_task(defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_forward<float>,
                       gpu_input.get<float>(), gpu_forward_output.get<float>(), batch_size,
                       channels, input_h, input_w, output_h, output_w, pool_h, pool_w, stride_h,
                       stride_w, 0, 0, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_forward_task->sync()) << "GPU maxpool forward task failed";

  dptr gpu_gradient = make_dptr_t<float>(getGPU(), gradient_data.size());
  dptr gpu_grad_input = make_dptr_t<float>(getGPU(), batch_size * channels * input_h * input_w);

  getGPU().copyToDevice(gpu_gradient.get<float>(), gradient_data.data(),
                        gradient_data.size() * sizeof(float));

  std::vector<float> zero_grad(batch_size * channels * input_h * input_w, 0.0f);
  getGPU().copyToDevice(gpu_grad_input.get<float>(), zero_grad.data(),
                        zero_grad.size() * sizeof(float));

  auto gpu_backward_task =
      create_cuda_task(defaultFlowHandle, cuda::maxpool_nchw::compute_max_pool_backward<float>,
                       gpu_gradient.get<float>(), gpu_grad_input.get<float>(), batch_size, channels,
                       output_h, output_w, gpu_mask.get<size_t>());
  ASSERT_FALSE(gpu_backward_task->sync()) << "GPU maxpool backward task failed";

  std::vector<float> gpu_grad_input_cpu(batch_size * channels * input_h * input_w);
  getGPU().copyToHost(gpu_grad_input_cpu.data(), gpu_grad_input.get<float>(),
                      (batch_size * channels * input_h * input_w) * sizeof(float));

  compareArrays(cpu_grad_input, gpu_grad_input_cpu);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
