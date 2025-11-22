/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "tensor/cpu/tensor_ops.hpp"
#include "tensor/cuda/tensor_ops.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace tnn;

#ifdef USE_CUDA

// Test fixture for GPU tensor operations
class GPUTensorOpsTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // Initialize devices once for all tests in this suite
    initializeDefaultDevices();
  }

  void SetUp() override {
    DeviceManager &manager = DeviceManager::getInstance();
    std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

    // Find GPU device
    has_gpu_ = false;
    for (const std::string &id : device_ids) {
      const Device &device = manager.getDevice(id);
      if (device.getDeviceType() == DeviceType::GPU) {
        gpu_device_ = &device;
        has_gpu_ = true;
        break;
      }
    }

    if (!has_gpu_) {
      GTEST_SKIP() << "No GPU device available, skipping GPU tensor ops tests";
    }
  }

  void TearDown() override {}

  static void TearDownTestSuite() {}

  // Helper function to compare tensors with tolerance
  template <typename T>
  void compareTensors(const Tensor<T, NCHW> &expected, const Tensor<T, NCHW> &actual,
                      T tolerance = static_cast<T>(1e-5)) {
    ASSERT_TRUE(expected.same_shape(actual))
        << "Tensors have different shapes. Expected: " << expected.shape_str()
        << ", Actual: " << actual.shape_str();

    Tensor<T, NCHW> expected_cpu = expected.is_on_cpu() ? expected.clone() : expected.to_cpu();
    Tensor<T, NCHW> actual_cpu = actual.is_on_cpu() ? actual.clone() : actual.to_cpu();

    for (size_t n = 0; n < expected_cpu.batch_size(); ++n) {
      for (size_t c = 0; c < expected_cpu.channels(); ++c) {
        for (size_t h = 0; h < expected_cpu.height(); ++h) {
          for (size_t w = 0; w < expected_cpu.width(); ++w) {
            T expected_val = expected_cpu(n, c, h, w);
            T actual_val = actual_cpu(n, c, h, w);
            EXPECT_NEAR(expected_val, actual_val, tolerance)
                << "Mismatch at position [" << n << "," << c << "," << h << "," << w
                << "]. Expected: " << expected_val << ", Got: " << actual_val;
          }
        }
      }
    }
  }

  bool has_gpu_;
  const Device *gpu_device_;
};

// ==================== Padding Tests ====================

TEST_F(GPUTensorOpsTest, PadBasic) {
  // Create a small CPU tensor with known values
  Tensor<float, NCHW> cpu_tensor({1, 1, 3, 3});
  for (size_t i = 0; i < 9; ++i) {
    cpu_tensor.data()[i] = static_cast<float>(i + 1);
  }

  // Pad using CPU implementation
  Tensor<float, NCHW> cpu_padded({1, 1, 5, 5});
  cpu::pad(cpu_tensor, cpu_padded, 1, 1, 0.0f);

  // Transfer to GPU and pad
  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_padded({1, 1, 5, 5}, gpu_device_);
  cuda::pad(gpu_tensor, gpu_padded, 1, 1, 0.0f);

  // Compare results
  compareTensors(cpu_padded, gpu_padded);
}

TEST_F(GPUTensorOpsTest, PadMultiChannel) {
  // Test with multiple channels
  Tensor<float, NCHW> cpu_tensor({2, 3, 4, 4});
  cpu_tensor.fill_random_uniform(10.0f);

  Tensor<float, NCHW> cpu_padded({2, 3, 8, 8});
  cpu::pad(cpu_tensor, cpu_padded, 2, 2, -1.0f);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_padded({2, 3, 8, 8}, gpu_device_);
  cuda::pad(gpu_tensor, gpu_padded, 2, 2, -1.0f);

  compareTensors(cpu_padded, gpu_padded);
}

TEST_F(GPUTensorOpsTest, PadAsymmetric) {
  Tensor<float, NCHW> cpu_tensor({1, 2, 5, 7});
  cpu_tensor.fill_random_uniform(5.0f);

  Tensor<float, NCHW> cpu_padded({1, 2, 11, 9});
  cpu::pad(cpu_tensor, cpu_padded, 3, 1, 2.5f);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_padded({1, 2, 11, 9}, gpu_device_);
  cuda::pad(gpu_tensor, gpu_padded, 3, 1, 2.5f);

  compareTensors(cpu_padded, gpu_padded);
}

// ==================== Unpadding Tests ====================

TEST_F(GPUTensorOpsTest, UnpadBasic) {
  Tensor<float, NCHW> cpu_tensor({1, 1, 5, 5});
  cpu_tensor.fill_random_uniform(10.0f);

  Tensor<float, NCHW> cpu_unpadded({1, 1, 3, 3});
  cpu::unpad(cpu_tensor, cpu_unpadded, 1, 1);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_unpadded({1, 1, 3, 3}, gpu_device_);
  cuda::unpad(gpu_tensor, gpu_unpadded, 1, 1);

  compareTensors(cpu_unpadded, gpu_unpadded);
}

TEST_F(GPUTensorOpsTest, UnpadMultiChannel) {
  Tensor<float, NCHW> cpu_tensor({2, 3, 8, 8});
  cpu_tensor.fill_random_uniform(15.0f);

  Tensor<float, NCHW> cpu_unpadded({2, 3, 4, 4});
  cpu::unpad(cpu_tensor, cpu_unpadded, 2, 2);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_unpadded({2, 3, 4, 4}, gpu_device_);
  cuda::unpad(gpu_tensor, gpu_unpadded, 2, 2);

  compareTensors(cpu_unpadded, gpu_unpadded);
}

TEST_F(GPUTensorOpsTest, PadUnpadRoundTrip) {
  // Test that pad -> unpad returns original tensor
  Tensor<float, NCHW> cpu_original({1, 2, 4, 4});
  cpu_original.fill_random_uniform(8.0f);

  // CPU version
  Tensor<float, NCHW> cpu_padded({1, 2, 8, 8});
  cpu::pad(cpu_original, cpu_padded, 2, 2, 0.0f);
  Tensor<float, NCHW> cpu_restored({1, 2, 4, 4});
  cpu::unpad(cpu_padded, cpu_restored, 2, 2);

  // GPU version
  Tensor<float, NCHW> gpu_original = cpu_original.to_gpu();
  Tensor<float, NCHW> gpu_padded({1, 2, 8, 8}, gpu_device_);
  cuda::pad(gpu_original, gpu_padded, 2, 2, 0.0f);
  Tensor<float, NCHW> gpu_restored({1, 2, 4, 4}, gpu_device_);
  cuda::unpad(gpu_padded, gpu_restored, 2, 2);

  compareTensors(cpu_original, cpu_restored);
  compareTensors(cpu_original, gpu_restored);
  compareTensors(cpu_restored, gpu_restored);
}

// ==================== Crop Tests ====================

TEST_F(GPUTensorOpsTest, CropBasic) {
  Tensor<float, NCHW> cpu_tensor({1, 1, 5, 5});
  for (size_t i = 0; i < 25; ++i) {
    cpu_tensor.data()[i] = static_cast<float>(i);
  }

  Tensor<float, NCHW> cpu_cropped({1, 1, 3, 3});
  cpu::crop(cpu_tensor, cpu_cropped, 1, 1, 3, 3);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_cropped({1, 1, 3, 3}, gpu_device_);
  cuda::crop(gpu_tensor, gpu_cropped, 1, 1, 3, 3);

  compareTensors(cpu_cropped, gpu_cropped);
}

TEST_F(GPUTensorOpsTest, CropMultiChannel) {
  Tensor<float, NCHW> cpu_tensor({2, 3, 10, 10});
  cpu_tensor.fill_random_uniform(20.0f);

  Tensor<float, NCHW> cpu_cropped({2, 3, 6, 6});
  cpu::crop(cpu_tensor, cpu_cropped, 2, 3, 7, 8);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_cropped({2, 3, 6, 6}, gpu_device_);
  cuda::crop(gpu_tensor, gpu_cropped, 2, 3, 7, 8);

  compareTensors(cpu_cropped, gpu_cropped);
}

TEST_F(GPUTensorOpsTest, CropCorner) {
  Tensor<float, NCHW> cpu_tensor({1, 2, 8, 8});
  cpu_tensor.fill_random_uniform(12.0f);

  // Crop top-left corner
  Tensor<float, NCHW> cpu_cropped({1, 2, 4, 4});
  cpu::crop(cpu_tensor, cpu_cropped, 0, 0, 3, 3);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_cropped({1, 2, 4, 4}, gpu_device_);
  cuda::crop(gpu_tensor, gpu_cropped, 0, 0, 3, 3);

  compareTensors(cpu_cropped, gpu_cropped);
}

TEST_F(GPUTensorOpsTest, CropBottomRight) {
  Tensor<float, NCHW> cpu_tensor({1, 1, 6, 6});
  cpu_tensor.fill_random_uniform(10.0f);

  // Crop bottom-right area
  Tensor<float, NCHW> cpu_cropped({1, 1, 3, 3});
  cpu::crop(cpu_tensor, cpu_cropped, 3, 3, 5, 5);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_cropped({1, 1, 3, 3}, gpu_device_);
  cuda::crop(gpu_tensor, gpu_cropped, 3, 3, 5, 5);

  compareTensors(cpu_cropped, gpu_cropped);
}

// ==================== Slice Batch Tests ====================

TEST_F(GPUTensorOpsTest, SliceBatchBasic) {
  Tensor<float, NCHW> cpu_tensor({4, 2, 3, 3});
  cpu_tensor.fill_random_uniform(15.0f);

  Tensor<float, NCHW> cpu_sliced({2, 2, 3, 3});
  cpu::slice_batch(cpu_tensor, cpu_sliced, 1, 3);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_sliced({2, 2, 3, 3}, gpu_device_);
  cuda::slice_batch(gpu_tensor, gpu_sliced, 1, 3);

  compareTensors(cpu_sliced, gpu_sliced);
}

TEST_F(GPUTensorOpsTest, SliceBatchSingle) {
  Tensor<float, NCHW> cpu_tensor({5, 3, 4, 4});
  cpu_tensor.fill_random_uniform(10.0f);

  // Extract single batch
  Tensor<float, NCHW> cpu_sliced({1, 3, 4, 4});
  cpu::slice_batch(cpu_tensor, cpu_sliced, 2, 3);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_sliced({1, 3, 4, 4}, gpu_device_);
  cuda::slice_batch(gpu_tensor, gpu_sliced, 2, 3);

  compareTensors(cpu_sliced, gpu_sliced);
}

TEST_F(GPUTensorOpsTest, SliceBatchFirstBatch) {
  Tensor<float, NCHW> cpu_tensor({3, 2, 5, 5});
  cpu_tensor.fill_random_uniform(8.0f);

  Tensor<float, NCHW> cpu_sliced({1, 2, 5, 5});
  cpu::slice_batch(cpu_tensor, cpu_sliced, 0, 1);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_sliced({1, 2, 5, 5}, gpu_device_);
  cuda::slice_batch(gpu_tensor, gpu_sliced, 0, 1);

  compareTensors(cpu_sliced, gpu_sliced);
}

// ==================== Slice Channels Tests ====================

TEST_F(GPUTensorOpsTest, SliceChannelsBasic) {
  Tensor<float, NCHW> cpu_tensor({2, 8, 4, 4});
  cpu_tensor.fill_random_uniform(12.0f);

  Tensor<float, NCHW> cpu_sliced({2, 4, 4, 4});
  cpu::slice_channels(cpu_tensor, cpu_sliced, 2, 5);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_sliced({2, 4, 4, 4}, gpu_device_);
  cuda::slice_channels(gpu_tensor, gpu_sliced, 2, 5);

  compareTensors(cpu_sliced, gpu_sliced);
}

TEST_F(GPUTensorOpsTest, SliceChannelsSingle) {
  Tensor<float, NCHW> cpu_tensor({1, 10, 6, 6});
  cpu_tensor.fill_random_uniform(15.0f);

  // Extract single channel
  Tensor<float, NCHW> cpu_sliced({1, 1, 6, 6});
  cpu::slice_channels(cpu_tensor, cpu_sliced, 5, 5);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_sliced({1, 1, 6, 6}, gpu_device_);
  cuda::slice_channels(gpu_tensor, gpu_sliced, 5, 5);

  compareTensors(cpu_sliced, gpu_sliced);
}

TEST_F(GPUTensorOpsTest, SliceChannelsFirstThree) {
  Tensor<float, NCHW> cpu_tensor({2, 6, 3, 3});
  cpu_tensor.fill_random_uniform(9.0f);

  Tensor<float, NCHW> cpu_sliced({2, 3, 3, 3});
  cpu::slice_channels(cpu_tensor, cpu_sliced, 0, 2);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_sliced({2, 3, 3, 3}, gpu_device_);
  cuda::slice_channels(gpu_tensor, gpu_sliced, 0, 2);

  compareTensors(cpu_sliced, gpu_sliced);
}

// ==================== Split Tests ====================

TEST_F(GPUTensorOpsTest, SplitBasic) {
  Tensor<float, NCHW> cpu_tensor({4, 2, 3, 3});
  cpu_tensor.fill_random_uniform(10.0f);

  std::vector<Tensor<float, NCHW>> cpu_splits{Tensor<float, NCHW>({2, 2, 3, 3}),
                                              Tensor<float, NCHW>({2, 2, 3, 3})};
  cpu::split(cpu_tensor, cpu_splits, 2);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  std::vector<Tensor<float, NCHW>> gpu_splits{Tensor<float, NCHW>({2, 2, 3, 3}, gpu_device_),
                                              Tensor<float, NCHW>({2, 2, 3, 3}, gpu_device_)};
  cuda::split(gpu_tensor, gpu_splits, 2);

  ASSERT_EQ(cpu_splits.size(), gpu_splits.size());

  for (size_t i = 0; i < cpu_splits.size(); ++i) {
    compareTensors(cpu_splits[i], gpu_splits[i]);
  }
}

TEST_F(GPUTensorOpsTest, SplitMultiple) {
  Tensor<float, NCHW> cpu_tensor({8, 3, 4, 4});
  cpu_tensor.fill_random_uniform(15.0f);

  std::vector<Tensor<float, NCHW>> cpu_splits = {
      Tensor<float, NCHW>({2, 3, 4, 4}), Tensor<float, NCHW>({2, 3, 4, 4}),
      Tensor<float, NCHW>({2, 3, 4, 4}), Tensor<float, NCHW>({2, 3, 4, 4})};
  cpu::split(cpu_tensor, cpu_splits, 4);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  std::vector<Tensor<float, NCHW>> gpu_splits = {Tensor<float, NCHW>({2, 3, 4, 4}, gpu_device_),
                                                 Tensor<float, NCHW>({2, 3, 4, 4}, gpu_device_),
                                                 Tensor<float, NCHW>({2, 3, 4, 4}, gpu_device_),
                                                 Tensor<float, NCHW>({2, 3, 4, 4}, gpu_device_)};
  cuda::split(gpu_tensor, gpu_splits, 4);

  ASSERT_EQ(cpu_splits.size(), gpu_splits.size());

  for (size_t i = 0; i < cpu_splits.size(); ++i) {
    compareTensors(cpu_splits[i], gpu_splits[i]);
  }
}

TEST_F(GPUTensorOpsTest, SplitSingleBatch) {
  Tensor<float, NCHW> cpu_tensor({6, 2, 5, 5});
  cpu_tensor.fill_random_uniform(12.0f);

  std::vector<Tensor<float, NCHW>> cpu_splits = {
      Tensor<float, NCHW>({1, 2, 5, 5}), Tensor<float, NCHW>({1, 2, 5, 5}),
      Tensor<float, NCHW>({1, 2, 5, 5}), Tensor<float, NCHW>({1, 2, 5, 5}),
      Tensor<float, NCHW>({1, 2, 5, 5}), Tensor<float, NCHW>({1, 2, 5, 5})};
  cpu::split(cpu_tensor, cpu_splits, 6);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  std::vector<Tensor<float, NCHW>> gpu_splits = {Tensor<float, NCHW>({1, 2, 5, 5}, gpu_device_),
                                                 Tensor<float, NCHW>({1, 2, 5, 5}, gpu_device_),
                                                 Tensor<float, NCHW>({1, 2, 5, 5}, gpu_device_),
                                                 Tensor<float, NCHW>({1, 2, 5, 5}, gpu_device_),
                                                 Tensor<float, NCHW>({1, 2, 5, 5}, gpu_device_),
                                                 Tensor<float, NCHW>({1, 2, 5, 5}, gpu_device_)};
  cuda::split(gpu_tensor, gpu_splits, 6);

  ASSERT_EQ(cpu_splits.size(), gpu_splits.size());

  for (size_t i = 0; i < cpu_splits.size(); ++i) {
    compareTensors(cpu_splits[i], gpu_splits[i]);
  }
}

// ==================== Softmax Tests ====================

TEST_F(GPUTensorOpsTest, SoftmaxBasic) {
  Tensor<float, NCHW> cpu_tensor({1, 3, 1, 1});
  cpu_tensor(0, 0, 0, 0) = 1.0f;
  cpu_tensor(0, 1, 0, 0) = 2.0f;
  cpu_tensor(0, 2, 0, 0) = 3.0f;

  Tensor<float, NCHW> cpu_copy = cpu_tensor.clone();
  cpu::apply_softmax(cpu_copy);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  cuda::apply_softmax(gpu_tensor);

  compareTensors(cpu_copy, gpu_tensor, 1e-4f);

  // Verify softmax properties: all values in [0,1] and sum to 1
  Tensor<float, NCHW> gpu_cpu = gpu_tensor.to_cpu();
  float sum = 0.0f;
  for (size_t c = 0; c < 3; ++c) {
    float val = gpu_cpu(0, c, 0, 0);
    EXPECT_GE(val, 0.0f);
    EXPECT_LE(val, 1.0f);
    sum += val;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(GPUTensorOpsTest, SoftmaxMultiBatch) {
  Tensor<float, NCHW> cpu_tensor({2, 4, 1, 1});
  cpu_tensor.fill_random_uniform(10.0f);

  Tensor<float, NCHW> cpu_copy = cpu_tensor.clone();
  cpu::apply_softmax(cpu_copy);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  cuda::apply_softmax(gpu_tensor);

  compareTensors(cpu_copy, gpu_tensor, 1e-4f);

  // Verify each batch sums to 1
  Tensor<float, NCHW> gpu_cpu = gpu_tensor.to_cpu();
  for (size_t n = 0; n < 2; ++n) {
    float sum = 0.0f;
    for (size_t c = 0; c < 4; ++c) {
      sum += gpu_cpu(n, c, 0, 0);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
  }
}

TEST_F(GPUTensorOpsTest, SoftmaxSpatial) {
  // Test softmax with spatial dimensions
  Tensor<float, NCHW> cpu_tensor({1, 5, 2, 2});
  cpu_tensor.fill_random_uniform(8.0f);

  Tensor<float, NCHW> cpu_copy = cpu_tensor.clone();
  cpu::apply_softmax(cpu_copy);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  cuda::apply_softmax(gpu_tensor);

  compareTensors(cpu_copy, gpu_tensor, 1e-4f);

  // Verify softmax over channels for each spatial location
  Tensor<float, NCHW> gpu_cpu = gpu_tensor.to_cpu();
  for (size_t h = 0; h < 2; ++h) {
    for (size_t w = 0; w < 2; ++w) {
      float sum = 0.0f;
      for (size_t c = 0; c < 5; ++c) {
        sum += gpu_cpu(0, c, h, w);
      }
      EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
  }
}

TEST_F(GPUTensorOpsTest, SoftmaxLargeChannels) {
  Tensor<float, NCHW> cpu_tensor({2, 64, 1, 1});
  cpu_tensor.fill_random_uniform(15.0f);

  Tensor<float, NCHW> cpu_copy = cpu_tensor.clone();
  cpu::apply_softmax(cpu_copy);

  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  cuda::apply_softmax(gpu_tensor);

  compareTensors(cpu_copy, gpu_tensor, 1e-4f);
}

// ==================== Im2col Tests ====================

TEST_F(GPUTensorOpsTest, Im2colBasicKernel3x3) {
  Tensor<float, NCHW> cpu_input({1, 1, 5, 5});
  for (size_t i = 0; i < 25; ++i) {
    cpu_input.data()[i] = static_cast<float>(i + 1);
  }

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 0, pad_w = 0;

  size_t output_h = (cpu_input.height() - kernel_h) / stride_h + 1;
  size_t output_w = (cpu_input.width() - kernel_w) / stride_w + 1;
  size_t col_size =
      cpu_input.batch_size() * cpu_input.channels() * kernel_h * kernel_w * output_h * output_w;

  // CPU version
  std::vector<float> cpu_col_data(col_size);
  cpu::im2col(cpu_input, cpu_col_data.data(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  // GPU version
  Tensor<float, NCHW> gpu_input = cpu_input.to_gpu();
  device_ptr<float[]> gpu_col_data = make_array_ptr<float[]>(gpu_device_, col_size);
  cuda::im2col(gpu_input, gpu_col_data.get(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  // Transfer GPU result to CPU for comparison
  std::vector<float> gpu_col_cpu(col_size);
  gpu_device_->copyToHost(gpu_col_cpu.data(), gpu_col_data.get(), col_size * sizeof(float));

  // Compare results
  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(cpu_col_data[i], gpu_col_cpu[i], 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUTensorOpsTest, Im2colWithPadding) {
  Tensor<float, NCHW> cpu_input({1, 2, 4, 4});
  cpu_input.fill_random_uniform(10.0f);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 1, pad_w = 1;

  size_t padded_h = cpu_input.height() + 2 * pad_h;
  size_t padded_w = cpu_input.width() + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;
  size_t col_size =
      cpu_input.batch_size() * cpu_input.channels() * kernel_h * kernel_w * output_h * output_w;

  // CPU version
  std::vector<float> cpu_col_data(col_size);
  cpu::im2col(cpu_input, cpu_col_data.data(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  // GPU version
  Tensor<float, NCHW> gpu_input = cpu_input.to_gpu();
  device_ptr<float[]> gpu_col_data = make_array_ptr<float[]>(gpu_device_, col_size);
  cuda::im2col(gpu_input, gpu_col_data.get(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  // Transfer GPU result to CPU for comparison
  std::vector<float> gpu_col_cpu(col_size);
  gpu_device_->copyToHost(gpu_col_cpu.data(), gpu_col_data.get(), col_size * sizeof(float));

  // Compare results
  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(cpu_col_data[i], gpu_col_cpu[i], 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUTensorOpsTest, Im2colWithStride) {
  Tensor<float, NCHW> cpu_input({1, 1, 8, 8});
  cpu_input.fill_random_uniform(15.0f);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 2, stride_w = 2;
  size_t pad_h = 0, pad_w = 0;

  size_t output_h = (cpu_input.height() - kernel_h) / stride_h + 1;
  size_t output_w = (cpu_input.width() - kernel_w) / stride_w + 1;
  size_t col_size =
      cpu_input.batch_size() * cpu_input.channels() * kernel_h * kernel_w * output_h * output_w;

  // CPU version
  std::vector<float> cpu_col_data(col_size);
  cpu::im2col(cpu_input, cpu_col_data.data(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  // GPU version
  Tensor<float, NCHW> gpu_input = cpu_input.to_gpu();
  device_ptr<float[]> gpu_col_data = make_array_ptr<float[]>(gpu_device_, col_size);
  cuda::im2col(gpu_input, gpu_col_data.get(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  // Transfer GPU result to CPU for comparison
  std::vector<float> gpu_col_cpu(col_size);
  gpu_device_->copyToHost(gpu_col_cpu.data(), gpu_col_data.get(), col_size * sizeof(float));

  // Compare results
  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(cpu_col_data[i], gpu_col_cpu[i], 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUTensorOpsTest, Im2colMultiBatch) {
  Tensor<float, NCHW> cpu_input({4, 3, 6, 6});
  cpu_input.fill_random_uniform(12.0f);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 1, pad_w = 1;

  size_t padded_h = cpu_input.height() + 2 * pad_h;
  size_t padded_w = cpu_input.width() + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;
  size_t col_size =
      cpu_input.batch_size() * cpu_input.channels() * kernel_h * kernel_w * output_h * output_w;

  // CPU version
  std::vector<float> cpu_col_data(col_size);
  cpu::im2col(cpu_input, cpu_col_data.data(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  // GPU version
  Tensor<float, NCHW> gpu_input = cpu_input.to_gpu();
  device_ptr<float[]> gpu_col_data = make_array_ptr<float[]>(gpu_device_, col_size);
  cuda::im2col(gpu_input, gpu_col_data.get(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  // Transfer GPU result to CPU for comparison
  std::vector<float> gpu_col_cpu(col_size);
  gpu_device_->copyToHost(gpu_col_cpu.data(), gpu_col_data.get(), col_size * sizeof(float));

  // Compare results
  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(cpu_col_data[i], gpu_col_cpu[i], 1e-5f) << "Mismatch at index " << i;
  }
}

// ==================== Col2im Tests ====================

TEST_F(GPUTensorOpsTest, Col2imBasic) {
  size_t batch_size = 1, channels = 1, height = 5, width = 5;
  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 0, pad_w = 0;

  size_t output_h = (height - kernel_h) / stride_h + 1;
  size_t output_w = (width - kernel_w) / stride_w + 1;
  size_t col_size = batch_size * channels * kernel_h * kernel_w * output_h * output_w;

  // Create col data
  std::vector<float> col_data(col_size);
  for (size_t i = 0; i < col_size; ++i) {
    col_data[i] = static_cast<float>(i % 10);
  }

  // CPU version
  std::vector<float> cpu_result(batch_size * channels * height * width, 0.0f);
  cpu::col2im(col_data.data(), cpu_result.data(), batch_size, channels, height, width, kernel_h,
              kernel_w, stride_h, stride_w, pad_h, pad_w);

  // GPU version
  device_ptr<float[]> gpu_col_data = make_array_ptr<float[]>(gpu_device_, col_size);
  gpu_device_->copyToDevice(gpu_col_data.get(), col_data.data(), col_size * sizeof(float));

  device_ptr<float[]> gpu_result =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * height * width);
  auto set_op = ops::set_scalar(gpu_result, 0.0f, batch_size * channels * height * width);
  ErrorStatus status = set_op->sync();
  ASSERT_FALSE(status) << "Failed to initialize GPU result buffer";

  cuda::col2im(gpu_col_data.get(), gpu_result.get(), batch_size, channels, height, width, kernel_h,
               kernel_w, stride_h, stride_w, pad_h, pad_w);

  // Transfer GPU result to CPU for comparison
  std::vector<float> gpu_result_cpu(batch_size * channels * height * width);
  gpu_device_->copyToHost(gpu_result_cpu.data(), gpu_result.get(),
                          batch_size * channels * height * width * sizeof(float));

  // Compare results
  for (size_t i = 0; i < cpu_result.size(); ++i) {
    EXPECT_NEAR(cpu_result[i], gpu_result_cpu[i], 1e-4f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUTensorOpsTest, Col2imWithPadding) {
  size_t batch_size = 1, channels = 2, height = 4, width = 4;
  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 1, pad_w = 1;

  size_t padded_h = height + 2 * pad_h;
  size_t padded_w = width + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;
  size_t col_size = batch_size * channels * kernel_h * kernel_w * output_h * output_w;

  // Create col data
  std::vector<float> col_data(col_size);
  for (size_t i = 0; i < col_size; ++i) {
    col_data[i] = static_cast<float>((i % 20) - 10);
  }

  // CPU version
  std::vector<float> cpu_result(batch_size * channels * height * width, 0.0f);
  cpu::col2im(col_data.data(), cpu_result.data(), batch_size, channels, height, width, kernel_h,
              kernel_w, stride_h, stride_w, pad_h, pad_w);

  // GPU version
  device_ptr<float[]> gpu_col_data = make_array_ptr<float[]>(gpu_device_, col_size);
  gpu_device_->copyToDevice(gpu_col_data.get(), col_data.data(), col_size * sizeof(float));

  device_ptr<float[]> gpu_result =
      make_array_ptr<float[]>(gpu_device_, batch_size * channels * height * width);
  auto set_op1 = ops::set_scalar(gpu_result, 0.0f, batch_size * channels * height * width);
  ErrorStatus status1 = set_op1->sync();
  ASSERT_FALSE(status1) << "Failed to initialize GPU result buffer";

  cuda::col2im(gpu_col_data.get(), gpu_result.get(), batch_size, channels, height, width, kernel_h,
               kernel_w, stride_h, stride_w, pad_h, pad_w);

  // Transfer GPU result to CPU for comparison
  std::vector<float> gpu_result_cpu(batch_size * channels * height * width);
  gpu_device_->copyToHost(gpu_result_cpu.data(), gpu_result.get(),
                          batch_size * channels * height * width * sizeof(float));

  // Compare results
  for (size_t i = 0; i < cpu_result.size(); ++i) {
    EXPECT_NEAR(cpu_result[i], gpu_result_cpu[i], 1e-4f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUTensorOpsTest, Im2colCol2imRoundTrip) {
  // Test that im2col -> col2im with stride=1, no padding approximately recovers original
  Tensor<float, NCHW> cpu_input({1, 1, 6, 6});
  cpu_input.fill_random_uniform(10.0f);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 0, pad_w = 0;

  size_t output_h = (cpu_input.height() - kernel_h) / stride_h + 1;
  size_t output_w = (cpu_input.width() - kernel_w) / stride_w + 1;
  size_t col_size =
      cpu_input.batch_size() * cpu_input.channels() * kernel_h * kernel_w * output_h * output_w;

  // CPU: im2col -> col2im
  std::vector<float> cpu_col_data(col_size);
  cpu::im2col(cpu_input, cpu_col_data.data(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor<float, NCHW> cpu_reconstructed({1, 1, 6, 6});
  cpu_reconstructed.fill(0.0f);
  cpu::col2im(cpu_col_data.data(), cpu_reconstructed.data(), cpu_input.batch_size(),
              cpu_input.channels(), cpu_input.height(), cpu_input.width(), kernel_h, kernel_w,
              stride_h, stride_w, pad_h, pad_w);

  // GPU: im2col -> col2im
  Tensor<float, NCHW> gpu_input = cpu_input.to_gpu();
  device_ptr<float[]> gpu_col_data = make_array_ptr<float[]>(gpu_device_, col_size);
  cuda::im2col(gpu_input, gpu_col_data.get(), kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor<float, NCHW> gpu_reconstructed({1, 1, 6, 6}, gpu_device_);
  gpu_reconstructed.fill(0.0f);
  cuda::col2im(gpu_col_data.get(), gpu_reconstructed.data(), gpu_input.batch_size(),
               gpu_input.channels(), gpu_input.height(), gpu_input.width(), kernel_h, kernel_w,
               stride_h, stride_w, pad_h, pad_w);

  // Compare CPU and GPU reconstructions
  compareTensors(cpu_reconstructed, gpu_reconstructed, 1e-4f);
}

// ==================== Combined Operations Tests ====================

TEST_F(GPUTensorOpsTest, CombinedPadCropSlice) {
  // Test combining multiple operations
  Tensor<float, NCHW> cpu_original({4, 3, 8, 8});
  cpu_original.fill_random_uniform(15.0f);

  // CPU: pad -> crop -> slice
  Tensor<float, NCHW> cpu_padded({4, 3, 12, 12});
  cpu::pad(cpu_original, cpu_padded, 2, 2, 0.0f);
  Tensor<float, NCHW> cpu_cropped({4, 3, 6, 6});
  cpu::crop(cpu_padded, cpu_cropped, 3, 3, 8, 8);
  Tensor<float, NCHW> cpu_sliced({2, 3, 6, 6});
  cpu::slice_batch(cpu_cropped, cpu_sliced, 1, 3);

  // GPU: pad -> crop -> slice
  Tensor<float, NCHW> gpu_original = cpu_original.to_gpu();
  Tensor<float, NCHW> gpu_padded({4, 3, 12, 12}, gpu_device_);
  cuda::pad(gpu_original, gpu_padded, 2, 2, 0.0f);
  Tensor<float, NCHW> gpu_cropped({4, 3, 6, 6}, gpu_device_);
  cuda::crop(gpu_padded, gpu_cropped, 3, 3, 8, 8);
  Tensor<float, NCHW> gpu_sliced({2, 3, 6, 6}, gpu_device_);
  cuda::slice_batch(gpu_cropped, gpu_sliced, 1, 3);

  compareTensors(cpu_sliced, gpu_sliced);
}

TEST_F(GPUTensorOpsTest, LargeTensorOperations) {
  // Test with larger tensors to ensure GPU operations scale properly
  Tensor<float, NCHW> cpu_tensor({8, 16, 32, 32});
  cpu_tensor.fill_random_uniform(20.0f);

  // Test padding
  Tensor<float, NCHW> cpu_padded({8, 16, 36, 36});
  cpu::pad(cpu_tensor, cpu_padded, 2, 2, 0.0f);
  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_padded({8, 16, 36, 36}, gpu_device_);
  cuda::pad(gpu_tensor, gpu_padded, 2, 2, 0.0f);
  compareTensors(cpu_padded, gpu_padded);

  // Test cropping
  Tensor<float, NCHW> cpu_cropped({8, 16, 22, 22});
  cpu::crop(cpu_tensor, cpu_cropped, 5, 5, 26, 26);
  Tensor<float, NCHW> gpu_cropped({8, 16, 22, 22}, gpu_device_);
  cuda::crop(gpu_tensor, gpu_cropped, 5, 5, 26, 26);
  compareTensors(cpu_cropped, gpu_cropped);

  // Test slicing
  Tensor<float, NCHW> cpu_sliced({4, 16, 32, 32});
  cpu::slice_batch(cpu_tensor, cpu_sliced, 2, 6);
  Tensor<float, NCHW> gpu_sliced({4, 16, 32, 32}, gpu_device_);
  cuda::slice_batch(gpu_tensor, gpu_sliced, 2, 6);
  compareTensors(cpu_sliced, gpu_sliced);
}

// ==================== Edge Cases ====================

TEST_F(GPUTensorOpsTest, MinimalTensor) {
  // Test with minimal sized tensor (1x1x1x1)
  Tensor<float, NCHW> cpu_tensor({1, 1, 1, 1});
  cpu_tensor(0, 0, 0, 0) = 42.0f;

  Tensor<float, NCHW> cpu_padded({1, 1, 3, 3});
  cpu::pad(cpu_tensor, cpu_padded, 1, 1, 0.0f);
  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_padded({1, 1, 3, 3}, gpu_device_);
  cuda::pad(gpu_tensor, gpu_padded, 1, 1, 0.0f);

  compareTensors(cpu_padded, gpu_padded);
}

TEST_F(GPUTensorOpsTest, SinglePixelPadding) {
  Tensor<float, NCHW> cpu_tensor({1, 1, 3, 3});
  cpu_tensor.fill_random_uniform(5.0f);

  Tensor<float, NCHW> cpu_padded({1, 1, 5, 5});
  cpu::pad(cpu_tensor, cpu_padded, 1, 1, -1.0f);
  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_padded({1, 1, 5, 5}, gpu_device_);
  cuda::pad(gpu_tensor, gpu_padded, 1, 1, -1.0f);

  compareTensors(cpu_padded, gpu_padded);
}

TEST_F(GPUTensorOpsTest, AsymmetricDimensions) {
  // Test with very asymmetric tensor (tall and narrow)
  Tensor<float, NCHW> cpu_tensor({1, 1, 20, 3});
  cpu_tensor.fill_random_uniform(10.0f);

  Tensor<float, NCHW> cpu_padded({1, 1, 24, 13});
  cpu::pad(cpu_tensor, cpu_padded, 2, 5, 1.0f);
  Tensor<float, NCHW> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float, NCHW> gpu_padded({1, 1, 24, 13}, gpu_device_);
  cuda::pad(gpu_tensor, gpu_padded, 2, 5, 1.0f);

  compareTensors(cpu_padded, gpu_padded);

  // Test with wide and short
  Tensor<float, NCHW> cpu_tensor2({1, 1, 3, 20});
  cpu_tensor2.fill_random_uniform(10.0f);

  Tensor<float, NCHW> cpu_padded2({1, 1, 13, 24});
  cpu::pad(cpu_tensor2, cpu_padded2, 5, 2, -2.0f);
  Tensor<float, NCHW> gpu_tensor2 = cpu_tensor2.to_gpu();
  Tensor<float, NCHW> gpu_padded2({1, 1, 13, 24}, gpu_device_);
  cuda::pad(gpu_tensor2, gpu_padded2, 5, 2, -2.0f);

  compareTensors(cpu_padded2, gpu_padded2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif // USE_CUDA