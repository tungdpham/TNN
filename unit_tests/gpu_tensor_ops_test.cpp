/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>

#include "device/device_manager.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"

using namespace tnn;

#ifdef USE_CUDA

class GPUopsTest : public ::testing::Test {
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
      GTEST_SKIP() << "No GPU device available, skipping GPU tensor ops tests";
    }
  }

  void TearDown() override {}

  static void TearDownTestSuite() {}

  template <typename T>
  void compareTensors(const ConstTensor &expected, const ConstTensor &actual,
                      T tolerance = static_cast<T>(1e-5)) {
    ASSERT_TRUE(expected->shape() == actual->shape())
        << "Tensors have different shapes. Expected: " << expected->shape_str()
        << ", Actual: " << actual->shape_str();

    Tensor expected_cpu =
        expected->device_type() == DeviceType::CPU ? expected->clone() : expected->to_cpu();
    Tensor actual_cpu =
        actual->device_type() == DeviceType::CPU ? actual->clone() : actual->to_cpu();

    auto shape = expected_cpu->shape();
    for (size_t n = 0; n < shape[0]; ++n) {
      for (size_t c = 0; c < shape[1]; ++c) {
        for (size_t h = 0; h < shape[2]; ++h) {
          for (size_t w = 0; w < shape[3]; ++w) {
            T expected_val = expected_cpu->at<T>({n, c, h, w});
            T actual_val = actual_cpu->at<T>({n, c, h, w});
            EXPECT_NEAR(expected_val, actual_val, tolerance)
                << "Mismatch at position [" << n << "," << c << "," << h << "," << w
                << "]. Expected: " << expected_val << ", Got: " << actual_val;
          }
        }
      }
    }
  }

  bool has_gpu_;
};

TEST_F(GPUopsTest, PadBasic) {
  Tensor cpu_tensor = make_tensor<float>({1, 1, 3, 3});
  auto cpu_data = cpu_tensor->data_as<float>();
  for (size_t i = 0; i < 9; ++i) {
    cpu_data[i] = static_cast<float>(i + 1);
  }

  Tensor cpu_padded = make_tensor<float>({1, 1, 5, 5});
  ops::pad<float>(cpu_tensor, cpu_padded, 1, 1, 0.0f);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_padded = make_tensor<float>({1, 1, 5, 5}, getGPU());
  ops::pad<float>(gpu_tensor, gpu_padded, 1, 1, 0.0f);

  compareTensors<float>(cpu_padded, gpu_padded);
}

TEST_F(GPUopsTest, PadMultiChannel) {
  Tensor cpu_tensor = make_tensor<float>({2, 3, 4, 4});
  cpu_tensor->fill_random_uniform(10.0f);

  Tensor cpu_padded = make_tensor<float>({2, 3, 8, 8});
  ops::pad<float>(cpu_tensor, cpu_padded, 2, 2, -1.0f);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_padded = make_tensor<float>({2, 3, 8, 8}, getGPU());
  ops::pad<float>(gpu_tensor, gpu_padded, 2, 2, -1.0f);

  compareTensors<float>(cpu_padded, gpu_padded);
}

TEST_F(GPUopsTest, PadAsymmetric) {
  Tensor cpu_tensor = make_tensor<float>({1, 2, 5, 7});
  cpu_tensor->fill_random_uniform(5.0f);

  Tensor cpu_padded = make_tensor<float>({1, 2, 11, 9});
  ops::pad<float>(cpu_tensor, cpu_padded, 3, 1, 2.5f);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_padded = make_tensor<float>({1, 2, 11, 9}, getGPU());
  ops::pad<float>(gpu_tensor, gpu_padded, 3, 1, 2.5f);

  compareTensors<float>(cpu_padded, gpu_padded);
}

TEST_F(GPUopsTest, UnpadBasic) {
  Tensor cpu_tensor = make_tensor<float>({1, 1, 5, 5});
  cpu_tensor->fill_random_uniform(10.0f);

  Tensor cpu_unpadded = make_tensor<float>({1, 1, 3, 3});
  ops::unpad<float>(cpu_tensor, cpu_unpadded, 1, 1);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_unpadded = make_tensor<float>({1, 1, 3, 3}, getGPU());
  ops::unpad<float>(gpu_tensor, gpu_unpadded, 1, 1);

  compareTensors<float>(cpu_unpadded, gpu_unpadded);
}

TEST_F(GPUopsTest, UnpadMultiChannel) {
  Tensor cpu_tensor = make_tensor<float>({2, 3, 8, 8});
  cpu_tensor->fill_random_uniform(15.0f);

  Tensor cpu_unpadded = make_tensor<float>({2, 3, 4, 4});
  ops::unpad<float>(cpu_tensor, cpu_unpadded, 2, 2);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_unpadded = make_tensor<float>({2, 3, 4, 4}, getGPU());
  ops::unpad<float>(gpu_tensor, gpu_unpadded, 2, 2);

  compareTensors<float>(cpu_unpadded, gpu_unpadded);
}

TEST_F(GPUopsTest, PadUnpadRoundTrip) {
  Tensor cpu_original = make_tensor<float>({1, 2, 4, 4});
  cpu_original->fill_random_uniform(8.0f);

  Tensor cpu_padded = make_tensor<float>({1, 2, 8, 8});
  ops::pad<float>(cpu_original, cpu_padded, 2, 2, 0.0f);
  Tensor cpu_restored = make_tensor<float>({1, 2, 4, 4});
  ops::unpad<float>(cpu_padded, cpu_restored, 2, 2);

  Tensor gpu_original = cpu_original->to_gpu();
  Tensor gpu_padded = make_tensor<float>({1, 2, 8, 8}, getGPU());
  ops::pad<float>(gpu_original, gpu_padded, 2, 2, 0.0f);
  Tensor gpu_restored = make_tensor<float>({1, 2, 4, 4}, getGPU());
  ops::unpad<float>(gpu_padded, gpu_restored, 2, 2);

  compareTensors<float>(cpu_original, cpu_restored);
  compareTensors<float>(cpu_original, gpu_restored);
  compareTensors<float>(cpu_restored, gpu_restored);
}

TEST_F(GPUopsTest, CropBasic) {
  Tensor cpu_tensor = make_tensor<float>({1, 1, 5, 5});
  auto cpu_data = cpu_tensor->data_as<float>();
  for (size_t i = 0; i < 25; ++i) {
    cpu_data[i] = static_cast<float>(i);
  }

  Tensor cpu_cropped = make_tensor<float>({1, 1, 3, 3});
  ops::crop<float>(cpu_tensor, cpu_cropped, 1, 1, 3, 3);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_cropped = make_tensor<float>({1, 1, 3, 3}, getGPU());
  ops::crop<float>(gpu_tensor, gpu_cropped, 1, 1, 3, 3);

  compareTensors<float>(cpu_cropped, gpu_cropped);
}

TEST_F(GPUopsTest, CropMultiChannel) {
  Tensor cpu_tensor = make_tensor<float>({2, 3, 10, 10});
  cpu_tensor->fill_random_uniform(20.0f);

  Tensor cpu_cropped = make_tensor<float>({2, 3, 6, 6});
  ops::crop<float>(cpu_tensor, cpu_cropped, 2, 3, 7, 8);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_cropped = make_tensor<float>({2, 3, 6, 6}, getGPU());
  ops::crop<float>(gpu_tensor, gpu_cropped, 2, 3, 7, 8);

  compareTensors<float>(cpu_cropped, gpu_cropped);
}

TEST_F(GPUopsTest, CropCorner) {
  Tensor cpu_tensor = make_tensor<float>({1, 2, 8, 8});
  cpu_tensor->fill_random_uniform(12.0f);

  Tensor cpu_cropped = make_tensor<float>({1, 2, 4, 4});
  ops::crop<float>(cpu_tensor, cpu_cropped, 0, 0, 3, 3);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_cropped = make_tensor<float>({1, 2, 4, 4}, getGPU());
  ops::crop<float>(gpu_tensor, gpu_cropped, 0, 0, 3, 3);

  compareTensors<float>(cpu_cropped, gpu_cropped);
}

TEST_F(GPUopsTest, CropBottomRight) {
  Tensor cpu_tensor = make_tensor<float>({1, 1, 6, 6});
  cpu_tensor->fill_random_uniform(10.0f);

  Tensor cpu_cropped = make_tensor<float>({1, 1, 3, 3});
  ops::crop<float>(cpu_tensor, cpu_cropped, 3, 3, 5, 5);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_cropped = make_tensor<float>({1, 1, 3, 3}, getGPU());
  ops::crop<float>(gpu_tensor, gpu_cropped, 3, 3, 5, 5);

  compareTensors<float>(cpu_cropped, gpu_cropped);
}

TEST_F(GPUopsTest, SliceBatchBasic) {
  Tensor cpu_tensor = make_tensor<float>({4, 2, 3, 3});
  cpu_tensor->fill_random_uniform(15.0f);

  Tensor cpu_sliced = make_tensor<float>({2, 2, 3, 3});
  ops::slice_batch<float>(cpu_tensor, cpu_sliced, 1, 3);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_sliced = make_tensor<float>({2, 2, 3, 3}, getGPU());
  ops::slice_batch<float>(gpu_tensor, gpu_sliced, 1, 3);

  compareTensors<float>(cpu_sliced, gpu_sliced);
}

TEST_F(GPUopsTest, SliceBatchSingle) {
  Tensor cpu_tensor = make_tensor<float>({5, 3, 4, 4});
  cpu_tensor->fill_random_uniform(10.0f);

  Tensor cpu_sliced = make_tensor<float>({1, 3, 4, 4});
  ops::slice_batch<float>(cpu_tensor, cpu_sliced, 2, 3);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_sliced = make_tensor<float>({1, 3, 4, 4}, getGPU());
  ops::slice_batch<float>(gpu_tensor, gpu_sliced, 2, 3);

  compareTensors<float>(cpu_sliced, gpu_sliced);
}

TEST_F(GPUopsTest, SliceBatchFirstBatch) {
  Tensor cpu_tensor = make_tensor<float>({3, 2, 5, 5});
  cpu_tensor->fill_random_uniform(8.0f);

  Tensor cpu_sliced = make_tensor<float>({1, 2, 5, 5});
  ops::slice_batch<float>(cpu_tensor, cpu_sliced, 0, 1);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_sliced = make_tensor<float>({1, 2, 5, 5}, getGPU());
  ops::slice_batch<float>(gpu_tensor, gpu_sliced, 0, 1);

  compareTensors<float>(cpu_sliced, gpu_sliced);
}

TEST_F(GPUopsTest, SplitBasic) {
  Tensor cpu_tensor = make_tensor<float>({4, 2, 3, 3});
  cpu_tensor->fill_random_uniform(10.0f);

  std::vector<Tensor> cpu_splits, gpu_splits;
  ops::split<float>(cpu_tensor, cpu_splits, 2);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  ops::split<float>(gpu_tensor, gpu_splits, 2);
  ASSERT_EQ(cpu_splits.size(), gpu_splits.size());

  for (size_t i = 0; i < cpu_splits.size(); ++i) {
    compareTensors<float>(cpu_splits[i], gpu_splits[i]);
  }
}

TEST_F(GPUopsTest, SplitMultiple) {
  Tensor cpu_tensor = make_tensor<float>({8, 3, 4, 4});
  cpu_tensor->fill_random_uniform(15.0f);
  std::vector<Tensor> cpu_splits;
  ops::split<float>(cpu_tensor, cpu_splits, 4);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  std::vector<Tensor> gpu_splits;
  ops::split<float>(gpu_tensor, gpu_splits, 4);

  ASSERT_EQ(cpu_splits.size(), gpu_splits.size());

  float *original_data = cpu_tensor->data_as<float>();
  int original_idx = 0;
  for (size_t i = 0; i < cpu_splits.size(); ++i) {
    float *split_data = cpu_splits[i]->data_as<float>();
    for (size_t idx = 0; idx < cpu_splits[i]->size(); ++idx) {
      EXPECT_EQ(original_data[original_idx++], split_data[idx])
          << "Mismatch in CPU split at index " << idx << " of split " << i;
    }
  }

  ASSERT_EQ(original_idx, cpu_tensor->size());

  for (size_t i = 0; i < cpu_splits.size(); ++i) {
    compareTensors<float>(cpu_splits[i], gpu_splits[i]);
  }
}

TEST_F(GPUopsTest, SplitSingleBatch) {
  Tensor cpu_tensor = make_tensor<float>({6, 2, 5, 5});
  cpu_tensor->fill_random_uniform(12.0f);

  std::vector<Tensor> cpu_splits, gpu_splits;
  ops::split<float>(cpu_tensor, cpu_splits, 6);

  Tensor gpu_tensor = cpu_tensor->to_gpu();
  ops::split<float>(gpu_tensor, gpu_splits, 6);

  ASSERT_EQ(cpu_splits.size(), gpu_splits.size());

  for (size_t i = 0; i < cpu_splits.size(); ++i) {
    compareTensors<float>(cpu_splits[i], gpu_splits[i]);
  }
}

TEST_F(GPUopsTest, Im2colBasicKernel3x3) {
  Tensor cpu_input = make_tensor<float>({1, 1, 5, 5});
  auto cpu_input_data = cpu_input->data_as<float>();
  for (size_t i = 0; i < 25; ++i) {
    cpu_input_data[i] = static_cast<float>(i + 1);
  }

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 0, pad_w = 0;

  auto input_shape = cpu_input->shape();
  size_t output_h = (input_shape[2] - kernel_h) / stride_h + 1;
  size_t output_w = (input_shape[3] - kernel_w) / stride_w + 1;
  size_t col_size = input_shape[0] * input_shape[1] * kernel_h * kernel_w * output_h * output_w;

  Tensor cpu_col_data = make_tensor<float>({col_size});
  ops::im2col<float>(cpu_input, cpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_input = cpu_input->to_gpu();
  Tensor gpu_col_data = make_tensor<float>({col_size}, getGPU());
  ops::im2col<float>(gpu_input, gpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  std::vector<float> cpu_col_cpu(col_size);
  std::vector<float> gpu_col_cpu(col_size);
  std::copy(cpu_col_data->data_as<float>(), cpu_col_data->data_as<float>() + col_size,
            cpu_col_cpu.data());
  getGPU().copyToHost(gpu_col_cpu.data(), gpu_col_data->data_as<float>(), col_size * sizeof(float));

  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(cpu_col_cpu[i], gpu_col_cpu[i], 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUopsTest, Im2colWithPadding) {
  Tensor cpu_input = make_tensor<float>({1, 2, 4, 4});
  cpu_input->fill_random_uniform(10.0f);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 1, pad_w = 1;

  auto input_shape = cpu_input->shape();
  size_t padded_h = input_shape[2] + 2 * pad_h;
  size_t padded_w = input_shape[3] + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;
  size_t col_size = input_shape[0] * input_shape[1] * kernel_h * kernel_w * output_h * output_w;

  Tensor cpu_col_data = make_tensor<float>({col_size});
  ops::im2col<float>(cpu_input, cpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_input = cpu_input->to_gpu();
  Tensor gpu_col_data = make_tensor<float>({col_size}, getGPU());
  ops::im2col<float>(gpu_input, gpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  std::vector<float> cpu_col_cpu(col_size);
  std::vector<float> gpu_col_cpu(col_size);
  std::copy(cpu_col_data->data_as<float>(), cpu_col_data->data_as<float>() + col_size,
            cpu_col_cpu.data());
  getGPU().copyToHost(gpu_col_cpu.data(), gpu_col_data->data_as<float>(), col_size * sizeof(float));

  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(cpu_col_cpu[i], gpu_col_cpu[i], 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUopsTest, Im2colWithStride) {
  Tensor cpu_input = make_tensor<float>({1, 1, 8, 8});
  cpu_input->fill_random_uniform(15.0f);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 2, stride_w = 2;
  size_t pad_h = 0, pad_w = 0;

  auto input_shape = cpu_input->shape();
  size_t output_h = (input_shape[2] - kernel_h) / stride_h + 1;
  size_t output_w = (input_shape[3] - kernel_w) / stride_w + 1;
  size_t col_size = input_shape[0] * input_shape[1] * kernel_h * kernel_w * output_h * output_w;

  Tensor cpu_col = make_tensor<float>({col_size});
  ops::im2col<float>(cpu_input, cpu_col, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_input = cpu_input->to_gpu();
  Tensor gpu_col_data = make_tensor<float>({col_size}, getGPU());
  ops::im2col<float>(gpu_input, gpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  std::vector<float> cpu_col_cpu(col_size);
  std::vector<float> gpu_col_cpu(col_size);
  std::copy(cpu_col->data_as<float>(), cpu_col->data_as<float>() + col_size, cpu_col_cpu.data());
  getGPU().copyToHost(gpu_col_cpu.data(), gpu_col_data->data_as<float>(), col_size * sizeof(float));

  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(cpu_col_cpu[i], gpu_col_cpu[i], 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUopsTest, Im2colMultiBatch) {
  Tensor cpu_input = make_tensor<float>({4, 3, 6, 6});
  cpu_input->fill_random_uniform(12.0f);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 1, pad_w = 1;

  auto input_shape = cpu_input->shape();
  size_t padded_h = input_shape[2] + 2 * pad_h;
  size_t padded_w = input_shape[3] + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;
  size_t col_size = input_shape[0] * input_shape[1] * kernel_h * kernel_w * output_h * output_w;

  Tensor cpu_col = make_tensor<float>({col_size});
  ops::im2col<float>(cpu_input, cpu_col, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_input = cpu_input->to_gpu();
  Tensor gpu_col_data = make_tensor<float>({col_size}, getGPU());
  ops::im2col<float>(gpu_input, gpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  std::vector<float> cpu_col_cpu(col_size);
  std::vector<float> gpu_col_cpu(col_size);
  std::copy(cpu_col->data_as<float>(), cpu_col->data_as<float>() + col_size, cpu_col_cpu.data());
  getGPU().copyToHost(gpu_col_cpu.data(), gpu_col_data->data_as<float>(), col_size * sizeof(float));

  for (size_t i = 0; i < col_size; ++i) {
    EXPECT_NEAR(cpu_col_cpu[i], gpu_col_cpu[i], 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUopsTest, Col2imBasic) {
  size_t batch_size = 1, channels = 1, height = 5, width = 5;
  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 0, pad_w = 0;

  size_t output_h = (height - kernel_h) / stride_h + 1;
  size_t output_w = (width - kernel_w) / stride_w + 1;
  size_t col_size = batch_size * channels * kernel_h * kernel_w * output_h * output_w;

  Tensor cpu_col_data = make_tensor<float>({col_size});
  auto cpu_col_ptr = cpu_col_data->data_as<float>();
  for (size_t i = 0; i < col_size; ++i) {
    cpu_col_ptr[i] = static_cast<float>(i % 10);
  }

  Tensor cpu_result = make_tensor<float>({batch_size * channels * height * width});
  cpu_result->fill(0.0f);
  ops::col2im<float>(cpu_col_data, cpu_result, batch_size, channels, height, width, kernel_h,
                     kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_col_data = cpu_col_data->to_gpu();

  Tensor gpu_result = make_tensor<float>({batch_size * channels * height * width}, getGPU());
  gpu_result->fill(0.0f);

  ops::col2im<float>(gpu_col_data, gpu_result, batch_size, channels, height, width, kernel_h,
                     kernel_w, stride_h, stride_w, pad_h, pad_w);

  std::vector<float> cpu_result_cpu(batch_size * channels * height * width);
  std::vector<float> gpu_result_cpu(batch_size * channels * height * width);
  std::copy(cpu_result->data_as<float>(), cpu_result->data_as<float>() + cpu_result_cpu.size(),
            cpu_result_cpu.data());
  getGPU().copyToHost(gpu_result_cpu.data(), gpu_result->data_as<float>(),
                      batch_size * channels * height * width * sizeof(float));

  for (size_t i = 0; i < cpu_result_cpu.size(); ++i) {
    EXPECT_NEAR(cpu_result_cpu[i], gpu_result_cpu[i], 1e-4f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUopsTest, Col2imWithPadding) {
  size_t batch_size = 1, channels = 2, height = 4, width = 4;
  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 1, pad_w = 1;

  size_t padded_h = height + 2 * pad_h;
  size_t padded_w = width + 2 * pad_w;
  size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  size_t output_w = (padded_w - kernel_w) / stride_w + 1;
  size_t col_size = batch_size * channels * kernel_h * kernel_w * output_h * output_w;

  Tensor cpu_col_data = make_tensor<float>({col_size});
  auto cpu_col_ptr = cpu_col_data->data_as<float>();
  for (size_t i = 0; i < col_size; ++i) {
    cpu_col_ptr[i] = static_cast<float>((i % 20) - 10);
  }

  Tensor cpu_result = make_tensor<float>({batch_size * channels * height * width});
  cpu_result->fill(0.0f);
  ops::col2im<float>(cpu_col_data, cpu_result, batch_size, channels, height, width, kernel_h,
                     kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_col_data = cpu_col_data->to_gpu();

  Tensor gpu_result = make_tensor<float>({batch_size * channels * height * width}, getGPU());
  gpu_result->fill(0.0f);

  ops::col2im<float>(gpu_col_data, gpu_result, batch_size, channels, height, width, kernel_h,
                     kernel_w, stride_h, stride_w, pad_h, pad_w);

  std::vector<float> cpu_result_cpu(batch_size * channels * height * width);
  std::vector<float> gpu_result_cpu(batch_size * channels * height * width);
  std::copy(cpu_result->data_as<float>(), cpu_result->data_as<float>() + cpu_result_cpu.size(),
            cpu_result_cpu.data());
  getGPU().copyToHost(gpu_result_cpu.data(), gpu_result->data_as<float>(),
                      batch_size * channels * height * width * sizeof(float));

  for (size_t i = 0; i < cpu_result_cpu.size(); ++i) {
    EXPECT_NEAR(cpu_result_cpu[i], gpu_result_cpu[i], 1e-4f) << "Mismatch at index " << i;
  }
}

TEST_F(GPUopsTest, Im2colCol2imRoundTrip) {
  Tensor cpu_input = make_tensor<float>({1, 1, 6, 6});
  cpu_input->fill_random_uniform(10.0f);

  size_t kernel_h = 3, kernel_w = 3;
  size_t stride_h = 1, stride_w = 1;
  size_t pad_h = 0, pad_w = 0;

  auto input_shape = cpu_input->shape();
  size_t output_h = (input_shape[2] - kernel_h) / stride_h + 1;
  size_t output_w = (input_shape[3] - kernel_w) / stride_w + 1;
  size_t col_size = input_shape[0] * input_shape[1] * kernel_h * kernel_w * output_h * output_w;

  Tensor cpu_col_data = make_tensor<float>({col_size});
  ops::im2col<float>(cpu_input, cpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor cpu_reconstructed = make_tensor<float>({1, 1, 6, 6});
  cpu_reconstructed->fill(0.0f);
  ops::col2im<float>(cpu_col_data, cpu_reconstructed, input_shape[0], input_shape[1],
                     input_shape[2], input_shape[3], kernel_h, kernel_w, stride_h, stride_w, pad_h,
                     pad_w);

  Tensor gpu_input = cpu_input->to_gpu();
  Tensor gpu_col_data = make_tensor<float>({col_size}, getGPU());
  ops::im2col<float>(gpu_input, gpu_col_data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);

  Tensor gpu_reconstructed = make_tensor<float>({1, 1, 6, 6}, getGPU());
  gpu_reconstructed->fill(0.0f);
  auto gpu_input_shape = gpu_input->shape();
  ops::col2im<float>(gpu_col_data, gpu_reconstructed, gpu_input_shape[0], gpu_input_shape[1],
                     gpu_input_shape[2], gpu_input_shape[3], kernel_h, kernel_w, stride_h, stride_w,
                     pad_h, pad_w);

  compareTensors<float>(cpu_reconstructed, gpu_reconstructed, 1e-4f);
}

TEST_F(GPUopsTest, CombinedPadCropSlice) {
  Tensor cpu_original = make_tensor<float>({4, 3, 8, 8});
  cpu_original->fill_random_uniform(15.0f);

  Tensor cpu_padded = make_tensor<float>({4, 3, 12, 12});
  ops::pad<float>(cpu_original, cpu_padded, 2, 2, 0.0f);
  Tensor cpu_cropped = make_tensor<float>({4, 3, 6, 6});
  ops::crop<float>(cpu_padded, cpu_cropped, 3, 3, 8, 8);
  Tensor cpu_sliced = make_tensor<float>({2, 3, 6, 6});
  ops::slice_batch<float>(cpu_cropped, cpu_sliced, 1, 3);

  Tensor gpu_original = cpu_original->to_gpu();
  Tensor gpu_padded = make_tensor<float>({4, 3, 12, 12}, getGPU());
  ops::pad<float>(gpu_original, gpu_padded, 2, 2, 0.0f);
  Tensor gpu_cropped = make_tensor<float>({4, 3, 6, 6}, getGPU());
  ops::crop<float>(gpu_padded, gpu_cropped, 3, 3, 8, 8);
  Tensor gpu_sliced = make_tensor<float>({2, 3, 6, 6}, getGPU());
  ops::slice_batch<float>(gpu_cropped, gpu_sliced, 1, 3);

  compareTensors<float>(cpu_sliced, gpu_sliced);
}

TEST_F(GPUopsTest, LargeTensorOperations) {
  Tensor cpu_tensor = make_tensor<float>({8, 16, 32, 32});
  cpu_tensor->fill_random_uniform(20.0f);

  Tensor cpu_padded = make_tensor<float>({8, 16, 36, 36});
  ops::pad<float>(cpu_tensor, cpu_padded, 2, 2, 0.0f);
  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_padded = make_tensor<float>({8, 16, 36, 36}, getGPU());
  ops::pad<float>(gpu_tensor, gpu_padded, 2, 2, 0.0f);
  compareTensors<float>(cpu_padded, gpu_padded);

  Tensor cpu_cropped = make_tensor<float>({8, 16, 22, 22});
  ops::crop<float>(cpu_tensor, cpu_cropped, 5, 5, 26, 26);
  Tensor gpu_cropped = make_tensor<float>({8, 16, 22, 22}, getGPU());
  ops::crop<float>(gpu_tensor, gpu_cropped, 5, 5, 26, 26);
  compareTensors<float>(cpu_cropped, gpu_cropped);

  Tensor cpu_sliced = make_tensor<float>({4, 16, 32, 32});
  ops::slice_batch<float>(cpu_tensor, cpu_sliced, 2, 6);
  Tensor gpu_sliced = make_tensor<float>({4, 16, 32, 32}, getGPU());
  ops::slice_batch<float>(gpu_tensor, gpu_sliced, 2, 6);
  compareTensors<float>(cpu_sliced, gpu_sliced);
}

TEST_F(GPUopsTest, MinimalTensor) {
  Tensor cpu_tensor = make_tensor<float>({1, 1, 1, 1});
  auto cpu_data = cpu_tensor->data_as<float>();
  cpu_data[0] = 42.0f;

  Tensor cpu_padded = make_tensor<float>({1, 1, 3, 3});
  ops::pad<float>(cpu_tensor, cpu_padded, 1, 1, 0.0f);
  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_padded = make_tensor<float>({1, 1, 3, 3}, getGPU());
  ops::pad<float>(gpu_tensor, gpu_padded, 1, 1, 0.0f);

  compareTensors<float>(cpu_padded, gpu_padded);
}

TEST_F(GPUopsTest, SinglePixelPadding) {
  Tensor cpu_tensor = make_tensor<float>({1, 1, 3, 3});
  cpu_tensor->fill_random_uniform(5.0f);

  Tensor cpu_padded = make_tensor<float>({1, 1, 5, 5});
  ops::pad<float>(cpu_tensor, cpu_padded, 1, 1, -1.0f);
  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_padded = make_tensor<float>({1, 1, 5, 5}, getGPU());
  ops::pad<float>(gpu_tensor, gpu_padded, 1, 1, -1.0f);

  compareTensors<float>(cpu_padded, gpu_padded);
}

TEST_F(GPUopsTest, AsymmetricDimensions) {
  Tensor cpu_tensor = make_tensor<float>({1, 1, 20, 3});
  cpu_tensor->fill_random_uniform(10.0f);

  Tensor cpu_padded = make_tensor<float>({1, 1, 24, 13});
  ops::pad<float>(cpu_tensor, cpu_padded, 2, 5, 1.0f);
  Tensor gpu_tensor = cpu_tensor->to_gpu();
  Tensor gpu_padded = make_tensor<float>({1, 1, 24, 13}, getGPU());
  ops::pad<float>(gpu_tensor, gpu_padded, 2, 5, 1.0f);

  compareTensors<float>(cpu_padded, gpu_padded);

  Tensor cpu_tensor2 = make_tensor<float>({1, 1, 3, 20});
  cpu_tensor2->fill_random_uniform(10.0f);

  Tensor cpu_padded2 = make_tensor<float>({1, 1, 13, 24});
  ops::pad<float>(cpu_tensor2, cpu_padded2, 5, 2, -2.0f);
  Tensor gpu_tensor2 = cpu_tensor2->to_gpu();
  Tensor gpu_padded2 = make_tensor<float>({1, 1, 13, 24}, getGPU());
  ops::pad<float>(gpu_tensor2, gpu_padded2, 5, 2, -2.0f);

  compareTensors<float>(cpu_padded2, gpu_padded2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif