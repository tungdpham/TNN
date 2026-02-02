#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>

#include "device/device_manager.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

class GPUTensorTest : public ::testing::Test {
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
      GTEST_SKIP() << "No GPU device available, skipping GPU tensor tests";
    }

    small_tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());
    small_tensor->fill(1.0);

    large_tensor = make_tensor<float>({2, 3, 4, 4}, getGPU());
    large_tensor->fill(2.0);
  }

  void TearDown() override {
    small_tensor = Tensor();
    large_tensor = Tensor();
  }

  static void TearDownTestSuite() {}

  bool has_gpu_;
  Tensor small_tensor;
  Tensor large_tensor;
};

TEST_F(GPUTensorTest, Constructor4D) {
  Tensor tensor = make_tensor<float>({2, 3, 4, 4}, getGPU());

  auto shape = tensor->shape();
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);
  EXPECT_EQ(shape[3], 4);
  EXPECT_EQ(tensor->size(), 96);
  EXPECT_TRUE(tensor->device_type() == DeviceType::GPU);
  EXPECT_FALSE(tensor->device_type() == DeviceType::CPU);
}

TEST_F(GPUTensorTest, ConstructorWithShape) {
  std::vector<size_t> shape = {2, 3, 4, 4};
  Tensor tensor = make_tensor<float>(shape, getGPU());

  EXPECT_EQ(tensor->shape(), shape);
  EXPECT_EQ(tensor->size(), 96);
  auto tensor_shape = tensor->shape();
  EXPECT_EQ(tensor_shape[0], 2);
  EXPECT_EQ(tensor_shape[1], 3);
  EXPECT_EQ(tensor_shape[2], 4);
  EXPECT_EQ(tensor_shape[3], 4);
  EXPECT_TRUE(tensor->device_type() == DeviceType::GPU);
}

TEST_F(GPUTensorTest, DeviceTypeCheck) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());

  EXPECT_EQ(tensor->device_type(), DeviceType::GPU);
  EXPECT_TRUE(tensor->device_type() == DeviceType::GPU);
  EXPECT_FALSE(tensor->device_type() == DeviceType::CPU);
}

TEST_F(GPUTensorTest, TensorAddition) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2}, getGPU());
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2}, getGPU());

  tensor1->fill(2.0);
  tensor2->fill(3.0);

  Tensor result = tensor1 + tensor2;

  Tensor cpu_result = result->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 5.0f);
}

TEST_F(GPUTensorTest, TensorSubtraction) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2}, getGPU());
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2}, getGPU());

  tensor1->fill(5.0);
  tensor2->fill(2.0);

  Tensor result = tensor1 - tensor2;

  Tensor cpu_result = result->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 3.0f);
}

TEST_F(GPUTensorTest, TensorMultiplication) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2}, getGPU());
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2}, getGPU());

  tensor1->fill(3.0);
  tensor2->fill(4.0);

  Tensor result = tensor1 * tensor2;

  Tensor cpu_result = result->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 12.0f);
}

TEST_F(GPUTensorTest, TensorDivision) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2}, getGPU());
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2}, getGPU());

  tensor1->fill(12.0);
  tensor2->fill(4.0);

  Tensor result = tensor1 / tensor2;

  Tensor cpu_result = result->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 3.0f);
}

TEST_F(GPUTensorTest, ScalarMultiplication) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());
  tensor->fill(3.0);

  Tensor result = tensor * 2.0;

  Tensor cpu_result = result->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 6.0f);
}

TEST_F(GPUTensorTest, ScalarDivision) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());
  tensor->fill(8.0);

  Tensor result = tensor / 2.0;

  Tensor cpu_result = result->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 4.0f);
}

TEST_F(GPUTensorTest, InPlaceAddition) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2}, getGPU());
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2}, getGPU());

  tensor1->fill(2.0);
  tensor2->fill(3.0);

  tensor1->add(tensor2);

  Tensor cpu_result = tensor1->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 5.0f);
}

TEST_F(GPUTensorTest, InPlaceSubtraction) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2}, getGPU());
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2}, getGPU());

  tensor1->fill(5.0);
  tensor2->fill(2.0);

  tensor1->sub(tensor2);

  Tensor cpu_result = tensor1->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 3.0f);
}

TEST_F(GPUTensorTest, InPlaceMultiplication) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2}, getGPU());
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2}, getGPU());

  tensor1->fill(3.0);
  tensor2->fill(4.0);

  tensor1->mul(tensor2);

  Tensor cpu_result = tensor1->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 12.0f);
}

TEST_F(GPUTensorTest, InPlaceScalarMultiplication) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());
  tensor->fill(3.0);

  tensor->mul_scalar(2.0);

  Tensor cpu_result = tensor->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 6.0f);
}

TEST_F(GPUTensorTest, InPlaceScalarDivision) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());
  tensor->fill(8.0);

  tensor->div_scalar(2.0);

  Tensor cpu_result = tensor->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 4.0f);
}

TEST_F(GPUTensorTest, SameShapeComparison) {
  Tensor tensor1 = make_tensor<float>({2, 3, 4, 5}, getGPU());
  Tensor tensor2 = make_tensor<float>({2, 3, 4, 5}, getGPU());
  Tensor tensor3 = make_tensor<float>({2, 3, 4, 6}, getGPU());

  EXPECT_TRUE(tensor1->shape() == tensor2->shape());
  EXPECT_FALSE(tensor1->shape() == tensor3->shape());
}

TEST_F(GPUTensorTest, AdditionShapeMismatch) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2}, getGPU());
  Tensor tensor2 = make_tensor<float>({1, 1, 3, 3}, getGPU());

  EXPECT_THROW(tensor1 + tensor2, std::invalid_argument);
}

TEST_F(GPUTensorTest, DivisionByZero) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());

  EXPECT_THROW(tensor / 0.0, std::invalid_argument);
}

TEST_F(GPUTensorTest, FillOperation) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());
  tensor->fill(42.0);

  Tensor cpu_result = tensor->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 42.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 42.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 42.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 42.0f);
}

TEST_F(GPUTensorTest, CloneOperation) {
  Tensor original = make_tensor<float>({1, 1, 2, 2}, getGPU());
  original->fill(5.0);

  Tensor cloned = original->clone();

  EXPECT_TRUE(original->shape() == cloned->shape());
  EXPECT_TRUE(cloned->device_type() == DeviceType::GPU);

  Tensor cpu_result = cloned->to_cpu();

  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 1}), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 0}), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 1, 1}), 5.0f);
}

TEST_F(GPUTensorTest, MeanCalculation) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());

  Tensor cpu_tensor = make_tensor<float>({1, 1, 2, 2});
  cpu_tensor->at<float>({0, 0, 0, 0}) = 1.0f;
  cpu_tensor->at<float>({0, 0, 0, 1}) = 2.0f;
  cpu_tensor->at<float>({0, 0, 1, 0}) = 3.0f;
  cpu_tensor->at<float>({0, 0, 1, 1}) = 4.0f;

  tensor = cpu_tensor->to_device(getGPU());

  double mean = tensor->mean();
  EXPECT_FLOAT_EQ(mean, 2.5);
}

TEST_F(GPUTensorTest, VarianceCalculation) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());

  Tensor cpu_tensor = make_tensor<float>({1, 1, 2, 2});
  cpu_tensor->at<float>({0, 0, 0, 0}) = 1.0f;
  cpu_tensor->at<float>({0, 0, 0, 1}) = 2.0f;
  cpu_tensor->at<float>({0, 0, 1, 0}) = 3.0f;
  cpu_tensor->at<float>({0, 0, 1, 1}) = 4.0f;

  tensor = cpu_tensor->to_device(getGPU());

  double variance = tensor->variance();
  EXPECT_NEAR(variance, 1.25, 1e-5);
}

TEST_F(GPUTensorTest, MoveConstructor) {
  Tensor original = make_tensor<float>({1, 1, 2, 2}, getGPU());
  original->fill(42.0);

  Tensor moved(std::move(original));

  EXPECT_EQ(moved->size(), 4);
  EXPECT_TRUE(moved->device_type() == DeviceType::GPU);

  Tensor cpu_result = moved->to_cpu();
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 42.0f);
  EXPECT_EQ(original.get(), nullptr);
}

TEST_F(GPUTensorTest, MoveAssignment) {
  Tensor original = make_tensor<float>({1, 1, 2, 2}, getGPU());
  original->fill(42.0);

  Tensor moved = make_tensor<float>({1, 1, 1, 1}, getGPU());
  moved = std::move(original);

  EXPECT_EQ(moved->size(), 4);
  EXPECT_TRUE(moved->device_type() == DeviceType::GPU);

  Tensor cpu_result = moved->to_cpu();
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 42.0f);
}

TEST_F(GPUTensorTest, MultiBatchAccess) {
  Tensor cpu_tensor = make_tensor<float>({2, 1, 2, 2});

  cpu_tensor->at<float>({0, 0, 0, 0}) = 1.0f;
  cpu_tensor->at<float>({1, 0, 0, 0}) = 2.0f;

  Tensor gpu_tensor = cpu_tensor->to_device(getGPU());
  Tensor result = gpu_tensor->to_cpu();

  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(result->at<float>({1, 0, 0, 0}), 2.0f);
}

TEST_F(GPUTensorTest, MultiChannelAccess) {
  Tensor cpu_tensor = make_tensor<float>({1, 3, 2, 2});

  cpu_tensor->at<float>({0, 0, 0, 0}) = 1.0f;
  cpu_tensor->at<float>({0, 1, 0, 0}) = 2.0f;
  cpu_tensor->at<float>({0, 2, 0, 0}) = 3.0f;

  Tensor gpu_tensor = cpu_tensor->to_device(getGPU());
  Tensor result = gpu_tensor->to_cpu();

  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 1, 0, 0}), 2.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 2, 0, 0}), 3.0f);
}

TEST_F(GPUTensorTest, CopyConstructor) {
  Tensor original = make_tensor<float>({1, 1, 2, 2}, getGPU());
  original->fill(42.0);

  Tensor copy = original->clone();

  EXPECT_EQ(copy->size(), original->size());
  EXPECT_TRUE(copy->shape() == original->shape());
  EXPECT_TRUE(copy->device_type() == DeviceType::GPU);

  Tensor cpu_copy = copy->to_cpu();
  EXPECT_FLOAT_EQ(cpu_copy->at<float>({0, 0, 0, 0}), 42.0f);
}

TEST_F(GPUTensorTest, ToCPU) {
  Tensor gpu_tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());
  gpu_tensor->fill(42.0);

  Tensor cpu_tensor = gpu_tensor->to_cpu();

  EXPECT_TRUE(cpu_tensor->device_type() == DeviceType::CPU);
  EXPECT_FALSE(cpu_tensor->device_type() == DeviceType::GPU);
  EXPECT_EQ(cpu_tensor->size(), 4);

  EXPECT_FLOAT_EQ(cpu_tensor->at<float>({0, 0, 0, 0}), 42.0f);
  EXPECT_FLOAT_EQ(cpu_tensor->at<float>({0, 0, 0, 1}), 42.0f);
  EXPECT_FLOAT_EQ(cpu_tensor->at<float>({0, 0, 1, 0}), 42.0f);
  EXPECT_FLOAT_EQ(cpu_tensor->at<float>({0, 0, 1, 1}), 42.0f);
}

TEST_F(GPUTensorTest, ToGPUFromCPU) {
  Tensor cpu_tensor = make_tensor<float>({1, 1, 2, 2});
  cpu_tensor->at<float>({0, 0, 0, 0}) = 1.0f;
  cpu_tensor->at<float>({0, 0, 0, 1}) = 2.0f;
  cpu_tensor->at<float>({0, 0, 1, 0}) = 3.0f;
  cpu_tensor->at<float>({0, 0, 1, 1}) = 4.0f;

  Tensor gpu_tensor = cpu_tensor->to_device(getGPU());

  EXPECT_TRUE(gpu_tensor->device_type() == DeviceType::GPU);
  EXPECT_FALSE(gpu_tensor->device_type() == DeviceType::CPU);
  EXPECT_EQ(gpu_tensor->size(), 4);

  Tensor result = gpu_tensor->to_cpu();
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 1}), 4.0f);
}

TEST_F(GPUTensorTest, ToGPUIdempotent) {
  Tensor gpu_tensor = make_tensor<float>({1, 1, 2, 2}, getGPU());
  gpu_tensor->fill(42.0);

  Tensor still_gpu = gpu_tensor->to_device(getGPU());

  EXPECT_TRUE(still_gpu->device_type() == DeviceType::GPU);

  Tensor cpu_result = still_gpu->to_cpu();
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 42.0f);
}

TEST_F(GPUTensorTest, ToCPUIdempotent) {
  Tensor cpu_tensor = make_tensor<float>({1, 1, 2, 2});
  cpu_tensor->at<float>({0, 0, 0, 0}) = 42.0f;

  Tensor still_cpu = cpu_tensor->to_cpu();

  EXPECT_TRUE(still_cpu->device_type() == DeviceType::CPU);
  EXPECT_FLOAT_EQ(still_cpu->at<float>({0, 0, 0, 0}), 42.0f);
}

TEST_F(GPUTensorTest, FillRandomUniform) {
  Tensor tensor = make_tensor<float>({1, 10, 10, 10}, getGPU());
  tensor->fill_random_uniform(1.0);

  Tensor cpu_result = tensor->to_cpu();

  bool all_in_range = true;
  for (size_t i = 0; i < cpu_result->size(); ++i) {
    float val = cpu_result->at<float>({i / 4, 0, (i / 2) % 2, i % 2});
    if (val < 0.0f || val > 1.0f) {
      all_in_range = false;
      break;
    }
  }
  EXPECT_TRUE(all_in_range);

  float first_val = cpu_result->at<float>({0, 0, 0, 0});
  bool has_different = false;
  for (size_t i = 1; i < std::min(cpu_result->size(), size_t(100)); ++i) {
    if (std::abs(cpu_result->at<float>({i / 4, 0, (i / 2) % 2, i % 2}) - first_val) > 1e-6f) {
      has_different = true;
      break;
    }
  }
  EXPECT_TRUE(has_different);
}

TEST_F(GPUTensorTest, FillRandomNormal) {
  Tensor tensor = make_tensor<float>({1, 10, 10, 10}, getGPU());
  tensor->fill_random_normal(0.0, 1.0);

  Tensor cpu_result = tensor->to_cpu();

  float first_val = cpu_result->at<float>({0, 0, 0, 0});
  bool has_different = false;
  for (size_t i = 1; i < std::min(cpu_result->size(), size_t(100)); ++i) {
    if (std::abs(cpu_result->at<float>({i / 4, 0, (i / 2) % 2, i % 2}) - first_val) > 1e-6f) {
      has_different = true;
      break;
    }
  }
  EXPECT_TRUE(has_different);

  float sum = 0.0f;
  for (size_t i = 0; i < cpu_result->size(); ++i) {
    sum += cpu_result->at<float>({i / 4, 0, (i / 2) % 2, i % 2});
  }
  float mean = sum / cpu_result->size();

  EXPECT_NEAR(mean, 0.0f, 0.2f);
}

class GPUTensorSizeTest
    : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t, size_t>> {
protected:
  static void SetUpTestSuite() {
    DeviceManager &manager = DeviceManager::getInstance();
    if (manager.getAvailableDeviceIDs().empty()) {
      initializeDefaultDevices();
    }
  }

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
      GTEST_SKIP() << "No GPU device available, skipping GPU tensor tests";
    }
  }

  static void TearDownTestSuite() {}

  bool has_gpu_;
};

TEST_P(GPUTensorSizeTest, ConstructorAndSize) {
  auto [batch, channels, height, width] = GetParam();
  Tensor tensor = make_tensor<float>({batch, channels, height, width}, getGPU());

  auto tensor_shape = tensor->shape();
  EXPECT_EQ(tensor_shape[0], batch);
  EXPECT_EQ(tensor_shape[1], channels);
  EXPECT_EQ(tensor_shape[2], height);
  EXPECT_EQ(tensor_shape[3], width);
  EXPECT_EQ(tensor->size(), batch * channels * height * width);
  EXPECT_TRUE(tensor->device_type() == DeviceType::GPU);
}

INSTANTIATE_TEST_SUITE_P(DifferentShapes, GPUTensorSizeTest,
                         ::testing::Values(std::make_tuple(1, 1, 1, 1),
                                           std::make_tuple(1, 3, 32, 32),
                                           std::make_tuple(16, 64, 28, 28),
                                           std::make_tuple(32, 128, 14, 14)));

TEST_F(GPUTensorTest, LargeTensorOperations) {
  Tensor tensor1 = make_tensor<float>({4, 16, 64, 64}, getGPU());
  Tensor tensor2 = make_tensor<float>({4, 16, 64, 64}, getGPU());

  tensor1->fill(1.5);
  tensor2->fill(2.5);

  Tensor result = tensor1 + tensor2;

  EXPECT_EQ(result->size(), 4 * 16 * 64 * 64);
  EXPECT_TRUE(result->device_type() == DeviceType::GPU);

  Tensor cpu_result = result->to_cpu();
  EXPECT_FLOAT_EQ(cpu_result->at<float>({0, 0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({1, 5, 10, 20}), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result->at<float>({3, 15, 63, 63}), 4.0f);
}

TEST(GPUTensorFloatingPointTest, FloatingPointComparisons) {
  initializeDefaultDevices();

  DeviceManager &manager = DeviceManager::getInstance();
  std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

  bool has_gpu = false;
  csref<Device> gpu_device = getGPU();

  if (!has_gpu) {
    GTEST_SKIP() << "No GPU device available";
  }

  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2}, gpu_device);
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2}, gpu_device);

  tensor1->fill(0.1 + 0.2);
  tensor2->fill(0.3);

  Tensor diff = tensor1 - tensor2;
  Tensor cpu_diff = diff->to_cpu();

  for (size_t i = 0; i < cpu_diff->size(); ++i) {
    EXPECT_NEAR(cpu_diff->at<float>({i / 4, 0, (i / 2) % 2, i % 2}), 0.0f, 1e-6f);
  }
}
