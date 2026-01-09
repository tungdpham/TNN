#include "device/device_manager.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace tnn;

// Test fixture for GPU tensor operations
class GPUTensorTest : public ::testing::Test {
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
      if (device.device_type() == DeviceType::GPU) {
        gpu_device_ = &device;
        has_gpu_ = true;
        break;
      }
    }

    if (!has_gpu_) {
      GTEST_SKIP() << "No GPU device available, skipping GPU tensor tests";
    }

    // All tensors are 4D for NCHW layout
    small_tensor = Tensor<float>({1, 1, 2, 2}, gpu_device_);
    small_tensor.fill(1.0f);

    large_tensor = Tensor<float>({2, 3, 4, 4}, gpu_device_);
    large_tensor.fill(2.0f);
  }

  void TearDown() override {
    // Clear tensors to release GPU memory before moving to next test
    small_tensor = Tensor<float>();
    large_tensor = Tensor<float>();
  }

  static void TearDownTestSuite() {}

  bool has_gpu_;
  const Device *gpu_device_;
  Tensor<float> small_tensor;
  Tensor<float> large_tensor;
};

// Basic constructor tests
TEST_F(GPUTensorTest, Constructor4D) {
  Tensor<float> tensor({2, 3, 4, 4}, gpu_device_);

  auto shape = tensor.shape();
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);
  EXPECT_EQ(shape[3], 4);
  EXPECT_EQ(tensor.size(), 96); // 2*3*4*4
  EXPECT_TRUE(tensor.is_on_gpu());
  EXPECT_FALSE(tensor.is_on_cpu());
}

TEST_F(GPUTensorTest, ConstructorWithShape) {
  std::vector<size_t> shape = {2, 3, 4, 4};
  Tensor<float> tensor(shape, gpu_device_);

  EXPECT_EQ(tensor.shape(), shape);
  EXPECT_EQ(tensor.size(), 96);
  auto tensor_shape = tensor.shape();
  EXPECT_EQ(tensor_shape[0], 2);
  EXPECT_EQ(tensor_shape[1], 3);
  EXPECT_EQ(tensor_shape[2], 4);
  EXPECT_EQ(tensor_shape[3], 4);
  EXPECT_TRUE(tensor.is_on_gpu());
}

// Device type tests
TEST_F(GPUTensorTest, DeviceTypeCheck) {
  Tensor<float> tensor({1, 1, 2, 2}, gpu_device_);

  EXPECT_EQ(tensor.device_type(), DeviceType::GPU);
  EXPECT_TRUE(tensor.is_on_gpu());
  EXPECT_FALSE(tensor.is_on_cpu());
}

// Arithmetic operations tests
TEST_F(GPUTensorTest, TensorAddition) {
  Tensor<float> tensor1({1, 1, 2, 2}, gpu_device_);
  Tensor<float> tensor2({1, 1, 2, 2}, gpu_device_);

  tensor1.fill(2.0f);
  tensor2.fill(3.0f);

  Tensor<float> result = tensor1 + tensor2;

  // Transfer to CPU for verification
  Tensor<float> cpu_result = result.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 5.0f);
}

TEST_F(GPUTensorTest, TensorSubtraction) {
  Tensor<float> tensor1({1, 1, 2, 2}, gpu_device_);
  Tensor<float> tensor2({1, 1, 2, 2}, gpu_device_);

  tensor1.fill(5.0f);
  tensor2.fill(2.0f);

  Tensor<float> result = tensor1 - tensor2;

  Tensor<float> cpu_result = result.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 3.0f);
}

TEST_F(GPUTensorTest, TensorMultiplication) {
  Tensor<float> tensor1({1, 1, 2, 2}, gpu_device_);
  Tensor<float> tensor2({1, 1, 2, 2}, gpu_device_);

  tensor1.fill(3.0f);
  tensor2.fill(4.0f);

  Tensor<float> result = tensor1 * tensor2;

  Tensor<float> cpu_result = result.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 12.0f);
}

TEST_F(GPUTensorTest, TensorDivision) {
  Tensor<float> tensor1({1, 1, 2, 2}, gpu_device_);
  Tensor<float> tensor2({1, 1, 2, 2}, gpu_device_);

  tensor1.fill(12.0f);
  tensor2.fill(4.0f);

  Tensor<float> result = tensor1 / tensor2;

  Tensor<float> cpu_result = result.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 3.0f);
}

TEST_F(GPUTensorTest, ScalarMultiplication) {
  Tensor<float> tensor({1, 1, 2, 2}, gpu_device_);
  tensor.fill(3.0f);

  Tensor<float> result = tensor * 2.0f;

  Tensor<float> cpu_result = result.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 6.0f);
}

TEST_F(GPUTensorTest, ScalarDivision) {
  Tensor<float> tensor({1, 1, 2, 2}, gpu_device_);
  tensor.fill(8.0f);

  Tensor<float> result = tensor / 2.0f;

  Tensor<float> cpu_result = result.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 4.0f);
}

// In-place operations tests
TEST_F(GPUTensorTest, InPlaceAddition) {
  Tensor<float> tensor1({1, 1, 2, 2}, gpu_device_);
  Tensor<float> tensor2({1, 1, 2, 2}, gpu_device_);

  tensor1.fill(2.0f);
  tensor2.fill(3.0f);

  tensor1 += tensor2;

  Tensor<float> cpu_result = tensor1.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 5.0f);
}

TEST_F(GPUTensorTest, InPlaceSubtraction) {
  Tensor<float> tensor1({1, 1, 2, 2}, gpu_device_);
  Tensor<float> tensor2({1, 1, 2, 2}, gpu_device_);

  tensor1.fill(5.0f);
  tensor2.fill(2.0f);

  tensor1 -= tensor2;

  Tensor<float> cpu_result = tensor1.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 3.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 3.0f);
}

TEST_F(GPUTensorTest, InPlaceMultiplication) {
  Tensor<float> tensor1({1, 1, 2, 2}, gpu_device_);
  Tensor<float> tensor2({1, 1, 2, 2}, gpu_device_);

  tensor1.fill(3.0f);
  tensor2.fill(4.0f);

  tensor1 *= tensor2;

  Tensor<float> cpu_result = tensor1.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 12.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 12.0f);
}

TEST_F(GPUTensorTest, InPlaceScalarMultiplication) {
  Tensor<float> tensor({1, 1, 2, 2}, gpu_device_);
  tensor.fill(3.0f);

  tensor *= 2.0f;

  Tensor<float> cpu_result = tensor.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 6.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 6.0f);
}

TEST_F(GPUTensorTest, InPlaceScalarDivision) {
  Tensor<float> tensor({1, 1, 2, 2}, gpu_device_);
  tensor.fill(8.0f);

  tensor /= 2.0f;

  Tensor<float> cpu_result = tensor.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 4.0f);
}

// Shape validation tests
TEST_F(GPUTensorTest, SameShapeComparison) {
  Tensor<float> tensor1({2, 3, 4, 5}, gpu_device_);
  Tensor<float> tensor2({2, 3, 4, 5}, gpu_device_);
  Tensor<float> tensor3({2, 3, 4, 6}, gpu_device_);

  EXPECT_TRUE(tensor1.same_shape(tensor2));
  EXPECT_FALSE(tensor1.same_shape(tensor3));
}

// Error handling tests
TEST_F(GPUTensorTest, AdditionShapeMismatch) {
  Tensor<float> tensor1({1, 1, 2, 2}, gpu_device_);
  Tensor<float> tensor2({1, 1, 3, 3}, gpu_device_);

  EXPECT_THROW(tensor1 + tensor2, std::invalid_argument);
}

TEST_F(GPUTensorTest, DivisionByZero) {
  Tensor<float> tensor({1, 1, 2, 2}, gpu_device_);

  EXPECT_THROW(tensor / 0.0f, std::invalid_argument);
}

// Data manipulation tests
TEST_F(GPUTensorTest, FillOperation) {
  Tensor<float> tensor({1, 1, 2, 2}, gpu_device_);
  tensor.fill(42.0f);

  Tensor<float> cpu_result = tensor.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 42.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 42.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 42.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 42.0f);
}

TEST_F(GPUTensorTest, CloneOperation) {
  Tensor<float> original({1, 1, 2, 2}, gpu_device_);
  original.fill(5.0f);

  Tensor<float> cloned = original.clone();

  EXPECT_TRUE(original.same_shape(cloned));
  EXPECT_TRUE(cloned.is_on_gpu());

  Tensor<float> cpu_result = cloned.to_cpu();

  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 5.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 5.0f);
}

// Statistical operations tests
TEST_F(GPUTensorTest, MeanCalculation) {
  Tensor<float> tensor({1, 1, 2, 2}, gpu_device_);

  // Create CPU tensor with known values, then transfer to GPU
  Tensor<float> cpu_tensor({1, 1, 2, 2});
  cpu_tensor(0, 0, 0, 0) = 1.0f;
  cpu_tensor(0, 0, 0, 1) = 2.0f;
  cpu_tensor(0, 0, 1, 0) = 3.0f;
  cpu_tensor(0, 0, 1, 1) = 4.0f;

  tensor = cpu_tensor.to_gpu();

  float mean = tensor.mean();
  EXPECT_FLOAT_EQ(mean, 2.5f); // (1+2+3+4)/4 = 2.5
}

TEST_F(GPUTensorTest, VarianceCalculation) {
  Tensor<float> tensor({1, 1, 2, 2}, gpu_device_);

  // Create CPU tensor with known values, then transfer to GPU
  Tensor<float> cpu_tensor({1, 1, 2, 2});
  cpu_tensor(0, 0, 0, 0) = 1.0f;
  cpu_tensor(0, 0, 0, 1) = 2.0f;
  cpu_tensor(0, 0, 1, 0) = 3.0f;
  cpu_tensor(0, 0, 1, 1) = 4.0f;

  tensor = cpu_tensor.to_gpu();

  float variance = tensor.variance();
  EXPECT_NEAR(variance, 1.25f, 1e-5f); // Variance of [1,2,3,4] is 1.25
}

// Move semantics tests
TEST_F(GPUTensorTest, MoveConstructor) {
  Tensor<float> original({1, 1, 2, 2}, gpu_device_);
  original.fill(42.0f);

  Tensor<float> moved(std::move(original));

  EXPECT_EQ(moved.size(), 4);
  EXPECT_TRUE(moved.is_on_gpu());

  Tensor<float> cpu_result = moved.to_cpu();
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 42.0f);
  EXPECT_EQ(original.data(), nullptr); // Original should be empty after move
}

TEST_F(GPUTensorTest, MoveAssignment) {
  Tensor<float> original({1, 1, 2, 2}, gpu_device_);
  original.fill(42.0f);

  Tensor<float> moved({1, 1, 1, 1}, gpu_device_);
  moved = std::move(original);

  EXPECT_EQ(moved.size(), 4);
  EXPECT_TRUE(moved.is_on_gpu());

  Tensor<float> cpu_result = moved.to_cpu();
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 42.0f);
}

// Multi-batch and multi-channel tests
TEST_F(GPUTensorTest, MultiBatchAccess) {
  Tensor<float> cpu_tensor({2, 1, 2, 2}); // 2 batches

  // Set different values for each batch
  cpu_tensor(0, 0, 0, 0) = 1.0f; // batch 0
  cpu_tensor(1, 0, 0, 0) = 2.0f; // batch 1

  Tensor<float> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float> result = gpu_tensor.to_cpu();

  EXPECT_FLOAT_EQ(result(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(result(1, 0, 0, 0), 2.0f);
}

TEST_F(GPUTensorTest, MultiChannelAccess) {
  Tensor<float> cpu_tensor({1, 3, 2, 2}); // 3 channels

  // Set different values for each channel
  cpu_tensor(0, 0, 0, 0) = 1.0f; // channel 0
  cpu_tensor(0, 1, 0, 0) = 2.0f; // channel 1
  cpu_tensor(0, 2, 0, 0) = 3.0f; // channel 2

  Tensor<float> gpu_tensor = cpu_tensor.to_gpu();
  Tensor<float> result = gpu_tensor.to_cpu();

  EXPECT_FLOAT_EQ(result(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(result(0, 1, 0, 0), 2.0f);
  EXPECT_FLOAT_EQ(result(0, 2, 0, 0), 3.0f);
}

// Copy constructor test
TEST_F(GPUTensorTest, CopyConstructor) {
  Tensor<float> original({1, 1, 2, 2}, gpu_device_);
  original.fill(42.0f);

  Tensor<float> copy = original.clone();

  EXPECT_EQ(copy.size(), original.size());
  EXPECT_TRUE(copy.same_shape(original));
  EXPECT_TRUE(copy.is_on_gpu());

  Tensor<float> cpu_copy = copy.to_cpu();
  EXPECT_FLOAT_EQ(cpu_copy(0, 0, 0, 0), 42.0f);
}

// Device transfer tests
TEST_F(GPUTensorTest, ToCPU) {
  Tensor<float> gpu_tensor({1, 1, 2, 2}, gpu_device_);
  gpu_tensor.fill(42.0f);

  Tensor<float> cpu_tensor = gpu_tensor.to_cpu();

  EXPECT_TRUE(cpu_tensor.is_on_cpu());
  EXPECT_FALSE(cpu_tensor.is_on_gpu());
  EXPECT_EQ(cpu_tensor.size(), 4);

  EXPECT_FLOAT_EQ(cpu_tensor(0, 0, 0, 0), 42.0f);
  EXPECT_FLOAT_EQ(cpu_tensor(0, 0, 0, 1), 42.0f);
  EXPECT_FLOAT_EQ(cpu_tensor(0, 0, 1, 0), 42.0f);
  EXPECT_FLOAT_EQ(cpu_tensor(0, 0, 1, 1), 42.0f);
}

TEST_F(GPUTensorTest, ToGPUFromCPU) {
  Tensor<float> cpu_tensor({1, 1, 2, 2});
  cpu_tensor(0, 0, 0, 0) = 1.0f;
  cpu_tensor(0, 0, 0, 1) = 2.0f;
  cpu_tensor(0, 0, 1, 0) = 3.0f;
  cpu_tensor(0, 0, 1, 1) = 4.0f;

  Tensor<float> gpu_tensor = cpu_tensor.to_gpu();

  EXPECT_TRUE(gpu_tensor.is_on_gpu());
  EXPECT_FALSE(gpu_tensor.is_on_cpu());
  EXPECT_EQ(gpu_tensor.size(), 4);

  // Verify by transferring back to CPU
  Tensor<float> result = gpu_tensor.to_cpu();
  EXPECT_FLOAT_EQ(result(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 0, 1), 2.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 0), 3.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 1), 4.0f);
}

TEST_F(GPUTensorTest, ToGPUIdempotent) {
  Tensor<float> gpu_tensor({1, 1, 2, 2}, gpu_device_);
  gpu_tensor.fill(42.0f);

  Tensor<float> still_gpu = gpu_tensor.to_gpu();

  EXPECT_TRUE(still_gpu.is_on_gpu());

  Tensor<float> cpu_result = still_gpu.to_cpu();
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 42.0f);
}

TEST_F(GPUTensorTest, ToCPUIdempotent) {
  Tensor<float> cpu_tensor({1, 1, 2, 2});
  cpu_tensor(0, 0, 0, 0) = 42.0f;

  Tensor<float> still_cpu = cpu_tensor.to_cpu();

  EXPECT_TRUE(still_cpu.is_on_cpu());
  EXPECT_FLOAT_EQ(still_cpu(0, 0, 0, 0), 42.0f);
}

// Random fill tests
TEST_F(GPUTensorTest, FillRandomUniform) {
  Tensor<float> tensor({1, 10, 10, 10}, gpu_device_);
  tensor.fill_random_uniform(1.0f);

  Tensor<float> cpu_result = tensor.to_cpu();

  // Check that values are in expected range [0, 1]
  bool all_in_range = true;
  for (size_t i = 0; i < cpu_result.size(); ++i) {
    float val = cpu_result.data()[i];
    if (val < 0.0f || val > 1.0f) {
      all_in_range = false;
      break;
    }
  }
  EXPECT_TRUE(all_in_range);

  // Check that not all values are the same (randomness check)
  float first_val = cpu_result.data()[0];
  bool has_different = false;
  for (size_t i = 1; i < std::min(cpu_result.size(), size_t(100)); ++i) {
    if (std::abs(cpu_result.data()[i] - first_val) > 1e-6f) {
      has_different = true;
      break;
    }
  }
  EXPECT_TRUE(has_different);
}

TEST_F(GPUTensorTest, FillRandomNormal) {
  Tensor<float> tensor({1, 10, 10, 10}, gpu_device_);
  tensor.fill_random_normal(0.0f, 1.0f);

  Tensor<float> cpu_result = tensor.to_cpu();

  // Check that not all values are the same (randomness check)
  float first_val = cpu_result.data()[0];
  bool has_different = false;
  for (size_t i = 1; i < std::min(cpu_result.size(), size_t(100)); ++i) {
    if (std::abs(cpu_result.data()[i] - first_val) > 1e-6f) {
      has_different = true;
      break;
    }
  }
  EXPECT_TRUE(has_different);

  // Calculate approximate mean and std dev
  float sum = 0.0f;
  for (size_t i = 0; i < cpu_result.size(); ++i) {
    sum += cpu_result.data()[i];
  }
  float mean = sum / cpu_result.size();

  // Mean should be close to 0.0 for large sample
  EXPECT_NEAR(mean, 0.0f, 0.2f);
}

// Batch copy tests
TEST_F(GPUTensorTest, CopyBatch) {
  Tensor<float> source({2, 1, 2, 2}, gpu_device_);
  Tensor<float> dest({2, 1, 2, 2}, gpu_device_);

  // Set up source tensor with different values per batch
  Tensor<float> cpu_source({2, 1, 2, 2});
  for (size_t i = 0; i < 4; ++i) {
    cpu_source.data()[i] = 1.0f;     // batch 0
    cpu_source.data()[i + 4] = 2.0f; // batch 1
  }
  source = cpu_source.to_gpu();

  dest.fill(0.0f);

  // Copy batch 1 from source to batch 0 of dest
  dest.copy_batch(source, 1, 0);

  Tensor<float> cpu_result = dest.to_cpu();

  // First batch should have values from source batch 1
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 2.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 1), 2.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 0), 2.0f);
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 1, 1), 2.0f);
}

TEST_F(GPUTensorTest, CopyBatchInvalidIndex) {
  Tensor<float> source({2, 1, 2, 2}, gpu_device_);
  Tensor<float> dest({2, 1, 2, 2}, gpu_device_);

  EXPECT_THROW(dest.copy_batch(source, 5, 0), std::invalid_argument);
  EXPECT_THROW(dest.copy_batch(source, 0, 5), std::invalid_argument);
}

// Parameterized tests for different tensor sizes
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
        gpu_device_ = &device;
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
  const Device *gpu_device_;
};

TEST_P(GPUTensorSizeTest, ConstructorAndSize) {
  auto [batch, channels, height, width] = GetParam();
  Tensor<float> tensor({batch, channels, height, width}, gpu_device_);

  auto tensor_shape = tensor.shape();
  EXPECT_EQ(tensor_shape[0], batch);
  EXPECT_EQ(tensor_shape[1], channels);
  EXPECT_EQ(tensor_shape[2], height);
  EXPECT_EQ(tensor_shape[3], width);
  EXPECT_EQ(tensor.size(), batch * channels * height * width);
  EXPECT_TRUE(tensor.is_on_gpu());
}

INSTANTIATE_TEST_SUITE_P(DifferentShapes, GPUTensorSizeTest,
                         ::testing::Values(std::make_tuple(1, 1, 1, 1),
                                           std::make_tuple(1, 3, 32, 32),
                                           std::make_tuple(16, 64, 28, 28),
                                           std::make_tuple(32, 128, 14, 14)));

// Large tensor tests
TEST_F(GPUTensorTest, LargeTensorOperations) {
  // Test with larger tensors to ensure GPU operations scale
  Tensor<float> tensor1({4, 16, 64, 64}, gpu_device_);
  Tensor<float> tensor2({4, 16, 64, 64}, gpu_device_);

  tensor1.fill(1.5f);
  tensor2.fill(2.5f);

  Tensor<float> result = tensor1 + tensor2;

  EXPECT_EQ(result.size(), 4 * 16 * 64 * 64);
  EXPECT_TRUE(result.is_on_gpu());

  // Spot check a few values
  Tensor<float> cpu_result = result.to_cpu();
  EXPECT_FLOAT_EQ(cpu_result(0, 0, 0, 0), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result(1, 5, 10, 20), 4.0f);
  EXPECT_FLOAT_EQ(cpu_result(3, 15, 63, 63), 4.0f);
}

// Floating point precision tests
TEST(GPUTensorFloatingPointTest, FloatingPointComparisons) {
  initializeDefaultDevices();

  DeviceManager &manager = DeviceManager::getInstance();
  std::vector<std::string> device_ids = manager.getAvailableDeviceIDs();

  bool has_gpu = false;
  const Device *gpu_device = nullptr;
  for (const std::string &id : device_ids) {
    const Device &device = manager.getDevice(id);
    if (device.device_type() == DeviceType::GPU) {
      gpu_device = &device;
      has_gpu = true;
      break;
    }
  }

  if (!has_gpu) {
    GTEST_SKIP() << "No GPU device available";
  }

  Tensor<float> tensor1({1, 1, 2, 2}, gpu_device);
  Tensor<float> tensor2({1, 1, 2, 2}, gpu_device);

  // Use values that might have floating point precision issues
  tensor1.fill(0.1f + 0.2f); // This might not be exactly 0.3
  tensor2.fill(0.3f);

  Tensor<float> diff = tensor1 - tensor2;
  Tensor<float> cpu_diff = diff.to_cpu();

  // Check that difference is small (within tolerance)
  for (size_t i = 0; i < cpu_diff.size(); ++i) {
    EXPECT_NEAR(cpu_diff.data()[i], 0.0f, 1e-6f);
  }
}
