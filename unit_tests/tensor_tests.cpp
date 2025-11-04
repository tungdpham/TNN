#include "tensor/tensor.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Test fixture for common tensor operations
class TensorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // All tensors are 4D for NCHW layout
    small_tensor = Tensor<float, NCHW>(1, 1, 2, 2); // batch=1, channels=1, height=2, width=2
    small_tensor.fill(1.0f);

    large_tensor = Tensor<float, NCHW>(2, 3, 4, 4); // batch=2, channels=3, height=4, width=4
    large_tensor.fill(2.0f);
  }

  Tensor<float, NCHW> small_tensor;
  Tensor<float, NCHW> large_tensor;
};

// Basic constructor tests
TEST_F(TensorTest, Constructor4D) {
  Tensor<float, NCHW> tensor(2, 3, 4, 4);

  EXPECT_EQ(tensor.batch_size(), 2);
  EXPECT_EQ(tensor.channels(), 3);
  EXPECT_EQ(tensor.height(), 4);
  EXPECT_EQ(tensor.width(), 4);
  EXPECT_EQ(tensor.size(), 96); // 2*3*4*4
}

TEST_F(TensorTest, ConstructorWithShape) {
  std::vector<size_t> shape = {2, 3, 4, 4};
  Tensor<float, NCHW> tensor(shape);

  EXPECT_EQ(tensor.shape(), shape);
  EXPECT_EQ(tensor.size(), 96);
  EXPECT_EQ(tensor.batch_size(), 2);
  EXPECT_EQ(tensor.channels(), 3);
  EXPECT_EQ(tensor.height(), 4);
  EXPECT_EQ(tensor.width(), 4);
}

// Element access tests - all using 4D indices (batch, channel, height, width)
TEST_F(TensorTest, ElementAccess) {
  Tensor<float, NCHW> tensor(1, 1, 2, 2);

  // Set values using 4D indices
  tensor(0, 0, 0, 0) = 1.0f;
  tensor(0, 0, 0, 1) = 2.0f;
  tensor(0, 0, 1, 0) = 3.0f;
  tensor(0, 0, 1, 1) = 4.0f;

  // Check values
  EXPECT_FLOAT_EQ(tensor(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(tensor(0, 0, 0, 1), 2.0f);
  EXPECT_FLOAT_EQ(tensor(0, 0, 1, 0), 3.0f);
  EXPECT_FLOAT_EQ(tensor(0, 0, 1, 1), 4.0f);
}

// Arithmetic operations tests
TEST_F(TensorTest, TensorAddition) {
  Tensor<float, NCHW> tensor1(1, 1, 2, 2);
  Tensor<float, NCHW> tensor2(1, 1, 2, 2);

  tensor1.fill(2.0f);
  tensor2.fill(3.0f);

  Tensor<float, NCHW> result = tensor1 + tensor2;

  EXPECT_FLOAT_EQ(result(0, 0, 0, 0), 5.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 0, 1), 5.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 0), 5.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 1), 5.0f);
}

TEST_F(TensorTest, TensorSubtraction) {
  Tensor<float, NCHW> tensor1(1, 1, 2, 2);
  Tensor<float, NCHW> tensor2(1, 1, 2, 2);

  tensor1.fill(5.0f);
  tensor2.fill(2.0f);

  Tensor<float, NCHW> result = tensor1 - tensor2;

  EXPECT_FLOAT_EQ(result(0, 0, 0, 0), 3.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 0, 1), 3.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 0), 3.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 1), 3.0f);
}

TEST_F(TensorTest, TensorMultiplication) {
  Tensor<float, NCHW> tensor1(1, 1, 2, 2);
  Tensor<float, NCHW> tensor2(1, 1, 2, 2);

  tensor1.fill(3.0f);
  tensor2.fill(4.0f);

  Tensor<float, NCHW> result = tensor1 * tensor2;

  EXPECT_FLOAT_EQ(result(0, 0, 0, 0), 12.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 0, 1), 12.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 0), 12.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 1), 12.0f);
}

TEST_F(TensorTest, ScalarMultiplication) {
  Tensor<float, NCHW> tensor(1, 1, 2, 2);
  tensor.fill(3.0f);

  Tensor<float, NCHW> result = tensor * 2.0f;

  EXPECT_FLOAT_EQ(result(0, 0, 0, 0), 6.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 0, 1), 6.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 0), 6.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 1), 6.0f);
}

TEST_F(TensorTest, ScalarDivision) {
  Tensor<float, NCHW> tensor(1, 1, 2, 2);
  tensor.fill(8.0f);

  Tensor<float, NCHW> result = tensor / 2.0f;

  EXPECT_FLOAT_EQ(result(0, 0, 0, 0), 4.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 0, 1), 4.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 0), 4.0f);
  EXPECT_FLOAT_EQ(result(0, 0, 1, 1), 4.0f);
}

// In-place operations tests
TEST_F(TensorTest, InPlaceAddition) {
  Tensor<float, NCHW> tensor1(1, 1, 2, 2);
  Tensor<float, NCHW> tensor2(1, 1, 2, 2);

  tensor1.fill(2.0f);
  tensor2.fill(3.0f);

  tensor1 += tensor2;

  EXPECT_FLOAT_EQ(tensor1(0, 0, 0, 0), 5.0f);
  EXPECT_FLOAT_EQ(tensor1(0, 0, 0, 1), 5.0f);
  EXPECT_FLOAT_EQ(tensor1(0, 0, 1, 0), 5.0f);
  EXPECT_FLOAT_EQ(tensor1(0, 0, 1, 1), 5.0f);
}

TEST_F(TensorTest, InPlaceScalarMultiplication) {
  Tensor<float, NCHW> tensor(1, 1, 2, 2);
  tensor.fill(3.0f);

  tensor *= 2.0f;

  EXPECT_FLOAT_EQ(tensor(0, 0, 0, 0), 6.0f);
  EXPECT_FLOAT_EQ(tensor(0, 0, 0, 1), 6.0f);
  EXPECT_FLOAT_EQ(tensor(0, 0, 1, 0), 6.0f);
  EXPECT_FLOAT_EQ(tensor(0, 0, 1, 1), 6.0f);
}

// Shape validation tests
TEST_F(TensorTest, SameShapeComparison) {
  Tensor<float, NCHW> tensor1(2, 3, 4, 5);
  Tensor<float, NCHW> tensor2(2, 3, 4, 5);
  Tensor<float, NCHW> tensor3(2, 3, 4, 6);

  EXPECT_TRUE(tensor1.same_shape(tensor2));
  EXPECT_FALSE(tensor1.same_shape(tensor3));
}

// Error handling tests
TEST_F(TensorTest, AdditionShapeMismatch) {
  Tensor<float, NCHW> tensor1(1, 1, 2, 2);
  Tensor<float, NCHW> tensor2(1, 1, 3, 3);

  EXPECT_THROW(tensor1 + tensor2, std::invalid_argument);
}

TEST_F(TensorTest, DivisionByZero) {
  Tensor<float, NCHW> tensor(1, 1, 2, 2);

  EXPECT_THROW(tensor / 0.0f, std::invalid_argument);
}

// Data manipulation tests
TEST_F(TensorTest, FillOperation) {
  Tensor<float, NCHW> tensor(1, 1, 2, 2);
  tensor.fill(42.0f);

  EXPECT_FLOAT_EQ(tensor(0, 0, 0, 0), 42.0f);
  EXPECT_FLOAT_EQ(tensor(0, 0, 0, 1), 42.0f);
  EXPECT_FLOAT_EQ(tensor(0, 0, 1, 0), 42.0f);
  EXPECT_FLOAT_EQ(tensor(0, 0, 1, 1), 42.0f);
}

TEST_F(TensorTest, CloneOperation) {
  Tensor<float, NCHW> original(1, 1, 2, 2);
  original.fill(5.0f);

  Tensor<float, NCHW> cloned = original.clone();

  EXPECT_TRUE(original.same_shape(cloned));
  EXPECT_FLOAT_EQ(cloned(0, 0, 0, 0), 5.0f);
  EXPECT_FLOAT_EQ(cloned(0, 0, 0, 1), 5.0f);
  EXPECT_FLOAT_EQ(cloned(0, 0, 1, 0), 5.0f);
  EXPECT_FLOAT_EQ(cloned(0, 0, 1, 1), 5.0f);
}

// Statistical operations tests
TEST_F(TensorTest, MeanCalculation) {
  Tensor<float, NCHW> tensor(1, 1, 2, 2);
  tensor(0, 0, 0, 0) = 1.0f;
  tensor(0, 0, 0, 1) = 2.0f;
  tensor(0, 0, 1, 0) = 3.0f;
  tensor(0, 0, 1, 1) = 4.0f;

  float mean = tensor.mean();
  EXPECT_FLOAT_EQ(mean, 2.5f); // (1+2+3+4)/4 = 2.5
}

// Memory alignment tests
TEST_F(TensorTest, DataAlignment) {
  Tensor<float, NCHW> tensor(10, 10, 10, 10);

  // Check if data is properly aligned (default is 16 bytes)
  EXPECT_TRUE(tensor.is_aligned(16));
}

// Move semantics tests
TEST_F(TensorTest, MoveConstructor) {
  Tensor<float, NCHW> original(1, 1, 2, 2);
  original.fill(42.0f);

  Tensor<float, NCHW> moved(std::move(original));

  EXPECT_EQ(moved.size(), 4);
  EXPECT_FLOAT_EQ(moved(0, 0, 0, 0), 42.0f);
  EXPECT_EQ(original.data(), nullptr); // Original should be empty after move
}

// Multi-batch and multi-channel tests
TEST_F(TensorTest, MultiBatchAccess) {
  Tensor<float, NCHW> tensor(2, 1, 2, 2); // 2 batches

  // Set different values for each batch
  tensor(0, 0, 0, 0) = 1.0f; // batch 0
  tensor(1, 0, 0, 0) = 2.0f; // batch 1

  EXPECT_FLOAT_EQ(tensor(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(tensor(1, 0, 0, 0), 2.0f);
}

TEST_F(TensorTest, MultiChannelAccess) {
  Tensor<float, NCHW> tensor(1, 3, 2, 2); // 3 channels

  // Set different values for each channel
  tensor(0, 0, 0, 0) = 1.0f; // channel 0
  tensor(0, 1, 0, 0) = 2.0f; // channel 1
  tensor(0, 2, 0, 0) = 3.0f; // channel 2

  EXPECT_FLOAT_EQ(tensor(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(tensor(0, 1, 0, 0), 2.0f);
  EXPECT_FLOAT_EQ(tensor(0, 2, 0, 0), 3.0f);
}

// Parameterized tests for different tensor sizes
class TensorSizeTest : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t, size_t>> {
};

TEST_P(TensorSizeTest, ConstructorAndSize) {
  auto [batch, channels, height, width] = GetParam();
  Tensor<float, NCHW> tensor(batch, channels, height, width);

  EXPECT_EQ(tensor.batch_size(), batch);
  EXPECT_EQ(tensor.channels(), channels);
  EXPECT_EQ(tensor.height(), height);
  EXPECT_EQ(tensor.width(), width);
  EXPECT_EQ(tensor.size(), batch * channels * height * width);
}

INSTANTIATE_TEST_SUITE_P(DifferentShapes, TensorSizeTest,
                         ::testing::Values(std::make_tuple(1, 1, 1, 1),
                                           std::make_tuple(1, 3, 224, 224),
                                           std::make_tuple(32, 64, 56, 56),
                                           std::make_tuple(64, 128, 28, 28)));

// Floating point precision tests
TEST(TensorFloatingPointTest, FloatingPointComparisons) {
  Tensor<float, NCHW> tensor1(1, 1, 2, 2);
  Tensor<float, NCHW> tensor2(1, 1, 2, 2);

  // Use values that might have floating point precision issues
  tensor1.fill(0.1f + 0.2f); // This might not be exactly 0.3
  tensor2.fill(0.3f);

  // For ML applications, we often need to test with tolerance
  Tensor<float, NCHW> diff = tensor1 - tensor2;

  // Check that difference is small (within tolerance)
  for (size_t i = 0; i < diff.size(); ++i) {
    EXPECT_NEAR(diff.data()[i], 0.0f, 1e-6f);
  }
}