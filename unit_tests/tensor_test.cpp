#include "tensor/tensor.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace tnn;

class TensorTest : public ::testing::Test {
protected:
  void SetUp() override {
    small_tensor = make_tensor<float>({1, 1, 2, 2});
    small_tensor->fill(1.0f);

    large_tensor = make_tensor<float>({2, 3, 4, 4});
    large_tensor->fill(2.0f);
  }

  Tensor small_tensor;
  Tensor large_tensor;
};

TEST_F(TensorTest, Constructor4D) {
  Tensor tensor = make_tensor<float>({2, 3, 4, 4});

  EXPECT_EQ(tensor->shape()[0], 2);
  EXPECT_EQ(tensor->shape()[1], 3);
  EXPECT_EQ(tensor->shape()[2], 4);
  EXPECT_EQ(tensor->shape()[3], 4);
  EXPECT_EQ(tensor->size(), 96);
}

TEST_F(TensorTest, ConstructorWithShape) {
  std::vector<size_t> shape = {2, 3, 4, 4};
  Tensor tensor = make_tensor<float>(shape);

  EXPECT_EQ(tensor->shape(), shape);
  EXPECT_EQ(tensor->size(), 96);
  EXPECT_EQ(tensor->shape()[0], 2);
  EXPECT_EQ(tensor->shape()[1], 3);
  EXPECT_EQ(tensor->shape()[2], 4);
  EXPECT_EQ(tensor->shape()[3], 4);
}

TEST_F(TensorTest, ElementAccess) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2});

  tensor->at<float>({0, 0, 0, 0}) = 1.0f;
  tensor->at<float>({0, 0, 0, 1}) = 2.0f;
  tensor->at<float>({0, 0, 1, 0}) = 3.0f;
  tensor->at<float>({0, 0, 1, 1}) = 4.0f;

  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 0, 1}), 2.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 1, 1}), 4.0f);
}

TEST_F(TensorTest, TensorAddition) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2});
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2});

  tensor1->fill(2.0f);
  tensor2->fill(3.0f);

  Tensor result = tensor1 + tensor2;

  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 1}), 5.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 0}), 5.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 1}), 5.0f);
}

TEST_F(TensorTest, TensorSubtraction) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2});
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2});

  tensor1->fill(5.0f);
  tensor2->fill(2.0f);

  Tensor result = tensor1 - tensor2;

  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 0}), 3.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 1}), 3.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 0}), 3.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 1}), 3.0f);
}

TEST_F(TensorTest, TensorMultiplication) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2});
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2});

  tensor1->fill(3.0f);
  tensor2->fill(4.0f);

  Tensor result = tensor1 * tensor2;

  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 0}), 12.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 1}), 12.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 0}), 12.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 1}), 12.0f);
}

TEST_F(TensorTest, ScalarMultiplication) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2});
  tensor->fill(3.0f);

  Tensor result = tensor * 2.0f;

  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 0}), 6.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 0}), 6.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 1}), 6.0f);
}

TEST_F(TensorTest, ScalarDivision) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2});
  tensor->fill(8.0f);

  Tensor result = tensor / 2.0f;

  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 0}), 4.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 0, 1}), 4.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 0}), 4.0f);
  EXPECT_FLOAT_EQ(result->at<float>({0, 0, 1, 1}), 4.0f);
}

TEST_F(TensorTest, InPlaceAddition) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2});
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2});

  tensor1->fill(2.0f);
  tensor2->fill(3.0f);

  tensor1->add(tensor2);

  EXPECT_FLOAT_EQ(tensor1->at<float>({0, 0, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(tensor1->at<float>({0, 0, 0, 1}), 5.0f);
  EXPECT_FLOAT_EQ(tensor1->at<float>({0, 0, 1, 0}), 5.0f);
  EXPECT_FLOAT_EQ(tensor1->at<float>({0, 0, 1, 1}), 5.0f);
}

TEST_F(TensorTest, InPlaceScalarMultiplication) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2});
  tensor->fill(3.0f);

  tensor->mul_scalar(2.0f);

  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 0, 0}), 6.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 0, 1}), 6.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 1, 0}), 6.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 1, 1}), 6.0f);
}

TEST_F(TensorTest, SameShapeComparison) {
  Tensor tensor1 = make_tensor<float>({2, 3, 4, 5});
  Tensor tensor2 = make_tensor<float>({2, 3, 4, 5});
  Tensor tensor3 = make_tensor<float>({2, 3, 4, 6});

  EXPECT_TRUE(tensor1->shape() == tensor2->shape());
  EXPECT_FALSE(tensor1->shape() == tensor3->shape());
}

TEST_F(TensorTest, AdditionShapeMismatch) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2});
  Tensor tensor2 = make_tensor<float>({1, 1, 3, 3});

  EXPECT_THROW(tensor1 + tensor2, std::invalid_argument);
}

TEST_F(TensorTest, DivisionByZero) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2});

  EXPECT_THROW(tensor / 0.0f, std::invalid_argument);
}

TEST_F(TensorTest, FillOperation) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2});
  tensor->fill(42.0f);

  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 0, 0}), 42.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 0, 1}), 42.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 1, 0}), 42.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 1, 1}), 42.0f);
}

TEST_F(TensorTest, CloneOperation) {
  Tensor original = make_tensor<float>({1, 1, 2, 2});
  original->fill(5.0f);

  Tensor cloned = original->clone();

  EXPECT_TRUE(original->shape() == cloned->shape());
  EXPECT_FLOAT_EQ(cloned->at<float>({0, 0, 0, 0}), 5.0f);
  EXPECT_FLOAT_EQ(cloned->at<float>({0, 0, 0, 1}), 5.0f);
  EXPECT_FLOAT_EQ(cloned->at<float>({0, 0, 1, 0}), 5.0f);
  EXPECT_FLOAT_EQ(cloned->at<float>({0, 0, 1, 1}), 5.0f);
}

TEST_F(TensorTest, MeanCalculation) {
  Tensor tensor = make_tensor<float>({1, 1, 2, 2});
  tensor->at<float>({0, 0, 0, 0}) = 1.0f;
  tensor->at<float>({0, 0, 0, 1}) = 2.0f;
  tensor->at<float>({0, 0, 1, 0}) = 3.0f;
  tensor->at<float>({0, 0, 1, 1}) = 4.0f;

  float mean = tensor->mean();
  EXPECT_FLOAT_EQ(mean, 2.5f);
}

TEST_F(TensorTest, DataAlignment) {
  Tensor tensor = make_tensor<float>({10, 10, 10, 10});

  EXPECT_TRUE(tensor->is_aligned(16));
}

TEST_F(TensorTest, MoveConstructor) {
  Tensor original = make_tensor<float>({1, 1, 2, 2});
  original->fill(42.0f);

  Tensor moved(std::move(original));

  EXPECT_EQ(moved->size(), 4);
  EXPECT_FLOAT_EQ(moved->at<float>({0, 0, 0, 0}), 42.0f);
  EXPECT_TRUE(original == nullptr);
}

TEST_F(TensorTest, MultiBatchAccess) {
  Tensor tensor = make_tensor<float>({2, 1, 2, 2});

  tensor->at<float>({0, 0, 0, 0}) = 1.0f;
  tensor->at<float>({1, 0, 0, 0}) = 2.0f;

  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({1, 0, 0, 0}), 2.0f);
}

TEST_F(TensorTest, MultiChannelAccess) {
  Tensor tensor = make_tensor<float>({1, 3, 2, 2});

  tensor->at<float>({0, 0, 0, 0}) = 1.0f;
  tensor->at<float>({0, 1, 0, 0}) = 2.0f;
  tensor->at<float>({0, 2, 0, 0}) = 3.0f;

  EXPECT_FLOAT_EQ(tensor->at<float>({0, 0, 0, 0}), 1.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 1, 0, 0}), 2.0f);
  EXPECT_FLOAT_EQ(tensor->at<float>({0, 2, 0, 0}), 3.0f);
}

class TensorSizeTest : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t, size_t>> {
};

TEST_P(TensorSizeTest, ConstructorAndSize) {
  auto [batch, channels, height, width] = GetParam();
  Tensor tensor = make_tensor<float>({batch, channels, height, width});

  EXPECT_EQ(tensor->shape()[0], batch);
  EXPECT_EQ(tensor->shape()[1], channels);
  EXPECT_EQ(tensor->shape()[2], height);
  EXPECT_EQ(tensor->shape()[3], width);
  EXPECT_EQ(tensor->size(), batch * channels * height * width);
}

INSTANTIATE_TEST_SUITE_P(DifferentShapes, TensorSizeTest,
                         ::testing::Values(std::make_tuple(1, 1, 1, 1),
                                           std::make_tuple(1, 3, 224, 224),
                                           std::make_tuple(32, 64, 56, 56),
                                           std::make_tuple(64, 128, 28, 28)));

TEST(TensorFloatingPointTest, FloatingPointComparisons) {
  Tensor tensor1 = make_tensor<float>({1, 1, 2, 2});
  Tensor tensor2 = make_tensor<float>({1, 1, 2, 2});

  tensor1->fill(0.1f + 0.2f);
  tensor2->fill(0.3f);

  Tensor diff = tensor1 - tensor2;

  for (size_t i = 0; i < diff->size(); ++i) {
    EXPECT_NEAR(diff->data_as<float>()[i], 0.0f, 1e-6f);
  }
}