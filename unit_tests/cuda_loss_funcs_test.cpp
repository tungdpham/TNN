/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "device/device_manager.hpp"
#include "device/device_ptr.hpp"
#include "nn/loss_impl/cpu/loss_ops.hpp"
#include "nn/loss_impl/cuda/loss_ops.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

using namespace tnn;

#ifdef USE_CUDA
// Test fixture for CUDA loss operations
class CUDALossOpsTest : public ::testing::Test {
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
      GTEST_SKIP() << "No GPU device available, skipping CUDA loss ops tests";
    }
  }

  void TearDown() override {}

  static void TearDownTestSuite() {}

  // Helper function to compare arrays with tolerance
  void compareArrays(const std::vector<float> &expected, const std::vector<float> &actual,
                     float tolerance = 1e-4f) {
    ASSERT_EQ(expected.size(), actual.size())
        << "Array sizes don't match: expected " << expected.size() << ", got " << actual.size();

    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance)
          << "Mismatch at index " << i << ": expected " << expected[i] << ", got " << actual[i];
    }
  }

  bool has_gpu_;
  const Device *gpu_device_;
};

// ==================== CrossEntropy Loss Tests ====================

TEST_F(CUDALossOpsTest, CrossEntropyLossBasic) {
  const size_t batch_size = 4;
  const size_t num_classes = 3;
  const float epsilon = 1e-7f;

  // One-hot encoded targets
  std::vector<float> predictions = {
      0.7f, 0.2f, 0.1f, // Class 0
      0.1f, 0.8f, 0.1f, // Class 1
      0.2f, 0.3f, 0.5f, // Class 2
      0.6f, 0.3f, 0.1f  // Class 0
  };

  std::vector<float> targets = {
      1.0f, 0.0f, 0.0f, // Class 0
      0.0f, 1.0f, 0.0f, // Class 1
      0.0f, 0.0f, 1.0f, // Class 2
      1.0f, 0.0f, 0.0f  // Class 0
  };

  // CPU version
  float cpu_loss = cpu::loss::compute_crossentropy_loss(predictions.data(), targets.data(),
                                                        batch_size, num_classes, epsilon);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  float gpu_loss = cuda::loss::compute_crossentropy_loss(gpu_predictions.get(), gpu_targets.get(),
                                                         batch_size, num_classes, epsilon);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-4f);
}

TEST_F(CUDALossOpsTest, CrossEntropyGradientBasic) {
  const size_t batch_size = 4;
  const size_t num_classes = 3;

  std::vector<float> predictions = {
      0.7f, 0.2f, 0.1f, // Class 0
      0.1f, 0.8f, 0.1f, // Class 1
      0.2f, 0.3f, 0.5f, // Class 2
      0.6f, 0.3f, 0.1f  // Class 0
  };

  std::vector<float> targets = {
      1.0f, 0.0f, 0.0f, // Class 0
      0.0f, 1.0f, 0.0f, // Class 1
      0.0f, 0.0f, 1.0f, // Class 2
      1.0f, 0.0f, 0.0f  // Class 0
  };

  // CPU version
  std::vector<float> cpu_gradient(batch_size * num_classes);
  cpu::loss::compute_crossentropy_gradient(predictions.data(), targets.data(), cpu_gradient.data(),
                                           batch_size, num_classes);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, batch_size * num_classes);

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  cuda::loss::compute_crossentropy_gradient(gpu_predictions.get(), gpu_targets.get(),
                                            gpu_gradient.get(), batch_size, num_classes);

  std::vector<float> gpu_gradient_cpu(batch_size * num_classes);
  gpu_device_->copyToHost(gpu_gradient_cpu.data(), gpu_gradient.get(),
                          (batch_size * num_classes) * sizeof(float));

  compareArrays(cpu_gradient, gpu_gradient_cpu);
}

TEST_F(CUDALossOpsTest, CrossEntropyLargeBatch) {
  const size_t batch_size = 64;
  const size_t num_classes = 10;
  const float epsilon = 1e-7f;

  std::vector<float> predictions(batch_size * num_classes);
  std::vector<float> targets(batch_size * num_classes, 0.0f);

  // Initialize with random-like predictions
  for (size_t i = 0; i < batch_size; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      predictions[i * num_classes + j] = static_cast<float>((i * 7 + j * 3) % 100) / 100.0f;
      sum += predictions[i * num_classes + j];
    }
    // Normalize to sum to 1
    for (size_t j = 0; j < num_classes; ++j) {
      predictions[i * num_classes + j] /= sum;
    }
    // Set one-hot target
    size_t target_class = i % num_classes;
    targets[i * num_classes + target_class] = 1.0f;
  }

  // CPU version
  float cpu_loss = cpu::loss::compute_crossentropy_loss(predictions.data(), targets.data(),
                                                        batch_size, num_classes, epsilon);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  float gpu_loss = cuda::loss::compute_crossentropy_loss(gpu_predictions.get(), gpu_targets.get(),
                                                         batch_size, num_classes, epsilon);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-3f);
}

// ==================== Softmax CrossEntropy Loss Tests ====================

TEST_F(CUDALossOpsTest, SoftmaxCrossEntropyLossBasic) {
  const size_t batch_size = 4;
  const size_t num_classes = 3;

  // Raw logits (not softmax probabilities)
  std::vector<float> logits = {
      2.0f, 1.0f, 0.1f, // Class 0 should be predicted
      0.5f, 3.0f, 0.2f, // Class 1 should be predicted
      1.0f, 0.5f, 2.5f, // Class 2 should be predicted
      3.0f, 1.5f, 0.5f  // Class 0 should be predicted
  };

  std::vector<float> targets = {
      1.0f, 0.0f, 0.0f, // Class 0
      0.0f, 1.0f, 0.0f, // Class 1
      0.0f, 0.0f, 1.0f, // Class 2
      1.0f, 0.0f, 0.0f  // Class 0
  };

  // CPU version
  float cpu_loss = cpu::loss::compute_softmax_crossentropy_loss(logits.data(), targets.data(),
                                                                batch_size, num_classes);

  // GPU version
  device_ptr<float[]> gpu_logits = make_array_ptr<float[]>(gpu_device_, logits.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

  gpu_device_->copyToDevice(gpu_logits.get(), logits.data(), logits.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  float gpu_loss = cuda::loss::compute_softmax_crossentropy_loss(
      gpu_logits.get(), gpu_targets.get(), batch_size, num_classes);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-4f);
}

TEST_F(CUDALossOpsTest, SoftmaxCrossEntropyGradientBasic) {
  const size_t batch_size = 4;
  const size_t num_classes = 3;

  std::vector<float> logits = {
      2.0f, 1.0f, 0.1f, // Class 0
      0.5f, 3.0f, 0.2f, // Class 1
      1.0f, 0.5f, 2.5f, // Class 2
      3.0f, 1.5f, 0.5f  // Class 0
  };

  std::vector<float> targets = {
      1.0f, 0.0f, 0.0f, // Class 0
      0.0f, 1.0f, 0.0f, // Class 1
      0.0f, 0.0f, 1.0f, // Class 2
      1.0f, 0.0f, 0.0f  // Class 0
  };

  // CPU version
  std::vector<float> cpu_gradient(batch_size * num_classes);
  cpu::loss::compute_softmax_crossentropy_gradient(logits.data(), targets.data(),
                                                   cpu_gradient.data(), batch_size, num_classes);

  // GPU version
  device_ptr<float[]> gpu_logits = make_array_ptr<float[]>(gpu_device_, logits.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, batch_size * num_classes);

  gpu_device_->copyToDevice(gpu_logits.get(), logits.data(), logits.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  cuda::loss::compute_softmax_crossentropy_gradient(gpu_logits.get(), gpu_targets.get(),
                                                    gpu_gradient.get(), batch_size, num_classes);

  std::vector<float> gpu_gradient_cpu(batch_size * num_classes);
  gpu_device_->copyToHost(gpu_gradient_cpu.data(), gpu_gradient.get(),
                          (batch_size * num_classes) * sizeof(float));

  compareArrays(cpu_gradient, gpu_gradient_cpu);
}

TEST_F(CUDALossOpsTest, SoftmaxCrossEntropyLargeBatch) {
  const size_t batch_size = 128;
  const size_t num_classes = 100;

  std::vector<float> logits(batch_size * num_classes);
  std::vector<float> targets(batch_size * num_classes, 0.0f);

  // Initialize with random-like logits
  for (size_t i = 0; i < batch_size * num_classes; ++i) {
    logits[i] = static_cast<float>((i * 13) % 200 - 100) / 50.0f;
  }

  // Set one-hot targets
  for (size_t i = 0; i < batch_size; ++i) {
    size_t target_class = i % num_classes;
    targets[i * num_classes + target_class] = 1.0f;
  }

  // CPU version
  float cpu_loss = cpu::loss::compute_softmax_crossentropy_loss(logits.data(), targets.data(),
                                                                batch_size, num_classes);

  // GPU version
  device_ptr<float[]> gpu_logits = make_array_ptr<float[]>(gpu_device_, logits.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

  gpu_device_->copyToDevice(gpu_logits.get(), logits.data(), logits.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  float gpu_loss = cuda::loss::compute_softmax_crossentropy_loss(
      gpu_logits.get(), gpu_targets.get(), batch_size, num_classes);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-3f);
}

// ==================== MSE Loss Tests ====================

TEST_F(CUDALossOpsTest, MSELossBasic) {
  const size_t batch_size = 4;
  const size_t output_size = 3;

  std::vector<float> predictions = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  std::vector<float> targets = {1.5f, 2.5f, 2.5f, 3.5f, 5.5f,  6.5f,
                                7.5f, 7.5f, 9.5f, 9.5f, 11.5f, 11.5f};

  // CPU version
  float cpu_loss =
      cpu::loss::compute_mse_loss(predictions.data(), targets.data(), batch_size, output_size);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  float gpu_loss = cuda::loss::compute_mse_loss(gpu_predictions.get(), gpu_targets.get(),
                                                batch_size, output_size);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-5f);
}

TEST_F(CUDALossOpsTest, MSEGradientBasic) {
  const size_t batch_size = 4;
  const size_t output_size = 3;

  std::vector<float> predictions = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  std::vector<float> targets = {1.5f, 2.5f, 2.5f, 3.5f, 5.5f,  6.5f,
                                7.5f, 7.5f, 9.5f, 9.5f, 11.5f, 11.5f};

  // CPU version
  std::vector<float> cpu_gradient(batch_size * output_size);
  cpu::loss::compute_mse_gradient(predictions.data(), targets.data(), cpu_gradient.data(),
                                  batch_size, output_size);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, batch_size * output_size);

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  cuda::loss::compute_mse_gradient(gpu_predictions.get(), gpu_targets.get(), gpu_gradient.get(),
                                   batch_size, output_size);

  std::vector<float> gpu_gradient_cpu(batch_size * output_size);
  gpu_device_->copyToHost(gpu_gradient_cpu.data(), gpu_gradient.get(),
                          (batch_size * output_size) * sizeof(float));

  compareArrays(cpu_gradient, gpu_gradient_cpu);
}

TEST_F(CUDALossOpsTest, MSELargeBatch) {
  const size_t batch_size = 256;
  const size_t output_size = 64;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 100) / 10.0f;
    targets[i] = static_cast<float>((i + 17) % 100) / 10.0f;
  }

  // CPU version
  float cpu_loss =
      cpu::loss::compute_mse_loss(predictions.data(), targets.data(), batch_size, output_size);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  float gpu_loss = cuda::loss::compute_mse_loss(gpu_predictions.get(), gpu_targets.get(),
                                                batch_size, output_size);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-5f);
}

// ==================== MAE Loss Tests ====================

TEST_F(CUDALossOpsTest, MAELossBasic) {
  const size_t batch_size = 4;
  const size_t output_size = 3;

  std::vector<float> predictions = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  std::vector<float> targets = {1.5f, 2.5f, 2.5f, 3.5f, 5.5f,  6.5f,
                                7.5f, 7.5f, 9.5f, 9.5f, 11.5f, 11.5f};

  // CPU version
  float cpu_loss =
      cpu::loss::compute_mae_loss(predictions.data(), targets.data(), batch_size, output_size);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  float gpu_loss = cuda::loss::compute_mae_loss(gpu_predictions.get(), gpu_targets.get(),
                                                batch_size, output_size);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-5f);
}

TEST_F(CUDALossOpsTest, MAEGradientBasic) {
  const size_t batch_size = 4;
  const size_t output_size = 3;

  std::vector<float> predictions = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  std::vector<float> targets = {1.5f, 2.5f, 2.5f, 3.5f, 5.5f,  6.5f,
                                7.5f, 7.5f, 9.5f, 9.5f, 11.5f, 11.5f};

  // CPU version
  std::vector<float> cpu_gradient(batch_size * output_size);
  cpu::loss::compute_mae_gradient(predictions.data(), targets.data(), cpu_gradient.data(),
                                  batch_size, output_size);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, batch_size * output_size);

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  cuda::loss::compute_mae_gradient(gpu_predictions.get(), gpu_targets.get(), gpu_gradient.get(),
                                   batch_size, output_size);

  std::vector<float> gpu_gradient_cpu(batch_size * output_size);
  gpu_device_->copyToHost(gpu_gradient_cpu.data(), gpu_gradient.get(),
                          (batch_size * output_size) * sizeof(float));

  compareArrays(cpu_gradient, gpu_gradient_cpu);
}

TEST_F(CUDALossOpsTest, MAELargeBatch) {
  const size_t batch_size = 256;
  const size_t output_size = 64;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 100) / 10.0f;
    targets[i] = static_cast<float>((i + 23) % 100) / 10.0f;
  }

  // CPU version
  float cpu_loss =
      cpu::loss::compute_mae_loss(predictions.data(), targets.data(), batch_size, output_size);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  float gpu_loss = cuda::loss::compute_mae_loss(gpu_predictions.get(), gpu_targets.get(),
                                                batch_size, output_size);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-5f);
}

// ==================== Huber Loss Tests ====================

TEST_F(CUDALossOpsTest, HuberLossBasic) {
  const size_t batch_size = 4;
  const size_t output_size = 3;
  const float delta = 1.0f;

  std::vector<float> predictions = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  std::vector<float> targets = {1.5f, 2.5f, 2.5f, 3.5f, 5.5f,  6.5f,
                                7.5f, 7.5f, 9.5f, 9.5f, 11.5f, 11.5f};

  // CPU version
  float cpu_loss = cpu::loss::compute_huber_loss(predictions.data(), targets.data(), batch_size,
                                                 output_size, delta);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  float gpu_loss = cuda::loss::compute_huber_loss(gpu_predictions.get(), gpu_targets.get(),
                                                  batch_size, output_size, delta);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-5f);
}

TEST_F(CUDALossOpsTest, HuberGradientBasic) {
  const size_t batch_size = 4;
  const size_t output_size = 3;
  const float delta = 1.0f;

  std::vector<float> predictions = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  std::vector<float> targets = {1.5f, 2.5f, 2.5f, 3.5f, 5.5f,  6.5f,
                                7.5f, 7.5f, 9.5f, 9.5f, 11.5f, 11.5f};

  // CPU version
  std::vector<float> cpu_gradient(batch_size * output_size);
  cpu::loss::compute_huber_gradient(predictions.data(), targets.data(), cpu_gradient.data(),
                                    batch_size, output_size, delta);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());
  device_ptr<float[]> gpu_gradient = make_array_ptr<float[]>(gpu_device_, batch_size * output_size);

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  cuda::loss::compute_huber_gradient(gpu_predictions.get(), gpu_targets.get(), gpu_gradient.get(),
                                     batch_size, output_size, delta);

  std::vector<float> gpu_gradient_cpu(batch_size * output_size);
  gpu_device_->copyToHost(gpu_gradient_cpu.data(), gpu_gradient.get(),
                          (batch_size * output_size) * sizeof(float));

  compareArrays(cpu_gradient, gpu_gradient_cpu);
}

TEST_F(CUDALossOpsTest, HuberLossVaryingDelta) {
  const size_t batch_size = 8;
  const size_t output_size = 4;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i) / 5.0f;
    targets[i] = static_cast<float>(i + 7) / 5.0f;
  }

  // Test with different delta values
  std::vector<float> deltas = {0.5f, 1.0f, 2.0f, 5.0f};

  for (float delta : deltas) {
    // CPU version
    float cpu_loss = cpu::loss::compute_huber_loss(predictions.data(), targets.data(), batch_size,
                                                   output_size, delta);

    // GPU version
    device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
    device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

    gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                              predictions.size() * sizeof(float));
    gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

    float gpu_loss = cuda::loss::compute_huber_loss(gpu_predictions.get(), gpu_targets.get(),
                                                    batch_size, output_size, delta);

    EXPECT_NEAR(cpu_loss, gpu_loss, 1e-5f) << "Mismatch for delta = " << delta;
  }
}

TEST_F(CUDALossOpsTest, HuberLargeBatch) {
  const size_t batch_size = 256;
  const size_t output_size = 64;
  const float delta = 1.0f;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 100) / 10.0f;
    targets[i] = static_cast<float>((i + 31) % 100) / 10.0f;
  }

  // CPU version
  float cpu_loss = cpu::loss::compute_huber_loss(predictions.data(), targets.data(), batch_size,
                                                 output_size, delta);

  // GPU version
  device_ptr<float[]> gpu_predictions = make_array_ptr<float[]>(gpu_device_, predictions.size());
  device_ptr<float[]> gpu_targets = make_array_ptr<float[]>(gpu_device_, targets.size());

  gpu_device_->copyToDevice(gpu_predictions.get(), predictions.data(),
                            predictions.size() * sizeof(float));
  gpu_device_->copyToDevice(gpu_targets.get(), targets.data(), targets.size() * sizeof(float));

  float gpu_loss = cuda::loss::compute_huber_loss(gpu_predictions.get(), gpu_targets.get(),
                                                  batch_size, output_size, delta);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-5f);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif // USE_CUDA
