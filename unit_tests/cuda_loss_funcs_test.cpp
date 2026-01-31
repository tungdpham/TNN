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
#include "nn/loss_impl/cpu/loss_ops.hpp"
#include "nn/loss_impl/cuda/loss_ops.hpp"

using namespace tnn;

#ifdef USE_CUDA

class CUDALossOpsTest : public ::testing::Test {
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
      GTEST_SKIP() << "No GPU device available, skipping CUDA loss ops tests";
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

  bool has_gpu_;
};

TEST_F(CUDALossOpsTest, CrossEntropyLossBasic) {
  const size_t batch_size = 32;
  const size_t num_classes = 10;
  const float epsilon = 1e-7f;

  std::vector<float> predictions(batch_size * num_classes);
  std::vector<float> targets(batch_size * num_classes, 0.0f);

  for (size_t i = 0; i < batch_size; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      predictions[i * num_classes + j] = static_cast<float>((i * 7 + j * 3) % 100) / 100.0f + 0.01f;
      sum += predictions[i * num_classes + j];
    }

    for (size_t j = 0; j < num_classes; ++j) {
      predictions[i * num_classes + j] /= sum;
    }

    size_t target_class = i % num_classes;
    targets[i * num_classes + target_class] = 1.0f;
  }

  float cpu_loss, gpu_loss;

  auto loss_task =
      create_cpu_task("default", cpu::loss::compute_crossentropy_loss<float>, predictions.data(),
                      targets.data(), cpu_loss, batch_size, num_classes, epsilon);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  auto gpu_loss_task = create_cuda_task("default", cuda::loss::compute_crossentropy_loss<float>,
                                        gpu_predictions.get<float>(), gpu_targets.get<float>(),
                                        gpu_loss, batch_size, num_classes, epsilon);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-4f);
}

TEST_F(CUDALossOpsTest, CrossEntropyGradientBasic) {
  const size_t batch_size = 16;
  const size_t num_classes = 5;
  float epsilon = 1e-7f;

  std::vector<float> predictions(batch_size * num_classes);
  std::vector<float> targets(batch_size * num_classes, 0.0f);

  for (size_t i = 0; i < batch_size; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      predictions[i * num_classes + j] = static_cast<float>((i * 7 + j * 3) % 100) / 100.0f + 0.01f;
      sum += predictions[i * num_classes + j];
    }

    for (size_t j = 0; j < num_classes; ++j) {
      predictions[i * num_classes + j] /= sum;
    }

    size_t target_class = i % num_classes;
    targets[i * num_classes + target_class] = 1.0f;
  }

  std::vector<float> cpu_gradient(batch_size * num_classes);

  create_cpu_task("default", cpu::loss::compute_crossentropy_gradient<float>, predictions.data(),
                  targets.data(), cpu_gradient.data(), batch_size, num_classes, epsilon);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());
  dptr gpu_gradient = make_dptr_t<float>(getGPU(), batch_size * num_classes);

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_crossentropy_gradient<float>,
                   gpu_predictions.get<float>(), gpu_targets.get<float>(),
                   gpu_gradient.get<float>(), batch_size, num_classes, epsilon);

  std::vector<float> gpu_gradient_cpu(batch_size * num_classes);
  getGPU().copyToHost(gpu_gradient_cpu.data(), gpu_gradient.get<float>(),
                      (batch_size * num_classes) * sizeof(float));

  compareArrays(cpu_gradient, gpu_gradient_cpu);
}

TEST_F(CUDALossOpsTest, CrossEntropyLargeBatch) {
  const size_t batch_size = 2048;
  const size_t num_classes = 1000;
  const float epsilon = 1e-7f;

  std::vector<float> predictions(batch_size * num_classes);
  std::vector<float> targets(batch_size * num_classes, 0.0f);

  for (size_t i = 0; i < batch_size; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      predictions[i * num_classes + j] = static_cast<float>((i * 7 + j * 3) % 100) / 100.0f;
      sum += predictions[i * num_classes + j];
    }

    for (size_t j = 0; j < num_classes; ++j) {
      predictions[i * num_classes + j] /= sum;
    }

    size_t target_class = i % num_classes;
    targets[i * num_classes + target_class] = 1.0f;
  }

  float cpu_loss, gpu_loss;

  create_cpu_task("default", cpu::loss::compute_crossentropy_loss<float>, predictions.data(),
                  targets.data(), cpu_loss, batch_size, num_classes, epsilon);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_crossentropy_loss<float>,
                   gpu_predictions.get<float>(), gpu_targets.get<float>(), gpu_loss, batch_size,
                   num_classes, epsilon);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-3f);
}

TEST_F(CUDALossOpsTest, MSELossBasic) {
  const size_t batch_size = 32;
  const size_t output_size = 16;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 50) / 5.0f;
    targets[i] = static_cast<float>((i + 3) % 50) / 5.0f;
  }

  float cpu_loss, gpu_loss;

  create_cpu_task("default", cpu::loss::compute_mse_loss<float>, predictions.data(), targets.data(),
                  cpu_loss, batch_size, output_size);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_mse_loss<float>, gpu_predictions.get<float>(),
                   gpu_targets.get<float>(), gpu_loss, batch_size, output_size);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-2f);
}

TEST_F(CUDALossOpsTest, MSEGradientBasic) {
  const size_t batch_size = 32;
  const size_t output_size = 16;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 50) / 5.0f;
    targets[i] = static_cast<float>((i + 3) % 50) / 5.0f;
  }

  std::vector<float> cpu_gradient(batch_size * output_size);
  create_cpu_task("default", cpu::loss::compute_mse_gradient<float>, predictions.data(),
                  targets.data(), cpu_gradient.data(), batch_size, output_size);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());
  dptr gpu_gradient = make_dptr_t<float>(getGPU(), batch_size * output_size);

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_mse_gradient<float>, gpu_predictions.get<float>(),
                   gpu_targets.get<float>(), gpu_gradient.get<float>(), batch_size, output_size);

  std::vector<float> gpu_gradient_cpu(batch_size * output_size);
  getGPU().copyToHost(gpu_gradient_cpu.data(), gpu_gradient.get<float>(),
                      (batch_size * output_size) * sizeof(float));

  compareArrays(cpu_gradient, gpu_gradient_cpu);
}

TEST_F(CUDALossOpsTest, MSELargeBatch) {
  const size_t batch_size = 8192;
  const size_t output_size = 512;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 100) / 10.0f;
    targets[i] = static_cast<float>((i + 17) % 100) / 10.0f;
  }

  float cpu_loss, gpu_loss;

  create_cpu_task("default", cpu::loss::compute_mse_loss<float>, predictions.data(), targets.data(),
                  cpu_loss, batch_size, output_size);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_mse_loss<float>, gpu_predictions.get<float>(),
                   gpu_targets.get<float>(), gpu_loss, batch_size, output_size);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-4f);
}

TEST_F(CUDALossOpsTest, MAELossBasic) {
  const size_t batch_size = 128;
  const size_t output_size = 256;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 100) / 10.0f;
    targets[i] = static_cast<float>((i + 7) % 100) / 10.0f;
  }

  float cpu_loss, gpu_loss;

  create_cpu_task("default", cpu::loss::compute_mae_loss<float>, predictions.data(), targets.data(),
                  cpu_loss, batch_size, output_size);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_mae_loss<float>, gpu_predictions.get<float>(),
                   gpu_targets.get<float>(), gpu_loss, batch_size, output_size);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-2f);
}

TEST_F(CUDALossOpsTest, MAEGradientBasic) {
  const size_t batch_size = 128;
  const size_t output_size = 256;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 100) / 10.0f;
    targets[i] = static_cast<float>((i + 7) % 100) / 10.0f;
  }

  std::vector<float> cpu_gradient(batch_size * output_size);
  create_cpu_task("default", cpu::loss::compute_mae_gradient<float>, predictions.data(),
                  targets.data(), cpu_gradient.data(), batch_size, output_size);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());
  dptr gpu_gradient = make_dptr_t<float>(getGPU(), batch_size * output_size);

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_mae_gradient<float>, gpu_predictions.get<float>(),
                   gpu_targets.get<float>(), gpu_gradient.get<float>(), batch_size, output_size);

  std::vector<float> gpu_gradient_cpu(batch_size * output_size);
  getGPU().copyToHost(gpu_gradient_cpu.data(), gpu_gradient.get<float>(),
                      (batch_size * output_size) * sizeof(float));

  compareArrays(cpu_gradient, gpu_gradient_cpu);
}

TEST_F(CUDALossOpsTest, MAELargeBatch) {
  const size_t batch_size = 8192;
  const size_t output_size = 512;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 100) / 10.0f;
    targets[i] = static_cast<float>((i + 23) % 100) / 10.0f;
  }

  float cpu_loss, gpu_loss;

  create_cpu_task("default", cpu::loss::compute_mae_loss<float>, predictions.data(), targets.data(),
                  cpu_loss, batch_size, output_size);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_mae_loss<float>, gpu_predictions.get<float>(),
                   gpu_targets.get<float>(), gpu_loss, batch_size, output_size);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-2f);
}

TEST_F(CUDALossOpsTest, HuberLossBasic) {
  const size_t batch_size = 128;
  const size_t output_size = 256;
  const float delta = 1.0f;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 100) / 10.0f;
    targets[i] = static_cast<float>((i + 11) % 100) / 10.0f;
  }

  float cpu_loss, gpu_loss;

  create_cpu_task("default", cpu::loss::compute_huber_loss<float>, predictions.data(),
                  targets.data(), cpu_loss, batch_size, output_size, delta);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_huber_loss<float>, gpu_predictions.get<float>(),
                   gpu_targets.get<float>(), gpu_loss, batch_size, output_size, delta);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-2f);
}

TEST_F(CUDALossOpsTest, HuberGradientBasic) {
  const size_t batch_size = 128;
  const size_t output_size = 256;
  const float delta = 1.0f;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 100) / 10.0f;
    targets[i] = static_cast<float>((i + 11) % 100) / 10.0f;
  }

  std::vector<float> cpu_gradient(batch_size * output_size);
  create_cpu_task("default", cpu::loss::compute_huber_gradient<float>, predictions.data(),
                  targets.data(), cpu_gradient.data(), batch_size, output_size, delta);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());
  dptr gpu_gradient = make_dptr_t<float>(getGPU(), batch_size * output_size);

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_huber_gradient<float>,
                   gpu_predictions.get<float>(), gpu_targets.get<float>(),
                   gpu_gradient.get<float>(), batch_size, output_size, delta);

  std::vector<float> gpu_gradient_cpu(batch_size * output_size);
  getGPU().copyToHost(gpu_gradient_cpu.data(), gpu_gradient.get<float>(),
                      (batch_size * output_size) * sizeof(float));

  compareArrays(cpu_gradient, gpu_gradient_cpu);
}

TEST_F(CUDALossOpsTest, HuberLossVaryingDelta) {
  const size_t batch_size = 128;
  const size_t output_size = 256;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i) / 5.0f;
    targets[i] = static_cast<float>(i + 7) / 5.0f;
  }

  std::vector<float> deltas = {0.5f, 1.0f, 2.0f, 5.0f};

  for (float delta : deltas) {
    float cpu_loss, gpu_loss;
    create_cpu_task("default", cpu::loss::compute_huber_loss<float>, predictions.data(),
                    targets.data(), cpu_loss, batch_size, output_size, delta);

    dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
    dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());

    getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                          predictions.size() * sizeof(float));
    getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

    create_cuda_task("default", cuda::loss::compute_huber_loss<float>, gpu_predictions.get<float>(),
                     gpu_targets.get<float>(), gpu_loss, batch_size, output_size, delta);

    EXPECT_NEAR(cpu_loss, gpu_loss, 1e-2f) << "Mismatch for delta = " << delta;
  }
}

TEST_F(CUDALossOpsTest, HuberLargeBatch) {
  const size_t batch_size = 8192;
  const size_t output_size = 512;
  const float delta = 1.0f;

  std::vector<float> predictions(batch_size * output_size);
  std::vector<float> targets(batch_size * output_size);

  for (size_t i = 0; i < predictions.size(); ++i) {
    predictions[i] = static_cast<float>(i % 100) / 10.0f;
    targets[i] = static_cast<float>((i + 31) % 100) / 10.0f;
  }

  float cpu_loss, gpu_loss;

  create_cpu_task("default", cpu::loss::compute_huber_loss<float>, predictions.data(),
                  targets.data(), cpu_loss, batch_size, output_size, delta);

  dptr gpu_predictions = make_dptr_t<float>(getGPU(), predictions.size());
  dptr gpu_targets = make_dptr_t<float>(getGPU(), targets.size());

  getGPU().copyToDevice(gpu_predictions.get<float>(), predictions.data(),
                        predictions.size() * sizeof(float));
  getGPU().copyToDevice(gpu_targets.get<float>(), targets.data(), targets.size() * sizeof(float));

  create_cuda_task("default", cuda::loss::compute_huber_loss<float>, gpu_predictions.get<float>(),
                   gpu_targets.get<float>(), gpu_loss, batch_size, output_size, delta);

  EXPECT_NEAR(cpu_loss, gpu_loss, 1e-2f);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#endif
