/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "../tensor/tensor.hpp"
#include "device/task.hpp"
#include "loss_impl/cpu/loss_ops.hpp"
#ifdef USE_CUDA
#include "loss_impl/cuda/loss_ops.hpp"
#endif
#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace tnn {

struct LossConfig {
  std::string type;
  std::string name;
  std::unordered_map<std::string, std::any> parameters;

  template <typename T> T get(const std::string &key, const T &default_value = T{}) const {
    auto it = parameters.find(key);
    if (it != parameters.end()) {
      try {
        return std::any_cast<T>(it->second);
      } catch (const std::bad_any_cast &) {
        return default_value;
      }
    }
    return default_value;
  }
};

template <typename T = float> class Loss {
public:
  virtual ~Loss() = default;

  virtual std::unique_ptr<Task> compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets,
                                             T &loss) = 0;
  virtual std::unique_ptr<Task> compute_gradient(const Tensor<T> &predictions,
                                                 const Tensor<T> &targets, Tensor<T> &gradient) = 0;

  virtual std::string name() const = 0;
  virtual LossConfig get_config() const = 0;
  virtual std::unique_ptr<Loss<T>> clone() const = 0;

  virtual size_t num_parameters() const { return 0; }

  virtual void reset() {}
};

template <typename T = float> class CrossEntropyLoss : public Loss<T> {
public:
  explicit CrossEntropyLoss(T epsilon = static_cast<T>(1e-15)) : epsilon_(epsilon) {}

  std::unique_ptr<Task> compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets,
                                     T &loss) override {
    const size_t batch_size = predictions.shape()[0];
    const size_t num_classes = predictions.shape()[1];

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task("default", cpu::loss::compute_crossentropy_loss<T>, predictions.data(),
                             targets.data(), loss, batch_size, num_classes, epsilon_);
    }
#ifdef USE_CUDA
    else if (predictions.device_type() == DeviceType::GPU) {
      return create_gpu_task("default", cuda::loss::compute_crossentropy_loss<T>,
                             predictions.data(), targets.data(), loss, batch_size, num_classes,
                             epsilon_);
    }
#endif
    throw std::runtime_error("Unsupported device type for CrossEntropyLoss.");
  }

  std::unique_ptr<Task> compute_gradient(const Tensor<T> &predictions, const Tensor<T> &targets,
                                         Tensor<T> &gradient) override {
    gradient.resize(predictions.shape());
    const size_t batch_size = predictions.shape()[0];
    const size_t num_classes = predictions.shape()[1];

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task("default", cpu::loss::compute_crossentropy_gradient<T>,
                             predictions.data(), targets.data(), gradient.data(), batch_size,
                             num_classes);
    }
#ifdef USE_CUDA
    else if (predictions.device_type() == DeviceType::GPU) {
      return create_gpu_task("default", cuda::loss::compute_crossentropy_gradient<T>,
                             predictions.data(), targets.data(), gradient.data(), batch_size,
                             num_classes);
    }
#endif
    throw std::runtime_error("Unsupported device type for CrossEntropyLoss.");
  }

  std::string name() const override { return "CrossEntropyLoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "crossentropy";
    config.name = "CrossEntropyLoss";
    config.parameters["epsilon"] = epsilon_;
    return config;
  }

  std::unique_ptr<Loss<T>> clone() const override {
    return std::make_unique<CrossEntropyLoss<T>>(epsilon_);
  }

private:
  T epsilon_;
};

// Numerically stable Softmax + CrossEntropy combined loss
template <typename T = float> class SoftmaxCrossEntropyLoss : public Loss<T> {
public:
  SoftmaxCrossEntropyLoss() = default;

  std::unique_ptr<Task> compute_loss(const Tensor<T> &logits, const Tensor<T> &targets,
                                     T &loss) override {
    const size_t batch_size = logits.shape()[0];
    const size_t num_classes = logits.shape()[1];

    if (logits.device_type() == DeviceType::CPU) {
      return create_cpu_task("default", cpu::loss::compute_softmax_crossentropy_loss<T>,
                             logits.data(), targets.data(), loss, batch_size, num_classes);
    }
#ifdef USE_CUDA
    else if (logits.device_type() == DeviceType::GPU) {
      return create_gpu_task("default", cuda::loss::compute_softmax_crossentropy_loss<T>,
                             logits.data(), targets.data(), loss, batch_size, num_classes);
    }
#endif
    throw std::runtime_error("Unsupported device type for SoftmaxCrossEntropyLoss.");
  }

  std::unique_ptr<Task> compute_gradient(const Tensor<T> &logits, const Tensor<T> &targets,
                                         Tensor<T> &gradient) override {
    gradient.resize(logits.shape());
    const size_t batch_size = logits.shape()[0];
    const size_t num_classes = logits.shape()[1];

    if (logits.device_type() == DeviceType::CPU) {
      return create_cpu_task("default", cpu::loss::compute_softmax_crossentropy_gradient<T>,
                             logits.data(), targets.data(), gradient.data(), batch_size,
                             num_classes);
    }
#ifdef USE_CUDA
    else if (logits.device_type() == DeviceType::GPU) {
      return create_gpu_task("default", cuda::loss::compute_softmax_crossentropy_gradient<T>,
                             logits.data(), targets.data(), gradient.data(), batch_size,
                             num_classes);
    }
#endif
    throw std::runtime_error("Unsupported device type for SoftmaxCrossEntropyLoss.");
  }

  std::string name() const override { return "SoftmaxCrossEntropyLoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "softmax_crossentropy";
    config.name = "SoftmaxCrossEntropyLoss";
    return config;
  }

  std::unique_ptr<Loss<T>> clone() const override {
    return std::make_unique<SoftmaxCrossEntropyLoss<T>>();
  }
};

template <typename T = float> class MSELoss : public Loss<T> {
public:
  MSELoss() = default;

  std::unique_ptr<Task> compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets,
                                     T &loss) override {
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task("default", cpu::loss::compute_mse_loss<T>, predictions.data(),
                             targets.data(), loss, batch_size, output_size);
    }
#ifdef USE_CUDA
    else if (predictions.device_type() == DeviceType::GPU) {
      return create_gpu_task("default", cuda::loss::compute_mse_loss<T>, predictions.data(),
                             targets.data(), loss, batch_size, output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MSELoss.");
  }

  std::unique_ptr<Task> compute_gradient(const Tensor<T> &predictions, const Tensor<T> &targets,
                                         Tensor<T> &gradient) override {
    gradient.resize(predictions.shape());
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task("default", cpu::loss::compute_mse_gradient<T>, predictions.data(),
                             targets.data(), gradient.data(), batch_size, output_size);
    }
#ifdef USE_CUDA
    else if (predictions.device_type() == DeviceType::GPU) {
      return create_gpu_task("default", cuda::loss::compute_mse_gradient<T>, predictions.data(),
                             targets.data(), gradient.data(), batch_size, output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MSELoss.");
  }

  std::string name() const override { return "MSELoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "mse";
    config.name = "MSELoss";
    return config;
  }

  std::unique_ptr<Loss<T>> clone() const override { return std::make_unique<MSELoss<T>>(); }
};

template <typename T = float> class MAELoss : public Loss<T> {
public:
  MAELoss() = default;

  std::unique_ptr<Task> compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets,
                                     T &loss) override {
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task("default", cpu::loss::compute_mae_loss<T>, predictions.data(),
                             targets.data(), loss, batch_size, output_size);
    }
#ifdef USE_CUDA
    else if (predictions.device_type() == DeviceType::GPU) {
      return create_gpu_task("default", cuda::loss::compute_mae_loss<T>, predictions.data(),
                             targets.data(), loss, batch_size, output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MAELoss.");
  }

  std::unique_ptr<Task> compute_gradient(const Tensor<T> &predictions, const Tensor<T> &targets,
                                         Tensor<T> &gradient) override {
    gradient.resize(predictions.shape());
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task("default", cpu::loss::compute_mae_gradient<T>, predictions.data(),
                             targets.data(), gradient.data(), batch_size, output_size);
    }
#ifdef USE_CUDA
    else if (predictions.device_type() == DeviceType::GPU) {
      return create_gpu_task("default", cuda::loss::compute_mae_gradient<T>, predictions.data(),
                             targets.data(), gradient.data(), batch_size, output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MAELoss.");
  }

  std::string name() const override { return "MAELoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "mae";
    config.name = "MAELoss";
    return config;
  }

  std::unique_ptr<Loss<T>> clone() const override { return std::make_unique<MAELoss<T>>(); }
};

template <typename T = float> class HuberLoss : public Loss<T> {
public:
  explicit HuberLoss(T delta = static_cast<T>(1.0)) : delta_(delta) {}

  std::unique_ptr<Task> compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets,
                                     T &loss) override {
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task("default", cpu::loss::compute_huber_loss<T>, predictions.data(),
                             targets.data(), loss, batch_size, output_size, delta_);
    }
#ifdef USE_CUDA
    else if (predictions.device_type() == DeviceType::GPU) {
      return create_gpu_task("default", cuda::loss::compute_huber_loss<T>, predictions.data(),
                             targets.data(), loss, batch_size, output_size, delta_);
    }
#endif
    throw std::runtime_error("Unsupported device type for HuberLoss.");
  }

  std::unique_ptr<Task> compute_gradient(const Tensor<T> &predictions, const Tensor<T> &targets,
                                         Tensor<T> &gradient) override {
    gradient.resize(predictions.shape());
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    if (predictions.device_type() == DeviceType::CPU) {
      return create_cpu_task("default", cpu::loss::compute_huber_gradient<T>, predictions.data(),
                             targets.data(), gradient.data(), batch_size, output_size, delta_);
    }
#ifdef USE_CUDA
    else if (predictions.device_type() == DeviceType::GPU) {
      return create_gpu_task("default", cuda::loss::compute_huber_gradient<T>, predictions.data(),
                             targets.data(), gradient.data(), batch_size, output_size, delta_);
    }
#endif
    throw std::runtime_error("Unsupported device type for HuberLoss.");
  }

  std::string name() const override { return "HuberLoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "huber";
    config.name = "HuberLoss";
    config.parameters["delta"] = delta_;
    return config;
  }

  std::unique_ptr<Loss<T>> clone() const override { return std::make_unique<HuberLoss<T>>(delta_); }

  void set_delta(T delta) { delta_ = delta; }
  T get_delta() const { return delta_; }

private:
  T delta_;
};

template <typename T = float> class LossFactory {
public:
  static std::unique_ptr<Loss<T>> create(const std::string &loss_type) {
    if (loss_type == "crossentropy" || loss_type == "ce") {
      return std::make_unique<CrossEntropyLoss<T>>();
    }
    if (loss_type == "softmax_crossentropy" || loss_type == "softmax_ce") {
      return std::make_unique<SoftmaxCrossEntropyLoss<T>>();
    }
    if (loss_type == "mse" || loss_type == "mean_squared_error") {
      return std::make_unique<MSELoss<T>>();
    }
    if (loss_type == "mae" || loss_type == "mean_absolute_error") {
      return std::make_unique<MAELoss<T>>();
    }
    if (loss_type == "huber") {
      return std::make_unique<HuberLoss<T>>();
    }
    throw std::invalid_argument("Unknown loss type: " + loss_type);
  }

  static std::unique_ptr<Loss<T>> create_from_config(const LossConfig &config) {
    if (config.type == "crossentropy" || config.type == "ce") {
      T epsilon = config.get<T>("epsilon", static_cast<T>(1e-15));
      return std::make_unique<CrossEntropyLoss<T>>(epsilon);
    }
    if (config.type == "softmax_crossentropy" || config.type == "softmax_ce") {
      return std::make_unique<SoftmaxCrossEntropyLoss<T>>();
    }
    if (config.type == "mse" || config.type == "mean_squared_error") {
      return std::make_unique<MSELoss<T>>();
    }
    if (config.type == "mae" || config.type == "mean_absolute_error") {
      return std::make_unique<MAELoss<T>>();
    }
    if (config.type == "huber") {
      T delta = config.get<T>("delta", static_cast<T>(1.0));
      return std::make_unique<HuberLoss<T>>(delta);
    }
    throw std::invalid_argument("Unknown loss type: " + config.type);
  }

  static std::unique_ptr<Loss<T>> create_crossentropy(T epsilon = static_cast<T>(1e-15)) {
    return std::make_unique<CrossEntropyLoss<T>>(epsilon);
  }

  static std::unique_ptr<Loss<T>> create_softmax_crossentropy() {
    return std::make_unique<SoftmaxCrossEntropyLoss<T>>();
  }

  static std::unique_ptr<Loss<T>> create_mse() { return std::make_unique<MSELoss<T>>(); }

  static std::unique_ptr<Loss<T>> create_mae() { return std::make_unique<MAELoss<T>>(); }

  static std::unique_ptr<Loss<T>> create_huber(T delta = static_cast<T>(1.0)) {
    return std::make_unique<HuberLoss<T>>(delta);
  }
};

} // namespace tnn
