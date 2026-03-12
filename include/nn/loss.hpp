/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "common/config.hpp"
#include "device/task.hpp"
#include "loss_impl/cpu/loss_ops.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDA
#include "loss_impl/cuda/loss_ops.hpp"
#endif
#include <memory>
#include <stdexcept>
#include <string>

namespace tnn {

using LossConfig = TConfig;

class Loss {
public:
  virtual ~Loss() = default;

  std::unique_ptr<Task> compute_loss(const ConstTensor &predictions, const ConstTensor &targets,
                                     float &loss) {
    if (!predictions || !targets) {
      throw std::runtime_error("Predictions and targets cannot be null for compute_loss.");
    }
    if (predictions->device() != targets->device()) {
      throw std::runtime_error(
          "Predictions and targets must be on the same device for compute_loss.");
    }
    return compute_loss_impl(predictions, targets, loss);
  }
  std::unique_ptr<Task> compute_gradient(const ConstTensor &predictions, const ConstTensor &targets,
                                         const Tensor &gradient) {
    if (!predictions || !targets || !gradient) {
      throw std::runtime_error(
          "Predictions, targets, and gradient cannot be null for compute_gradient.");
    }
    if (predictions->device() != targets->device() || predictions->device() != gradient->device()) {
      throw std::runtime_error(
          "Predictions, targets, and gradient must be on the same device for compute_gradient.");
    }
    return compute_gradient_impl(predictions, targets, gradient);
  }

  virtual std::string name() const = 0;
  virtual LossConfig get_config() const = 0;
  virtual std::unique_ptr<Loss> clone() const = 0;

  virtual size_t num_parameters() const { return 0; }

  virtual void reset() {}

protected:
  virtual std::unique_ptr<Task> compute_loss_impl(const ConstTensor &predictions,
                                                  const ConstTensor &targets, float &loss) = 0;
  virtual std::unique_ptr<Task> compute_gradient_impl(const ConstTensor &predictions,
                                                      const ConstTensor &targets,
                                                      const Tensor &gradient) = 0;
};

class CrossEntropyLoss : public Loss {
public:
  explicit CrossEntropyLoss(double epsilon = 1e-15)
      : epsilon_(epsilon) {}

  std::string name() const override { return "CrossEntropyLoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "crossentropy";
    config.name = "CrossEntropyLoss";
    config.set("epsilon", epsilon_);
    return config;
  }

  std::unique_ptr<Loss> clone() const override {
    return std::make_unique<CrossEntropyLoss>(epsilon_);
  }

private:
  double epsilon_;

  std::unique_ptr<Task> compute_loss_impl(const ConstTensor &predictions,
                                          const ConstTensor &targets, float &loss) override {
    DISPATCH_DTYPE(predictions->data_type(), T,
                   return compute_loss_t<T>(predictions, targets, loss));
  }

  std::unique_ptr<Task> compute_gradient_impl(const ConstTensor &predictions,
                                              const ConstTensor &targets,
                                              const Tensor &gradient) override {
    DISPATCH_DTYPE(predictions->data_type(), T,
                   return compute_gradient_t<T>(predictions, targets, gradient));
  }

  template <typename T>
  std::unique_ptr<Task> compute_loss_t(const ConstTensor &predictions, const ConstTensor &targets,
                                       float &loss) {
    const size_t num_classes = predictions->shape().back();
    size_t batch_size = 1;
    for (size_t i = 0; i < predictions->dims() - 1; ++i) {
      batch_size *= predictions->shape()[i];
    }

    if (predictions->device_type() == DeviceType::CPU) {
      return create_cpu_task(defaultFlowHandle, cpu::loss::compute_crossentropy_loss<T>,
                             predictions->data_as<T>(), targets->data_as<T>(), loss, batch_size,
                             num_classes, static_cast<T>(epsilon_));
    }
#ifdef USE_CUDA
    else if (predictions->device_type() == DeviceType::GPU) {
      return create_cuda_task(defaultFlowHandle, cuda::loss::compute_crossentropy_loss<T>,
                              predictions->data_as<T>(), targets->data_as<T>(), loss, batch_size,
                              num_classes, static_cast<T>(epsilon_));
    }
#endif
    throw std::runtime_error("Unsupported device type for CrossEntropyLoss.");
  }

  template <typename T>
  std::unique_ptr<Task> compute_gradient_t(const ConstTensor &predictions,
                                           const ConstTensor &targets, const Tensor &gradient) {
    gradient->ensure(predictions->shape());
    const size_t num_classes = predictions->shape().back();
    size_t batch_size = 1;
    for (size_t i = 0; i < predictions->dims() - 1; ++i) {
      batch_size *= predictions->shape()[i];
    }

    if (predictions->device_type() == DeviceType::CPU) {
      return create_cpu_task(defaultFlowHandle, cpu::loss::compute_crossentropy_gradient<T>,
                             predictions->data_as<T>(), targets->data_as<T>(),
                             gradient->data_as<T>(), batch_size, num_classes,
                             static_cast<T>(epsilon_));
    }
#ifdef USE_CUDA
    else if (predictions->device_type() == DeviceType::GPU) {
      return create_cuda_task(defaultFlowHandle, cuda::loss::compute_crossentropy_gradient<T>,
                              predictions->data_as<T>(), targets->data_as<T>(),
                              gradient->data_as<T>(), batch_size, num_classes,
                              static_cast<T>(epsilon_));
    }
#endif
    throw std::runtime_error("Unsupported device type for CrossEntropyLoss.");
  }
};

// Numerically stable LogSoftmax + CrossEntropy combined loss
class LogSoftmaxCrossEntropyLoss : public Loss {
public:
  LogSoftmaxCrossEntropyLoss() = default;

  std::string name() const override { return "LogSoftmaxCrossEntropyLoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "logsoftmax_crossentropy";
    config.name = "LogSoftmaxCrossEntropyLoss";
    return config;
  }

  std::unique_ptr<Loss> clone() const override {
    return std::make_unique<LogSoftmaxCrossEntropyLoss>();
  }

private:
  std::unique_ptr<Task> compute_loss_impl(const ConstTensor &logits, const ConstTensor &targets,
                                          float &loss) override {
    if (logits->device() != targets->device()) {
      throw std::runtime_error(
          "Logits and targets must be on the same device for LogSoftmaxCrossEntropyLoss.");
    }
    DISPATCH_DTYPE(logits->data_type(), T, return compute_loss_t<T>(logits, targets, loss));
  }

  std::unique_ptr<Task> compute_gradient_impl(const ConstTensor &logits, const ConstTensor &targets,
                                              const Tensor &gradient) override {
    if (logits->device() != targets->device() || logits->device() != gradient->device()) {
      throw std::runtime_error(
          "Logits, targets, and gradient must be on the same device for "
          "LogSoftmaxCrossEntropyLoss.");
    }
    DISPATCH_DTYPE(logits->data_type(), T, return compute_gradient_t<T>(logits, targets, gradient));
  }

  template <typename T>
  std::unique_ptr<Task> compute_loss_t(const ConstTensor &logits, const ConstTensor &targets,
                                       float &loss) {
    const size_t num_classes = logits->shape().back();
    size_t batch_size = 1;
    for (size_t i = 0; i < logits->dims() - 1; ++i) {
      batch_size *= logits->shape()[i];
    }

    if (logits->device_type() == DeviceType::CPU) {
      return create_cpu_task(defaultFlowHandle, cpu::loss::compute_logsoftmax_crossentropy_loss<T>,
                             logits->data_as<T>(), targets->data_as<T>(), loss, batch_size,
                             num_classes);
    }
#ifdef USE_CUDA
    else if (logits->device_type() == DeviceType::GPU) {
      return create_cuda_task(
          defaultFlowHandle, cuda::loss::compute_logsoftmax_crossentropy_loss<T>,
          logits->data_as<T>(), targets->data_as<T>(), loss, batch_size, num_classes);
    }
#endif
    throw std::runtime_error("Unsupported device type for LogSoftmaxCrossEntropyLoss.");
  }

  template <typename T>
  std::unique_ptr<Task> compute_gradient_t(const ConstTensor &logits, const ConstTensor &targets,
                                           const Tensor &gradient) {
    gradient->ensure(logits->shape());
    const size_t num_classes = logits->shape().back();
    size_t batch_size = 1;
    for (size_t i = 0; i < logits->dims() - 1; ++i) {
      batch_size *= logits->shape()[i];
    }

    if (logits->device_type() == DeviceType::CPU) {
      return create_cpu_task(defaultFlowHandle,
                             cpu::loss::compute_logsoftmax_crossentropy_gradient<T>,
                             logits->data_as<T>(), targets->data_as<T>(), gradient->data_as<T>(),
                             batch_size, num_classes);
    }
#ifdef USE_CUDA
    else if (logits->device_type() == DeviceType::GPU) {
      return create_cuda_task(defaultFlowHandle,
                              cuda::loss::compute_logsoftmax_crossentropy_gradient<T>,
                              logits->data_as<T>(), targets->data_as<T>(), gradient->data_as<T>(),
                              batch_size, num_classes);
    }
#endif
    throw std::runtime_error("Unsupported device type for LogSoftmaxCrossEntropyLoss.");
  }
};

class MSELoss : public Loss {
public:
  MSELoss() = default;

  std::string name() const override { return "MSELoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "mse";
    config.name = "MSELoss";
    return config;
  }

  std::unique_ptr<Loss> clone() const override { return std::make_unique<MSELoss>(); }

private:
  std::unique_ptr<Task> compute_loss_impl(const ConstTensor &predictions,
                                          const ConstTensor &targets, float &loss) override {
    if (predictions->device() != targets->device()) {
      throw std::runtime_error("Predictions and targets must be on the same device for MSELoss.");
    }
    DISPATCH_DTYPE(predictions->data_type(), T,
                   return compute_loss_t<T>(predictions, targets, loss));
  }

  std::unique_ptr<Task> compute_gradient_impl(const ConstTensor &predictions,
                                              const ConstTensor &targets,
                                              const Tensor &gradient) override {
    if (predictions->device() != targets->device() || predictions->device() != gradient->device()) {
      throw std::runtime_error(
          "Predictions, targets, and gradient must be on the same device for MSELoss.");
    }
    DISPATCH_DTYPE(predictions->data_type(), T,
                   return compute_gradient_t<T>(predictions, targets, gradient));
  }

  template <typename T>
  std::unique_ptr<Task> compute_loss_t(const ConstTensor &predictions, const ConstTensor &targets,
                                       float &loss) {
    const size_t batch_size = predictions->shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions->dims(); ++i) {
      output_size *= predictions->shape()[i];
    }

    if (predictions->device_type() == DeviceType::CPU) {
      return create_cpu_task(defaultFlowHandle, cpu::loss::compute_mse_loss<T>,
                             predictions->data_as<T>(), targets->data_as<T>(), loss, batch_size,
                             output_size);
    }
#ifdef USE_CUDA
    else if (predictions->device_type() == DeviceType::GPU) {
      return create_cuda_task(defaultFlowHandle, cuda::loss::compute_mse_loss<T>,
                              predictions->data_as<T>(), targets->data_as<T>(), loss, batch_size,
                              output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MSELoss.");
  }

  template <typename T>
  std::unique_ptr<Task> compute_gradient_t(const ConstTensor &predictions,
                                           const ConstTensor &targets, const Tensor &gradient) {
    gradient->ensure(predictions->shape());
    const size_t batch_size = predictions->shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions->dims(); ++i) {
      output_size *= predictions->shape()[i];
    }

    if (predictions->device_type() == DeviceType::CPU) {
      return create_cpu_task(defaultFlowHandle, cpu::loss::compute_mse_gradient<T>,
                             predictions->data_as<T>(), targets->data_as<T>(),
                             gradient->data_as<T>(), batch_size, output_size);
    }
#ifdef USE_CUDA
    else if (predictions->device_type() == DeviceType::GPU) {
      return create_cuda_task(defaultFlowHandle, cuda::loss::compute_mse_gradient<T>,
                              predictions->data_as<T>(), targets->data_as<T>(),
                              gradient->data_as<T>(), batch_size, output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MSELoss.");
  }
};

class MAELoss : public Loss {
public:
  MAELoss() = default;

  std::string name() const override { return "MAELoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "mae";
    config.name = "MAELoss";
    return config;
  }

  std::unique_ptr<Loss> clone() const override { return std::make_unique<MAELoss>(); }

private:
  std::unique_ptr<Task> compute_loss_impl(const ConstTensor &predictions,
                                          const ConstTensor &targets, float &loss) override {
    if (predictions->device() != targets->device()) {
      throw std::runtime_error("Predictions and targets must be on the same device for MAELoss.");
    }
    DISPATCH_DTYPE(predictions->data_type(), T,
                   return compute_loss_t<T>(predictions, targets, loss));
  }

  std::unique_ptr<Task> compute_gradient_impl(const ConstTensor &predictions,
                                              const ConstTensor &targets,
                                              const Tensor &gradient) override {
    if (predictions->device() != targets->device() || predictions->device() != gradient->device()) {
      throw std::runtime_error(
          "Predictions, targets, and gradient must be on the same device for MAELoss.");
    }
    DISPATCH_DTYPE(predictions->data_type(), T,
                   return compute_gradient_t<T>(predictions, targets, gradient));
  }

  template <typename T>
  std::unique_ptr<Task> compute_loss_t(const ConstTensor &predictions, const ConstTensor &targets,
                                       float &loss) {
    const size_t batch_size = predictions->shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions->dims(); ++i) {
      output_size *= predictions->shape()[i];
    }

    if (predictions->device_type() == DeviceType::CPU) {
      return create_cpu_task(defaultFlowHandle, cpu::loss::compute_mae_loss<T>,
                             predictions->data_as<T>(), targets->data_as<T>(), loss, batch_size,
                             output_size);
    }
#ifdef USE_CUDA
    else if (predictions->device_type() == DeviceType::GPU) {
      return create_cuda_task(defaultFlowHandle, cuda::loss::compute_mae_loss<T>,
                              predictions->data_as<T>(), targets->data_as<T>(), loss, batch_size,
                              output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MAELoss.");
  }

  template <typename T>
  std::unique_ptr<Task> compute_gradient_t(const ConstTensor &predictions,
                                           const ConstTensor &targets, const Tensor &gradient) {
    gradient->ensure(predictions->shape());
    const size_t batch_size = predictions->shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions->dims(); ++i) {
      output_size *= predictions->shape()[i];
    }

    if (predictions->device_type() == DeviceType::CPU) {
      return create_cpu_task(defaultFlowHandle, cpu::loss::compute_mae_gradient<T>,
                             predictions->data_as<T>(), targets->data_as<T>(),
                             gradient->data_as<T>(), batch_size, output_size);
    }
#ifdef USE_CUDA
    else if (predictions->device_type() == DeviceType::GPU) {
      return create_cuda_task(defaultFlowHandle, cuda::loss::compute_mae_gradient<T>,
                              predictions->data_as<T>(), targets->data_as<T>(),
                              gradient->data_as<T>(), batch_size, output_size);
    }
#endif
    throw std::runtime_error("Unsupported device type for MAELoss.");
  }
};

class HuberLoss : public Loss {
public:
  explicit HuberLoss(double delta = 1.0)
      : delta_(delta) {}

  std::string name() const override { return "HuberLoss"; }

  LossConfig get_config() const override {
    LossConfig config;
    config.type = "huber";
    config.name = "HuberLoss";
    config.set("delta", delta_);
    return config;
  }

  std::unique_ptr<Loss> clone() const override { return std::make_unique<HuberLoss>(delta_); }

  void set_delta(double delta) { delta_ = delta; }
  double get_delta() const { return delta_; }

private:
  double delta_;

  std::unique_ptr<Task> compute_loss_impl(const ConstTensor &predictions,
                                          const ConstTensor &targets, float &loss) override {
    if (predictions->device() != targets->device()) {
      throw std::runtime_error("Predictions and targets must be on the same device for HuberLoss.");
    }
    DISPATCH_DTYPE(predictions->data_type(), T,
                   return compute_loss_t<T>(predictions, targets, loss));
  }

  std::unique_ptr<Task> compute_gradient_impl(const ConstTensor &predictions,
                                              const ConstTensor &targets,
                                              const Tensor &gradient) override {
    if (predictions->device() != targets->device() || predictions->device() != gradient->device()) {
      throw std::runtime_error(
          "Predictions, targets, and gradient must be on the same device for HuberLoss.");
    }
    DISPATCH_DTYPE(predictions->data_type(), T,
                   return compute_gradient_t<T>(predictions, targets, gradient));
  }

  template <typename T>
  std::unique_ptr<Task> compute_loss_t(const ConstTensor &predictions, const ConstTensor &targets,
                                       float &loss) {
    const size_t batch_size = predictions->shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions->dims(); ++i) {
      output_size *= predictions->shape()[i];
    }

    if (predictions->device_type() == DeviceType::CPU) {
      return create_cpu_task(defaultFlowHandle, cpu::loss::compute_huber_loss<T>,
                             predictions->data_as<T>(), targets->data_as<T>(), loss, batch_size,
                             output_size, static_cast<T>(delta_));
    }
#ifdef USE_CUDA
    else if (predictions->device_type() == DeviceType::GPU) {
      return create_cuda_task(defaultFlowHandle, cuda::loss::compute_huber_loss<T>,
                              predictions->data_as<T>(), targets->data_as<T>(), loss, batch_size,
                              output_size, static_cast<T>(delta_));
    }
#endif
    throw std::runtime_error("Unsupported device type for HuberLoss.");
  }

  template <typename T>
  std::unique_ptr<Task> compute_gradient_t(const ConstTensor &predictions,
                                           const ConstTensor &targets, const Tensor &gradient) {
    gradient->ensure(predictions->shape());
    const size_t batch_size = predictions->shape()[0];
    size_t output_size = 1;
    for (size_t i = 1; i < predictions->dims(); ++i) {
      output_size *= predictions->shape()[i];
    }

    if (predictions->device_type() == DeviceType::CPU) {
      return create_cpu_task(defaultFlowHandle, cpu::loss::compute_huber_gradient<T>,
                             predictions->data_as<T>(), targets->data_as<T>(),
                             gradient->data_as<T>(), batch_size, output_size,
                             static_cast<T>(delta_));
    }
#ifdef USE_CUDA
    else if (predictions->device_type() == DeviceType::GPU) {
      return create_cuda_task(defaultFlowHandle, cuda::loss::compute_huber_gradient<T>,
                              predictions->data_as<T>(), targets->data_as<T>(),
                              gradient->data_as<T>(), batch_size, output_size,
                              static_cast<T>(delta_));
    }
#endif
    throw std::runtime_error("Unsupported device type for HuberLoss.");
  }
};

class LossFactory {
public:
  static std::unique_ptr<Loss> create(const std::string &loss_type) {
    if (loss_type == "crossentropy" || loss_type == "ce") {
      return std::make_unique<CrossEntropyLoss>();
    }
    if (loss_type == "logsoftmax_crossentropy" || loss_type == "logsoftmax_ce") {
      return std::make_unique<LogSoftmaxCrossEntropyLoss>();
    }
    if (loss_type == "mse" || loss_type == "mean_squared_error") {
      return std::make_unique<MSELoss>();
    }
    if (loss_type == "mae" || loss_type == "mean_absolute_error") {
      return std::make_unique<MAELoss>();
    }
    if (loss_type == "huber") {
      return std::make_unique<HuberLoss>();
    }
    throw std::invalid_argument("Unknown loss type: " + loss_type);
  }

  static std::unique_ptr<Loss> create_from_config(const LossConfig &config) {
    if (config.type == "crossentropy" || config.type == "ce") {
      double epsilon = config.get<double>("epsilon", 1e-15);
      return std::make_unique<CrossEntropyLoss>(epsilon);
    }
    if (config.type == "logsoftmax_crossentropy" || config.type == "logsoftmax_ce") {
      return std::make_unique<LogSoftmaxCrossEntropyLoss>();
    }
    if (config.type == "mse" || config.type == "mean_squared_error") {
      return std::make_unique<MSELoss>();
    }
    if (config.type == "mae" || config.type == "mean_absolute_error") {
      return std::make_unique<MAELoss>();
    }
    if (config.type == "huber") {
      double delta = config.get("delta", 1.0);
      return std::make_unique<HuberLoss>(delta);
    }
    throw std::invalid_argument("Unknown loss type: " + config.type);
  }

  static std::unique_ptr<Loss> create_crossentropy(double epsilon = 1e-15) {
    return std::make_unique<CrossEntropyLoss>(epsilon);
  }

  static std::unique_ptr<Loss> create_logsoftmax_crossentropy() {
    return std::make_unique<LogSoftmaxCrossEntropyLoss>();
  }

  static std::unique_ptr<Loss> create_mse() { return std::make_unique<MSELoss>(); }

  static std::unique_ptr<Loss> create_mae() { return std::make_unique<MAELoss>(); }

  static std::unique_ptr<Loss> create_huber(double delta = 1.0) {
    return std::make_unique<HuberLoss>(delta);
  }
};

}  // namespace tnn
