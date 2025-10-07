/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "../tensor/tensor.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

  virtual T compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets) = 0;
  virtual Tensor<T> compute_gradient(const Tensor<T> &predictions, const Tensor<T> &targets) = 0;

  virtual std::string name() const = 0;
  virtual LossConfig get_config() const = 0;
  virtual std::unique_ptr<Loss<T>> clone() const = 0;

  virtual size_t num_parameters() const { return 0; }

  virtual void reset() {}
};

template <typename T = float> class CrossEntropyLoss : public Loss<T> {
public:
  explicit CrossEntropyLoss(T epsilon = static_cast<T>(1e-15)) : epsilon_(epsilon) {}

  T compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets) override {
    const size_t batch_size = predictions.shape()[0];
    const size_t num_classes = predictions.shape()[1];

    double total_loss = 0.0;

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_classes; ++j) {
        if (targets(i, j, 0, 0) > static_cast<T>(0.5)) {
          const T pred =
              std::clamp(predictions(i, j, 0, 0), epsilon_, static_cast<T>(1.0) - epsilon_);
          total_loss -= std::log(pred);
          break;
        }
      }
    }

    return static_cast<T>(total_loss / batch_size);
  }

  Tensor<T> compute_gradient(const Tensor<T> &predictions, const Tensor<T> &targets) override {
    Tensor<T> gradient = predictions.clone();
    const size_t batch_size = predictions.shape()[0];
    const size_t num_classes = predictions.shape()[1];
    const T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_classes; ++j) {
        gradient(i, j, 0, 0) = (predictions(i, j, 0, 0) - targets(i, j, 0, 0)) * inv_batch_size;
      }
    }

    return gradient;
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
// Takes raw logits as input (NOT probabilities)
// Uses Log-Sum-Exp trick for numerical stability
template <typename T = float> class SoftmaxCrossEntropyLoss : public Loss<T> {
public:
  SoftmaxCrossEntropyLoss() = default;

  T compute_loss(const Tensor<T> &logits, const Tensor<T> &targets) override {
    const size_t batch_size = logits.shape()[0];
    const size_t num_classes = logits.shape()[1];

    double total_loss = 0.0;

    for (size_t i = 0; i < batch_size; ++i) {
      T max_logit = logits(i, 0, 0, 0);
      for (size_t j = 1; j < num_classes; ++j) {
        max_logit = std::max(max_logit, logits(i, j, 0, 0));
      }

      double sum_exp = 0.0;
      for (size_t j = 0; j < num_classes; ++j) {
        sum_exp += std::exp(static_cast<double>(logits(i, j, 0, 0) - max_logit));
      }
      const T log_sum_exp = static_cast<T>(std::log(sum_exp)) + max_logit;

      for (size_t j = 0; j < num_classes; ++j) {
        if (targets(i, j, 0, 0) > static_cast<T>(0.5)) {
          total_loss += static_cast<double>(log_sum_exp - logits(i, j, 0, 0));
          break;
        }
      }
    }

    return static_cast<T>(total_loss / batch_size);
  }

  Tensor<T> compute_gradient(const Tensor<T> &logits, const Tensor<T> &targets) override {
    const size_t batch_size = logits.shape()[0];
    const size_t num_classes = logits.shape()[1];
    const T inv_batch_size = static_cast<T>(1.0) / static_cast<T>(batch_size);

    Tensor<T> gradient = logits.clone();

    for (size_t i = 0; i < batch_size; ++i) {
      T max_logit = logits(i, 0, 0, 0);
      for (size_t j = 1; j < num_classes; ++j) {
        max_logit = std::max(max_logit, logits(i, j, 0, 0));
      }

      double sum_exp = 0.0;
      for (size_t j = 0; j < num_classes; ++j) {
        sum_exp += std::exp(static_cast<double>(logits(i, j, 0, 0) - max_logit));
      }

      for (size_t j = 0; j < num_classes; ++j) {
        const T softmax_prob =
            static_cast<T>(std::exp(static_cast<double>(logits(i, j, 0, 0) - max_logit)) / sum_exp);
        gradient(i, j, 0, 0) = (softmax_prob - targets(i, j, 0, 0)) * inv_batch_size;
      }
    }

    return gradient;
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

  T compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets) override {
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    double total_loss = 0.0;

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < output_size; ++j) {
        const T diff = predictions(i, j, 0, 0) - targets(i, j, 0, 0);
        total_loss += static_cast<double>(diff * diff);
      }
    }

    return static_cast<T>(total_loss / (batch_size * output_size));
  }

  Tensor<T> compute_gradient(const Tensor<T> &predictions, const Tensor<T> &targets) override {
    Tensor<T> gradient = predictions;
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];
    const T scale = static_cast<T>(2.0) / static_cast<T>(batch_size * output_size);

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < output_size; ++j) {
        gradient(i, j, 0, 0) = (predictions(i, j, 0, 0) - targets(i, j, 0, 0)) * scale;
      }
    }

    return gradient;
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

  T compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets) override {
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    double total_loss = 0.0;

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < output_size; ++j) {
        total_loss += std::abs(predictions(i, j, 0, 0) - targets(i, j, 0, 0));
      }
    }

    return static_cast<T>(total_loss / (batch_size * output_size));
  }

  Tensor<T> compute_gradient(const Tensor<T> &predictions, const Tensor<T> &targets) override {
    Tensor<T> gradient = predictions;
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];
    const T scale = static_cast<T>(1.0) / static_cast<T>(batch_size * output_size);

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < output_size; ++j) {
        const T diff = predictions(i, j, 0, 0) - targets(i, j, 0, 0);
        gradient(i, j, 0, 0) = (diff > static_cast<T>(0) ? scale : -scale);
      }
    }

    return gradient;
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

  T compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets) override {
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    double total_loss = 0.0;

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < output_size; ++j) {
        const T diff = std::abs(predictions(i, j, 0, 0) - targets(i, j, 0, 0));
        if (diff <= delta_) {
          total_loss += static_cast<double>(0.5 * diff * diff);
        } else {
          total_loss += static_cast<double>(delta_ * diff - 0.5 * delta_ * delta_);
        }
      }
    }

    return static_cast<T>(total_loss / (batch_size * output_size));
  }

  Tensor<T> compute_gradient(const Tensor<T> &predictions, const Tensor<T> &targets) override {
    Tensor<T> gradient = predictions;
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];
    const T scale = static_cast<T>(1.0) / static_cast<T>(batch_size * output_size);

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < output_size; ++j) {
        const T diff = predictions(i, j, 0, 0) - targets(i, j, 0, 0);
        const T abs_diff = std::abs(diff);

        if (abs_diff <= delta_) {
          gradient(i, j, 0, 0) = diff * scale;
        } else {
          gradient(i, j, 0, 0) = (diff > static_cast<T>(0) ? delta_ : -delta_) * scale;
        }
      }
    }

    return gradient;
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
