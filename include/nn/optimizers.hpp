/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device_ptr.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include <any>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

struct OptimizerConfig {
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

template <typename T = float> class Optimizer {
public:
  explicit Optimizer(float learning_rate) : learning_rate_(learning_rate) {}
  virtual ~Optimizer() = default;

  virtual void update(std::vector<Tensor<T> *> &params, const std::vector<Tensor<T> *> &grads) = 0;

  void set_learning_rate(float lr) { learning_rate_ = lr; }
  float get_learning_rate() const { return learning_rate_; }

  virtual std::string name() const = 0;
  virtual OptimizerConfig get_config() const = 0;
  virtual std::unique_ptr<Optimizer<T>> clone() const = 0;

protected:
  float learning_rate_;
};

template <typename T = float> class SGD : public Optimizer<T> {
public:
  SGD(float learning_rate = 0.01f, float momentum = 0.0f)
      : Optimizer<T>(learning_rate), momentum_(momentum), initialized_(false) {}

  void update(std::vector<Tensor<T> *> &params, const std::vector<Tensor<T> *> &grads) override {
    if (momentum_ > 0.0f && !initialized_) {
      velocities_.resize(params.size());
      for (size_t i = 0; i < params.size(); ++i) {
        velocities_[i] = Tensor<T>(params[i]->shape(), params[i]->device());
        velocities_[i].fill(0.0f);
      }
      initialized_ = true;
    }

#ifndef USE_CUDA
    parallel_for<size_t>(0, params.size(), [&](size_t i) {
      if (momentum_ > 0.0f) {
        velocities_[i] *= momentum_;
        Tensor<T> scaled_grad = (*grads[i]) * this->learning_rate_;
        velocities_[i] -= scaled_grad;
        (*params[i]) += velocities_[i];
      } else {
        Tensor<T> scaled_grad = (*grads[i]) * this->learning_rate_;
        (*params[i]) -= scaled_grad;
      }
    });
#else
    for (size_t i = 0; i < params.size(); ++i) {
      if (momentum_ > 0.0f) {
        ops::mul_scalar(velocities_[i].data_ptr(), momentum_, velocities_[i].data_ptr(),
                        velocities_[i].size());
        Tensor<T> scaled_grad = (*grads[i]) * this->learning_rate_;
        ops::axpy(-1.0f, scaled_grad.data_ptr(), velocities_[i].data_ptr(), velocities_[i].size());
        ops::axpy(1.0f, velocities_[i].data_ptr(), (*params[i]).data_ptr(), params[i]->size());
      } else {
        Tensor<T> scaled_grad = (*grads[i]) * this->learning_rate_;
        ops::axpy(-1.0f, scaled_grad.data_ptr(), (*params[i]).data_ptr(), params[i]->size());
      }
    }
#endif
  }

  std::string name() const override { return "SGD"; }

  OptimizerConfig get_config() const override {
    OptimizerConfig config;
    config.type = "sgd";
    config.name = "SGD";
    config.parameters["learning_rate"] = this->learning_rate_;
    config.parameters["momentum"] = momentum_;
    return config;
  }

  std::unique_ptr<Optimizer<T>> clone() const override {
    return std::make_unique<SGD<T>>(this->learning_rate_, momentum_);
  }

private:
  float momentum_;
  bool initialized_;
  std::vector<Tensor<T>> velocities_;
};

template <typename T = float> class Adam : public Optimizer<T> {
public:
  Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
       float epsilon = 1e-8f, float weight_decay = 0.0f, bool decouple_weight_decay = false)
      : Optimizer<T>(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
        weight_decay_(weight_decay), decouple_weight_decay_(decouple_weight_decay), t_(0),
        initialized_(false) {}

  void update(std::vector<Tensor<T> *> &params, const std::vector<Tensor<T> *> &grads) override {
    if (!initialized_) {
      m_.resize(params.size());
      v_.resize(params.size());
      temp_buffers_.resize(params.size());
      for (size_t i = 0; i < params.size(); ++i) {
        m_[i] = Tensor<T>(params[i]->shape(), params[i]->device());
        m_[i].fill(0.0f);
        v_[i] = Tensor<T>(params[i]->shape(), params[i]->device());
        v_[i].fill(0.0f);
        // Pre-allocate temporary buffers to avoid allocations in hot loop
        temp_buffers_[i].grad_sq = Tensor<T>(params[i]->shape(), params[i]->device());
        temp_buffers_[i].v_sqrt = Tensor<T>(params[i]->shape(), params[i]->device());
        temp_buffers_[i].m_hat = Tensor<T>(params[i]->shape(), params[i]->device());
        temp_buffers_[i].v_hat = Tensor<T>(params[i]->shape(), params[i]->device());
      }
      initialized_ = true;
    }

    t_++;

    // Precompute scalar coefficients outside the loop
    const T one_minus_beta1 = static_cast<T>(1.0) - beta1_;
    const T one_minus_beta2 = static_cast<T>(1.0) - beta2_;
    const T bias_correction1 = static_cast<T>(1.0) - std::pow(beta1_, static_cast<T>(t_));
    const T bias_correction2 = static_cast<T>(1.0) - std::pow(beta2_, static_cast<T>(t_));

    parallel_for<size_t>(0, params.size(), [&](size_t i) {
      // Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
      m_[i] *= beta1_;
      m_[i] += (*grads[i]) * one_minus_beta1;

      // Update biased second raw moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
      temp_buffers_[i].grad_sq = (*grads[i]) * (*grads[i]);
      v_[i] *= beta2_;
      v_[i] += temp_buffers_[i].grad_sq * one_minus_beta2;

      // Compute bias-corrected first moment estimate: m̂_t = m_t / (1 - β₁^t)
      temp_buffers_[i].m_hat = m_[i] / bias_correction1;

      // Compute bias-corrected second raw moment estimate: v̂_t = v_t / (1 - β₂^t)
      temp_buffers_[i].v_hat = v_[i] / bias_correction2;

      // Compute sqrt(v̂_t)
      ops::sqrt(temp_buffers_[i].v_hat.data_ptr(), temp_buffers_[i].v_sqrt.data_ptr(),
                temp_buffers_[i].v_hat.size());

      // Compute denominator: sqrt(v̂_t) + ε (with safe operator precedence)
      auto denom = temp_buffers_[i].v_sqrt + epsilon_;

      // Parameter update: θ_t = θ_{t-1} - α * m̂_t / (sqrt(v̂_t) + ε)
      // Use explicit denominator to ensure correct operator precedence
      auto update = (temp_buffers_[i].m_hat / denom) * this->learning_rate_;

      // Apply weight decay (AdamW-style decoupled weight decay if enabled)
      if (weight_decay_ > 0.0f) {
        if (decouple_weight_decay_) {
          // AdamW: θ_t = θ_{t-1} - λ * θ_{t-1} - α * m̂_t / (sqrt(v̂_t) + ε)
          // Decoupled weight decay (applied directly to parameters)
          (*params[i]) -= (*params[i]) * (weight_decay_ * this->learning_rate_);
        } else {
          // Adam with L2 regularization: add λ * θ_{t-1} to gradient
          // This is equivalent to adding weight decay to the gradient before momentum
          update += (*params[i]) * (weight_decay_ * this->learning_rate_);
        }
      }

      (*params[i]) -= update;
    });
  }

  std::string name() const override { return decouple_weight_decay_ ? "AdamW" : "Adam"; }

  OptimizerConfig get_config() const override {
    OptimizerConfig config;
    config.type = decouple_weight_decay_ ? "adamw" : "adam";
    config.name = decouple_weight_decay_ ? "AdamW" : "Adam";
    config.parameters["learning_rate"] = this->learning_rate_;
    config.parameters["beta1"] = beta1_;
    config.parameters["beta2"] = beta2_;
    config.parameters["epsilon"] = epsilon_;
    config.parameters["weight_decay"] = weight_decay_;
    config.parameters["decouple_weight_decay"] = decouple_weight_decay_;
    return config;
  }

  std::unique_ptr<Optimizer<T>> clone() const override {
    return std::make_unique<Adam<T>>(this->learning_rate_, beta1_, beta2_, epsilon_, weight_decay_,
                                     decouple_weight_decay_);
  }

private:
  float beta1_;
  float beta2_;
  float epsilon_;
  float weight_decay_;
  bool decouple_weight_decay_;
  unsigned long t_;
  bool initialized_;
  std::vector<Tensor<T>> m_;
  std::vector<Tensor<T>> v_;

  // Pre-allocated temporary buffers to avoid allocations in hot loop
  struct TempBuffers {
    Tensor<T> grad_sq; // For squared gradients
    Tensor<T> v_sqrt;  // For sqrt(v_hat)
    Tensor<T> m_hat;   // For bias-corrected first moment
    Tensor<T> v_hat;   // For bias-corrected second moment
  };
  std::vector<TempBuffers> temp_buffers_;
};

template <typename T = float> class OptimizerFactory {
public:
  static std::unique_ptr<Optimizer<T>> create(const std::string &name, float learning_rate,
                                              float momentum = 0.9f) {
    if (name == "sgd") {
      return std::make_unique<SGD<T>>(learning_rate, momentum);
    }
    if (name == "adam") {
      return std::make_unique<Adam<T>>(learning_rate);
    }
    throw std::invalid_argument("Unknown optimizer type: " + name);
  }

  static std::unique_ptr<Optimizer<T>> create_from_config(const OptimizerConfig &config) {
    if (config.type == "sgd") {
      float learning_rate = config.get<float>("learning_rate", 0.01f);
      float momentum = config.get<float>("momentum", 0.0f);
      return std::make_unique<SGD<T>>(learning_rate, momentum);
    }
    if (config.type == "adam" || config.type == "adamw") {
      float learning_rate = config.get<float>("learning_rate", 0.001f);
      float beta1 = config.get<float>("beta1", 0.9f);
      float beta2 = config.get<float>("beta2", 0.999f);
      float epsilon = config.get<float>("epsilon", 1e-8f);
      float weight_decay = config.get<float>("weight_decay", 0.0f);
      bool decouple_weight_decay =
          config.get<bool>("decouple_weight_decay", config.type == "adamw");
      return std::make_unique<Adam<T>>(learning_rate, beta1, beta2, epsilon, weight_decay,
                                       decouple_weight_decay);
    }
    throw std::invalid_argument("Unknown optimizer type: " + config.type);
  }
};

} // namespace tnn
