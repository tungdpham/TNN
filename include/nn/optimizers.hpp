/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

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
       float epsilon = 1e-8f)
      : Optimizer<T>(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0),
        initialized_(false) {}

  void update(std::vector<Tensor<T> *> &params, const std::vector<Tensor<T> *> &grads) override {
    if (!initialized_) {
      m_.resize(params.size());
      v_.resize(params.size());
      for (size_t i = 0; i < params.size(); ++i) {
        m_[i] = Tensor<T>(params[i]->shape(), params[i]->device());
        m_[i].fill(0.0f);
        v_[i] = Tensor<T>(params[i]->shape(), params[i]->device());
        v_[i].fill(0.0f);
      }
      initialized_ = true;
    }

    t_++;

    // Precompute scalar coefficients outside the loop
    const T one_minus_beta1 = static_cast<T>(1.0) - beta1_;
    const T one_minus_beta2 = static_cast<T>(1.0) - beta2_;
    const T bias_correction1 = static_cast<T>(1.0) - std::pow(beta1_, static_cast<T>(t_));
    const T bias_correction2 = static_cast<T>(1.0) - std::pow(beta2_, static_cast<T>(t_));
    const T step_size = this->learning_rate_ / bias_correction1;

    parallel_for<size_t>(0, params.size(), [&](size_t i) {
      std::cout << "Updating parameter " << i << std::endl;
      m_[i] *= beta1_;
      m_[i] += (*grads[i]) * one_minus_beta1;
      std::cout << "m updated." << std::endl;

      Tensor<T> grad_sq = (*grads[i]) * (*grads[i]);
      v_[i] *= beta2_;
      v_[i] += grad_sq * one_minus_beta2;
      std::cout << "v updated." << std::endl;

      T *param_data = params[i]->data_ptr().get();
      const T *m_data = m_[i].data_ptr().get();
      const T *v_data = v_[i].data_ptr().get();
      for (size_t j = 0; j < params[i]->size(); ++j) {
        param_data[j] -=
            step_size * m_data[j] / (std::sqrt(v_data[j] / bias_correction2) + epsilon_);
      }
      std::cout << "Parameter " << i << " updated." << std::endl;
    });
  }

  std::string name() const override { return "Adam"; }

  OptimizerConfig get_config() const override {
    OptimizerConfig config;
    config.type = "adam";
    config.name = "Adam";
    config.parameters["learning_rate"] = this->learning_rate_;
    config.parameters["beta1"] = beta1_;
    config.parameters["beta2"] = beta2_;
    config.parameters["epsilon"] = epsilon_;
    return config;
  }

  std::unique_ptr<Optimizer<T>> clone() const override {
    return std::make_unique<Adam<T>>(this->learning_rate_, beta1_, beta2_, epsilon_);
  }

private:
  float beta1_;
  float beta2_;
  float epsilon_;
  unsigned long t_;
  bool initialized_;
  std::vector<Tensor<T>> m_;
  std::vector<Tensor<T>> v_;
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
    if (config.type == "adam") {
      float learning_rate = config.get<float>("learning_rate", 0.001f);
      float beta1 = config.get<float>("beta1", 0.9f);
      float beta2 = config.get<float>("beta2", 0.999f);
      float epsilon = config.get<float>("epsilon", 1e-8f);
      return std::make_unique<Adam<T>>(learning_rate, beta1, beta2, epsilon);
    }
    throw std::invalid_argument("Unknown optimizer type: " + config.type);
  }
};

} // namespace tnn
