/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#include "optimizers_impl/cpu/adam_kernels.hpp"
#include "optimizers_impl/cpu/sgd_kernels.hpp"
#include "optimizers_impl/cuda/adam_kernels.hpp"
#include "optimizers_impl/cuda/sgd_kernels.hpp"
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

    for (size_t i = 0; i < params.size(); ++i) {
      const size_t size = params[i]->size();

      if (params[i]->device_type() == DeviceType::CPU) {
        if (momentum_ > 0.0f) {
          create_cpu_task("default", cpu::sgd::update_sgd_momentum<T>, params[i]->data_ptr().get(),
                          grads[i]->data_ptr().get(), velocities_[i].data_ptr().get(), size,
                          this->learning_rate_, momentum_);
        } else {
          create_cpu_task("default", cpu::sgd::update_sgd<T>, params[i]->data_ptr().get(),
                          grads[i]->data_ptr().get(), size, this->learning_rate_);
        }
      }
#ifdef USE_CUDA
      else if (params[i]->device_type() == DeviceType::GPU) {
        if (momentum_ > 0.0f) {
          create_gpu_task("default", cuda::sgd::update_sgd_momentum<T>, params[i]->data_ptr().get(),
                          grads[i]->data_ptr().get(), velocities_[i].data_ptr().get(), size,
                          this->learning_rate_, momentum_);
        } else {
          create_gpu_task("default", cuda::sgd::update_sgd<T>, params[i]->data_ptr().get(),
                          grads[i]->data_ptr().get(), size, this->learning_rate_);
        }
      }
#endif
      else {
        throw std::runtime_error("Unsupported device type for SGD optimizer");
      }
    }
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
      for (size_t i = 0; i < params.size(); ++i) {
        m_[i] = Tensor<T>(params[i]->shape(), params[i]->device());
        m_[i].fill(0.0f);
        v_[i] = Tensor<T>(params[i]->shape(), params[i]->device());
        v_[i].fill(0.0f);
      }
      initialized_ = true;
    }

    t_++;

    // Precompute bias correction terms outside the loop
    const float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    const float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params.size(); ++i) {
      const size_t size = params[i]->size();

      if (params[i]->device_type() == DeviceType::CPU) {
        create_cpu_task("default", cpu::adam::update_adam<T>, params[i]->data_ptr().get(),
                        grads[i]->data_ptr().get(), m_[i].data_ptr().get(), v_[i].data_ptr().get(),
                        size, this->learning_rate_, beta1_, beta2_, epsilon_, bias_correction1,
                        bias_correction2, weight_decay_, decouple_weight_decay_);
      }
#ifdef USE_CUDA
      else if (params[i]->device_type() == DeviceType::GPU) {
        create_gpu_task("default", cuda::adam::update_adam<T>, params[i]->data_ptr().get(),
                        grads[i]->data_ptr().get(), m_[i].data_ptr().get(), v_[i].data_ptr().get(),
                        size, this->learning_rate_, beta1_, beta2_, epsilon_, bias_correction1,
                        bias_correction2, weight_decay_, decouple_weight_decay_);
      }
#endif
      else {
        throw std::runtime_error("Unsupported device type for Adam optimizer");
      }
    }
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
