/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "common/config.hpp"
#include "device/task.hpp"
#include "nn/graph_context.hpp"
#include "optimizers_impl/cpu/adam_kernels.hpp"
#include "optimizers_impl/cpu/sgd_kernels.hpp"
#include "optimizers_impl/cuda/adam_kernels.hpp"
#include "optimizers_impl/cuda/sgd_kernels.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

using OptimizerConfig = TConfig;

class Optimizer {
public:
  explicit Optimizer(float learning_rate)
      : learning_rate_(learning_rate) {}
  virtual ~Optimizer() = default;

  void attach(GraphContext &context) {
    context_ = &context;
    auto params = context.parameters();
    auto grads = context.gradients();
    if (params.size() != grads.size()) {
      throw std::invalid_argument("Parameters and gradients size mismatch in optimizer attach" +
                                  std::to_string(params.size()) + " vs " +
                                  std::to_string(grads.size()));
    }
    parameters_ = params;
    gradients_ = grads;
    on_attach();
    std::cout << "Optimizer attached to " << parameters_.size() << " parameters and "
              << gradients_.size() << " gradients." << std::endl;
  }

  virtual void update() = 0;

  void zero_grads() { context_->zero_grads(); }

  void set_learning_rate(float lr) { learning_rate_ = lr; }
  float get_learning_rate() const { return learning_rate_; }

  virtual std::string name() const = 0;
  virtual OptimizerConfig get_config() const = 0;
  virtual std::unique_ptr<Optimizer> clone() const = 0;

  size_t debug_num_parameters() const { return parameters_.size(); }

  // Expensive debug helper: copies up to max_tensors and max_items_per_tensor to host.
  // Use only for short diagnostic runs.
  double debug_abs_mean_grad(size_t max_tensors = 2, size_t max_items_per_tensor = 4096) const {
    return debug_abs_mean_tensors(gradients_, max_tensors, max_items_per_tensor);
  }

  double debug_abs_mean_param(size_t max_tensors = 2, size_t max_items_per_tensor = 4096) const {
    return debug_abs_mean_tensors(parameters_, max_tensors, max_items_per_tensor);
  }

protected:
  template <typename T>
  static double debug_abs_mean_tensor_typed(const Tensor &tensor, size_t max_items_per_tensor) {
    if (!tensor) return 0.0;
    Tensor h = tensor->to_host();
    const size_t n = std::min(h->size(), max_items_per_tensor);
    if (n == 0) return 0.0;
    const T *ptr = h->data_as<T>();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, fp16>) {
        sum += std::abs(static_cast<float>(ptr[i]));
      } else {
        sum += std::abs(static_cast<double>(ptr[i]));
      }
    }
    return sum / static_cast<double>(n);
  }

  static double debug_abs_mean_tensors(const Vec<Tensor> &tensors, size_t max_tensors,
                                       size_t max_items_per_tensor) {
    const size_t nt = std::min(tensors.size(), max_tensors);
    if (nt == 0) return 0.0;
    double sum = 0.0;
    size_t used = 0;
    for (size_t i = 0; i < nt; ++i) {
      if (!tensors[i]) continue;
      double v = 0.0;
      switch (tensors[i]->data_type()) {
        case DType_t::FP16:
          v = debug_abs_mean_tensor_typed<fp16>(tensors[i], max_items_per_tensor);
          break;
        case DType_t::BF16:
          v = debug_abs_mean_tensor_typed<bf16>(tensors[i], max_items_per_tensor);
          break;
        case DType_t::FP32:
          v = debug_abs_mean_tensor_typed<float>(tensors[i], max_items_per_tensor);
          break;
        case DType_t::FP64:
          v = debug_abs_mean_tensor_typed<double>(tensors[i], max_items_per_tensor);
          break;
        default:
          continue;
      }
      sum += v;
      used++;
    }
    return used > 0 ? sum / static_cast<double>(used) : 0.0;
  }

protected:
  float learning_rate_;
  GraphContext *context_;
  Vec<Tensor> parameters_;
  Vec<Tensor> gradients_;

  virtual void on_attach() {}
};

class SGD : public Optimizer {
public:
  SGD(float learning_rate = 0.01f, float momentum = 0.0f)
      : Optimizer(learning_rate),
        momentum_(momentum) {}

  void update() override {
    auto &params = this->parameters_;
    auto &grads = this->gradients_;

    for (size_t i = 0; i < params.size(); ++i) {
      DISPATCH_DTYPE(params[i]->data_type(), T,
                     update_impl<T>(params[i], grads[i], velocities_[i]));
    }
  }

  std::string name() const override { return "SGD"; }

  OptimizerConfig get_config() const override {
    OptimizerConfig config;
    config.type = "sgd";
    config.name = "SGD";
    config.set("learning_rate", this->learning_rate_);
    config.set("momentum", momentum_);
    return config;
  }

  std::unique_ptr<Optimizer> clone() const override {
    return std::make_unique<SGD>(this->learning_rate_, momentum_);
  }

protected:
  void on_attach() override {
    if (momentum_ > 0.0f) {
      velocities_.resize(this->parameters_.size());
      for (size_t i = 0; i < this->parameters_.size(); ++i) {
        velocities_[i] = make_tensor(this->parameters_[i]->data_type(),
                                     this->parameters_[i]->shape(), this->parameters_[i]->device());
        velocities_[i]->fill(0.0f);
      }
    }
  }

private:
  float momentum_;
  Vec<Tensor> velocities_;

  template <typename T>
  void update_impl(const Tensor &param, const Tensor &grad, const Tensor &velocity) {
    const size_t size = param->size();

    if (param->device_type() == DeviceType::CPU) {
      if (momentum_ > 0.0f) {
        create_cpu_task(defaultFlowHandle, cpu::sgd::update_sgd_momentum<T>, param->data_as<T>(),
                        grad->data_as<T>(), velocity->data_as<T>(), size, this->learning_rate_,
                        momentum_);
      } else {
        create_cpu_task(defaultFlowHandle, cpu::sgd::update_sgd<T>, param->data_as<T>(),
                        grad->data_as<T>(), size, this->learning_rate_);
      }
    }
#ifdef USE_CUDA
    else if (param->device_type() == DeviceType::GPU) {
      if (momentum_ > 0.0f) {
        create_cuda_task(defaultFlowHandle, cuda::sgd::update_sgd_momentum<T>, param->data_as<T>(),
                         grad->data_as<T>(), velocity->data_as<T>(), size, this->learning_rate_,
                         momentum_);
      } else {
        create_cuda_task(defaultFlowHandle, cuda::sgd::update_sgd<T>, param->data_as<T>(),
                         grad->data_as<T>(), size, this->learning_rate_);
      }
    }
#endif
    else {
      throw std::runtime_error("Unsupported device type for SGD optimizer");
    }
  }
};

class Adam : public Optimizer {
public:
  Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
       float epsilon = 1e-8f, float weight_decay = 0.0f, bool decouple_weight_decay = false)
      : Optimizer(learning_rate),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        weight_decay_(weight_decay),
        decouple_weight_decay_(decouple_weight_decay),
        t_(0) {}

  void update() override {
    auto &params = this->parameters_;
    auto &grads = this->gradients_;

    t_++;

    // Precompute bias correction terms outside the loop
    const float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    const float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params.size(); ++i) {
      DISPATCH_DTYPE(
          params[i]->data_type(), T,
          update_impl<T>(params[i], grads[i], m_[i], v_[i], bias_correction1, bias_correction2));
    }
  }

  std::string name() const override { return decouple_weight_decay_ ? "AdamW" : "Adam"; }

  OptimizerConfig get_config() const override {
    OptimizerConfig config;
    config.type = decouple_weight_decay_ ? "adamw" : "adam";
    config.name = decouple_weight_decay_ ? "AdamW" : "Adam";
    config.set("learning_rate", this->learning_rate_);
    config.set("beta1", beta1_);
    config.set("beta2", beta2_);
    config.set("epsilon", epsilon_);
    config.set("weight_decay", weight_decay_);
    config.set("decouple_weight_decay", decouple_weight_decay_);
    return config;
  }

  std::unique_ptr<Optimizer> clone() const override {
    return std::make_unique<Adam>(this->learning_rate_, beta1_, beta2_, epsilon_, weight_decay_,
                                  decouple_weight_decay_);
  }

protected:
  void on_attach() override {
    m_.resize(this->parameters_.size());
    v_.resize(this->parameters_.size());
    for (size_t i = 0; i < this->parameters_.size(); ++i) {
      m_[i] = make_tensor(this->parameters_[i]->data_type(), this->parameters_[i]->shape(),
                          this->parameters_[i]->device());
      m_[i]->fill(0.0f);
      v_[i] = make_tensor(this->parameters_[i]->data_type(), this->parameters_[i]->shape(),
                          this->parameters_[i]->device());
      v_[i]->fill(0.0f);
    }
    t_ = 0;
  }

private:
  float beta1_;
  float beta2_;
  float epsilon_;
  float weight_decay_;
  bool decouple_weight_decay_;
  unsigned long t_;
  Vec<Tensor> m_;
  Vec<Tensor> v_;

  template <typename T>
  void update_impl(const Tensor &param, const Tensor &grad, const Tensor &m, const Tensor &v,
                   float bias_correction1, float bias_correction2) {
    const size_t size = param->size();
    if (param->device_type() == DeviceType::CPU) {
      create_cpu_task(defaultFlowHandle, cpu::adam::update_adam<T>, param->data_as<T>(),
                      grad->data_as<T>(), m->data_as<T>(), v->data_as<T>(), size,
                      this->learning_rate_, beta1_, beta2_, epsilon_, bias_correction1,
                      bias_correction2, weight_decay_, decouple_weight_decay_);
    }
#ifdef USE_CUDA
    else if (param->device_type() == DeviceType::GPU) {
      create_cuda_task(defaultFlowHandle, cuda::adam::update_adam<T>, param->data_as<T>(),
                       grad->data_as<T>(), m->data_as<T>(), v->data_as<T>(), size,
                       this->learning_rate_, beta1_, beta2_, epsilon_, bias_correction1,
                       bias_correction2, weight_decay_, decouple_weight_decay_);
    }
#endif
    else {
      throw std::runtime_error("Unsupported device type for Adam optimizer");
    }
  }
};

class OptimizerFactory {
public:
  static std::unique_ptr<Optimizer> create_from_config(const OptimizerConfig &config) {
    if (config.type == "sgd") {
      float learning_rate = config.get<float>("learning_rate", 0.01f);
      float momentum = config.get<float>("momentum", 0.0f);
      return std::make_unique<SGD>(learning_rate, momentum);
    }
    if (config.type == "adam" || config.type == "adamw") {
      float learning_rate = config.get<float>("learning_rate", 0.001f);
      float beta1 = config.get<float>("beta1", 0.9f);
      float beta2 = config.get<float>("beta2", 0.999f);
      float epsilon = config.get<float>("epsilon", 1e-8f);
      float weight_decay = config.get<float>("weight_decay", 0.0f);
      bool decouple_weight_decay =
          config.get<bool>("decouple_weight_decay", config.type == "adamw");
      return std::make_unique<Adam>(learning_rate, beta1, beta2, epsilon, weight_decay,
                                    decouple_weight_decay);
    }
    throw std::invalid_argument("Unknown optimizer type: " + config.type);
  }

  static std::unique_ptr<Optimizer> create_sgd(float learning_rate = 0.01f, float momentum = 0.0f) {
    return std::make_unique<SGD>(learning_rate, momentum);
  }

  static std::unique_ptr<Optimizer> create_adam(float learning_rate = 0.001f, float beta1 = 0.9f,
                                                float beta2 = 0.999f, float epsilon = 1e-8f,
                                                float weight_decay = 0.0f,
                                                bool decouple_weight_decay = false) {
    return std::make_unique<Adam>(learning_rate, beta1, beta2, epsilon, weight_decay,
                                  decouple_weight_decay);
  }
};

}  // namespace tnn
