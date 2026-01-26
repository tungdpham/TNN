/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#include "nlohmann/json.hpp"
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

  nlohmann::json to_json() const {
    nlohmann::json j;
    j["type"] = type;
    j["name"] = name;
    j["parameters"] = nlohmann::json::object();
    for (const auto &[key, value] : parameters) {
      try {
        if (auto *int_ptr = std::any_cast<int>(&value)) {
          j["parameters"][key] = *int_ptr;
        } else if (auto *size_ptr = std::any_cast<size_t>(&value)) {
          j["parameters"][key] = *size_ptr;
        } else if (auto *float_ptr = std::any_cast<float>(&value)) {
          j["parameters"][key] = *float_ptr;
        } else if (auto *double_ptr = std::any_cast<double>(&value)) {
          j["parameters"][key] = *double_ptr;
        } else if (auto *bool_ptr = std::any_cast<bool>(&value)) {
          j["parameters"][key] = *bool_ptr;
        } else if (auto *string_ptr = std::any_cast<std::string>(&value)) {
          j["parameters"][key] = *string_ptr;
        }
      } catch (const std::bad_any_cast &) {
      }
    }
    return j;
  }

  static OptimizerConfig from_json(const nlohmann::json &j) {
    OptimizerConfig config;
    config.type = j.value("type", "");
    config.name = j.value("name", "");
    if (j.contains("parameters")) {
      for (const auto &[key, value] : j["parameters"].items()) {
        if (value.is_number_integer()) {
          config.parameters[key] = value.template get<size_t>();
        } else if (value.is_number_float()) {
          config.parameters[key] = value.template get<float>();
        } else if (value.is_boolean()) {
          config.parameters[key] = value.template get<bool>();
        } else if (value.is_string()) {
          config.parameters[key] = value.template get<std::string>();
        }
      }
    }
    return config;
  }
};

class Optimizer {
public:
  explicit Optimizer(float learning_rate) : learning_rate_(learning_rate) {}
  virtual ~Optimizer() = default;

  void attach(std::vector<Tensor> params, const std::vector<Tensor> grads) {
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

  void clear_gradients() {
    for (auto &grad : gradients_) {
      grad->fill(0.0);
    }
  }

  void set_learning_rate(float lr) { learning_rate_ = lr; }
  float get_learning_rate() const { return learning_rate_; }

  virtual std::string name() const = 0;
  virtual OptimizerConfig get_config() const = 0;
  virtual std::unique_ptr<Optimizer> clone() const = 0;

protected:
  float learning_rate_;
  std::vector<Tensor> parameters_;
  std::vector<Tensor> gradients_;

  virtual void on_attach() {}
};

class SGD : public Optimizer {
public:
  SGD(float learning_rate = 0.01f, float momentum = 0.0f)
      : Optimizer(learning_rate), momentum_(momentum) {}

  void update() override {
    auto &params = this->parameters_;
    auto &grads = this->gradients_;

    for (size_t i = 0; i < params.size(); ++i) {
      DISPATCH_ON_DTYPE(params[i]->data_type(), T,
                        update_impl<T>(params[i], grads[i], velocities_[i]));
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

  std::unique_ptr<Optimizer> clone() const override {
    return std::make_unique<SGD>(this->learning_rate_, momentum_);
  }

protected:
  void on_attach() override {
    if (momentum_ > 0.0f) {
      velocities_.resize(this->parameters_.size());
      for (size_t i = 0; i < this->parameters_.size(); ++i) {
        velocities_[i] =
            Tensor::create(this->parameters_[i]->data_type(), this->parameters_[i]->shape(),
                           this->parameters_[i]->device());
        velocities_[i]->fill(0.0f);
      }
    }
  }

private:
  float momentum_;
  std::vector<Tensor> velocities_;

  template <typename T> void update_impl(Tensor &param, Tensor &grad, Tensor &velocity) {
    const size_t size = param->size();

    if (param->device_type() == DeviceType::CPU) {
      if (momentum_ > 0.0f) {
        create_cpu_task("default", cpu::sgd::update_sgd_momentum<T>, param->data_as<T>(),
                        grad->data_as<T>(), velocity->data_as<T>(), size, this->learning_rate_,
                        momentum_);
      } else {
        create_cpu_task("default", cpu::sgd::update_sgd<T>, param->data_as<T>(), grad->data_as<T>(),
                        size, this->learning_rate_);
      }
    }
#ifdef USE_CUDA
    else if (param->device_type() == DeviceType::GPU) {
      if (momentum_ > 0.0f) {
        create_cuda_task("default", cuda::sgd::update_sgd_momentum<T>, param->data_as<T>(),
                         grad->data_as<T>(), velocity->data_as<T>(), size, this->learning_rate_,
                         momentum_);
      } else {
        create_cuda_task("default", cuda::sgd::update_sgd<T>, param->data_as<T>(),
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
      : Optimizer(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
        weight_decay_(weight_decay), decouple_weight_decay_(decouple_weight_decay), t_(0) {}

  void update() override {
    auto &params = this->parameters_;
    auto &grads = this->gradients_;

    t_++;

    // Precompute bias correction terms outside the loop
    const float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    const float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params.size(); ++i) {
      DISPATCH_ON_DTYPE(
          params[i]->data_type(), T,
          update_impl<T>(params[i], grads[i], m_[i], v_[i], bias_correction1, bias_correction2));
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

  std::unique_ptr<Optimizer> clone() const override {
    return std::make_unique<Adam>(this->learning_rate_, beta1_, beta2_, epsilon_, weight_decay_,
                                  decouple_weight_decay_);
  }

protected:
  void on_attach() override {
    m_.resize(this->parameters_.size());
    v_.resize(this->parameters_.size());
    for (size_t i = 0; i < this->parameters_.size(); ++i) {
      m_[i] = Tensor::create(this->parameters_[i]->data_type(), this->parameters_[i]->shape(),
                             this->parameters_[i]->device());
      m_[i]->fill(0.0f);
      v_[i] = Tensor::create(this->parameters_[i]->data_type(), this->parameters_[i]->shape(),
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
  std::vector<Tensor> m_;
  std::vector<Tensor> v_;

  template <typename T>
  void update_impl(Tensor &param, Tensor &grad, Tensor &m, Tensor &v, float bias_correction1,
                   float bias_correction2) {
    const size_t size = param->size();
    if (param->device_type() == DeviceType::CPU) {
      create_cpu_task("default", cpu::adam::update_adam<T>, param->data_as<T>(), grad->data_as<T>(),
                      m->data_as<T>(), v->data_as<T>(), size, this->learning_rate_, beta1_, beta2_,
                      epsilon_, bias_correction1, bias_correction2, weight_decay_,
                      decouple_weight_decay_);
    }
#ifdef USE_CUDA
    else if (param->device_type() == DeviceType::GPU) {
      create_cuda_task("default", cuda::adam::update_adam<T>, param->data_as<T>(),
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

} // namespace tnn
