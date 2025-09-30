/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "tensor/tensor.hpp"
#include "utils/parallel_for.hpp"

namespace tnn {

#include "activations_impl/base_activation.hpp"
#include "activations_impl/elu.hpp"
#include "activations_impl/leaky_relu.hpp"
#include "activations_impl/linear.hpp"
#include "activations_impl/relu.hpp"
#include "activations_impl/sigmoid.hpp"
#include "activations_impl/softmax.hpp"
#include "activations_impl/tanh.hpp"

template <typename T = float> class ActivationFactory {
private:
  static std::unordered_map<std::string, std::function<std::unique_ptr<ActivationFunction<T>>()>>
      creators_;

public:
  static void register_activation(const std::string &name,
                                  std::function<std::unique_ptr<ActivationFunction<T>>()> creator) {
    creators_[name] = creator;
  }

  static std::unique_ptr<ActivationFunction<T>> create(const std::string &name) {
    auto it = creators_.find(name);
    if (it != creators_.end()) {
      return it->second();
    }
    throw std::invalid_argument("Unknown activation function: " + name);
  }

  static void register_defaults() {
    register_activation("none", []() { return nullptr; });
    register_activation("relu", []() { return std::make_unique<ReLU<T>>(); });
    register_activation("leaky_relu", []() { return std::make_unique<LeakyReLU<T>>(T(0.01)); });
    register_activation("sigmoid", []() { return std::make_unique<Sigmoid<T>>(); });
    register_activation("softmax", []() { return std::make_unique<Softmax<T>>(); });
    register_activation("linear", []() { return std::make_unique<Linear<T>>(); });
    register_activation("tanh", []() { return std::make_unique<Tanh<T>>(); });
    register_activation("elu", []() { return std::make_unique<ELU<T>>(); });
  }

  static std::vector<std::string> get_available_activations() {
    std::vector<std::string> names;
    for (const auto &pair : creators_) {
      names.push_back(pair.first);
    }
    return names;
  }
};

template <typename T>
std::unordered_map<std::string, std::function<std::unique_ptr<ActivationFunction<T>>()>>
    ActivationFactory<T>::creators_;

} // namespace tnn