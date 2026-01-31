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

#include "activations_impl/base_activation.hpp"
#include "activations_impl/elu.hpp"
#include "activations_impl/gelu.hpp"
#include "activations_impl/leaky_relu.hpp"
#include "activations_impl/linear.hpp"
#include "activations_impl/relu.hpp"
#include "activations_impl/sigmoid.hpp"
#include "activations_impl/tanh.hpp"

namespace tnn {
class ActivationFactory {
private:
  static std::unordered_map<std::string, std::function<std::unique_ptr<ActivationFunction>()>>
      creators_;

public:
  static void register_activation(const std::string &name,
                                  std::function<std::unique_ptr<ActivationFunction>()> creator) {
    creators_[name] = creator;
  }

  static std::unique_ptr<ActivationFunction> create(const std::string &name) {
    auto it = creators_.find(name);
    if (it != creators_.end()) {
      return it->second();
    }
    throw std::invalid_argument("Unknown activation function: " + name);
  }

  static void register_defaults() {
    register_activation("none", []() { return nullptr; });
    register_activation("relu", []() { return std::make_unique<ReLU>(); });
    register_activation("leaky_relu", []() { return std::make_unique<LeakyReLU>(0.01); });
    register_activation("sigmoid", []() { return std::make_unique<Sigmoid>(); });
    register_activation("linear", []() { return std::make_unique<Linear>(); });
    register_activation("tanh", []() { return std::make_unique<Tanh>(); });
    register_activation("elu", []() { return std::make_unique<ELU>(); });
    register_activation("gelu", []() { return std::make_unique<GELU>(); });
  }

  static std::vector<std::string> get_available_activations() {
    std::vector<std::string> names;
    for (const auto &pair : creators_) {
      names.push_back(pair.first);
    }
    return names;
  }
};

inline std::unordered_map<std::string, std::function<std::unique_ptr<ActivationFunction>()>>
    ActivationFactory::creators_;

}  // namespace tnn