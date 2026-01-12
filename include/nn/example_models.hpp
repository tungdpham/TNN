/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/sequential.hpp"

namespace tnn {

template <typename T = float> class ExampleModels {
private:
  static std::unordered_map<std::string, Sequential<T>> creators_;

public:
  static void register_model(const Sequential<T> &model) { creators_[model.name()] = model; }

  static Sequential<T> create(const std::string &name) {
    auto it = creators_.find(name);
    if (it != creators_.end()) {
      return it->second;
    }
    throw std::invalid_argument("Unknown model: " + name);
  }

  static std::vector<std::string> available_models() {
    std::vector<std::string> models;
    for (const auto &pair : creators_) {
      models.push_back(pair.first);
    }
    return models;
  }

  static void register_defaults();
};

} // namespace tnn