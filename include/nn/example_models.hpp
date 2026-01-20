/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/base_layer.hpp"
#include "nn/sequential.hpp"
#include <memory>

namespace tnn {

class ExampleModels {
private:
  static std::unordered_map<std::string, std::function<std::unique_ptr<Layer>(DType_t)>> creators_;

public:
  static void register_model(const Sequential &model) {
    auto model_clone = std::shared_ptr<Layer>(model.clone());
    std::string model_name = model.name();
    creators_[model_name] = [model_ptr = std::move(model_clone)](DType_t) {
      return model_ptr->clone();
    };
  }

  static std::unique_ptr<Layer> create(const std::string &name, DType_t io_dtype_ = DType_t::FP32) {
    auto it = creators_.find(name);
    if (it != creators_.end()) {
      return it->second(io_dtype_);
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