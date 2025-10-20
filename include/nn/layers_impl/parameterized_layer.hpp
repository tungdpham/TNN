/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "../optimizers.hpp"
#include "base_layer.hpp"
#include "tensor/tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

template <typename T = float> class ParameterizedLayer : public Layer<T> {
public:
  explicit ParameterizedLayer(const std::string &name = "") { this->name_ = name; }

  void initialize() override;
  std::vector<Tensor<T> *> parameters() override;
  std::vector<Tensor<T> *> gradients() override;
  bool has_parameters() const override { return true; }
  virtual void clear_gradients() = 0;

protected:
  virtual void initialize_params() = 0;
  virtual void collect_parameters(std::vector<Tensor<T> *> &params) = 0;
  virtual void collect_gradients(std::vector<Tensor<T> *> &grads) = 0;
  bool initialized_ = false;
};
} // namespace tnn