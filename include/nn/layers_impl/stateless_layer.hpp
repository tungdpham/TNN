/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include "base_layer.hpp"

#include <string>
#include <vector>

namespace tnn {

template <typename T = float> class StatelessLayer : public Layer<T> {
public:
  explicit StatelessLayer(const std::string &name = "") { this->name_ = name; }
  std::vector<Tensor<T> *> parameters() override { return {}; }
  std::vector<Tensor<T> *> gradients() override { return {}; }
  bool has_parameters() const override { return false; }
};

} // namespace tnn