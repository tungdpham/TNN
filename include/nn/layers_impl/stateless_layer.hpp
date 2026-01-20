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

class StatelessLayer : public Layer {
public:
  explicit StatelessLayer(const std::string &name = "") { this->name_ = name; }
  std::vector<Tensor> parameters() override { return {}; }
  std::vector<Tensor> gradients() override { return {}; }
  bool has_parameters() const override { return false; }

private:
  // Stateless layers do not need initialization
  void init_impl() override {
    (void)this; // no-op
  }

  // stateless layers don't have params so device change don't really matter but there may be
  // special cases.
  void on_set_device(const Device &device) override {
    (void)device; // no-op
  }
};

} // namespace tnn