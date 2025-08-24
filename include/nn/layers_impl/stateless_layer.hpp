
#pragma once

#include "base_layer.hpp"

#include <string>
#include <vector>

namespace tnn {

// Base class for layers without parameters (activation, pooling, etc.)
template <typename T = float> class StatelessLayer : public Layer<T> {
public:
  explicit StatelessLayer(const std::string &name = "") { this->name_ = name; }
  std::vector<Tensor<T> *> parameters() override { return {}; }
  std::vector<Tensor<T> *> gradients() override { return {}; }
  bool has_parameters() const override { return false; }
};

} // namespace tnn