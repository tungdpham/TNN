/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <string>
#include <vector>

#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class ParameterizedLayer : public Layer {
public:
  explicit ParameterizedLayer(const std::string &name = "") { this->name_ = name; }

  std::vector<Tensor> parameters() override;
  std::vector<Tensor> gradients() override;
  bool has_parameters() const override { return true; }

private:
  void init_impl() override;

protected:
  virtual void init_params() = 0;
  virtual void collect_parameters(std::vector<Tensor> &params) = 0;
  virtual void collect_gradients(std::vector<Tensor> &grads) = 0;
};
}  // namespace tnn
