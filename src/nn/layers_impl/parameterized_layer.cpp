/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/parameterized_layer.hpp"

namespace tnn {

void ParameterizedLayer::init_impl() { init_params(); }

std::vector<Tensor> ParameterizedLayer::parameters() {
  std::vector<Tensor> params;
  collect_parameters(params);
  return params;
}

std::vector<Tensor> ParameterizedLayer::gradients() {
  std::vector<Tensor> grads;
  collect_gradients(grads);
  return grads;
}

} // namespace tnn
