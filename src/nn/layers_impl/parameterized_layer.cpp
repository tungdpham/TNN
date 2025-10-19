/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/parameterized_layer.hpp"

namespace tnn {

template <typename T> void ParameterizedLayer<T>::initialize() {
  if (initialized_) {
    return;
  }
  initialize_params();
  initialized_ = true;
}

template <typename T> std::vector<Tensor<T> *> ParameterizedLayer<T>::parameters() {
  std::vector<Tensor<T> *> params;
  collect_parameters(params);
  return params;
}

template <typename T> std::vector<Tensor<T> *> ParameterizedLayer<T>::gradients() {
  std::vector<Tensor<T> *> grads;
  collect_gradients(grads);
  return grads;
}

template <typename T> void ParameterizedLayer<T>::update_parameters() {
  if (!layer_optimizer_) {
    throw std::runtime_error("No optimizer set for layer: " + this->name_);
  }
  auto params = parameters();
  auto grads = gradients();
  layer_optimizer_->update(params, grads);
  clear_gradients();
}

template <typename T>
void ParameterizedLayer<T>::set_optimizer(std::unique_ptr<Optimizer<T>> optimizer) {
  layer_optimizer_ = std::move(optimizer);
}

// Explicit template instantiations
template class ParameterizedLayer<float>;
template class ParameterizedLayer<double>;

} // namespace tnn
