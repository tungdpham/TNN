/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/activations_impl/base_activation.hpp"
#include "nn/activations_impl/tanh.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace tnn {

template <typename T> void Tanh<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  size_t size = tensor.size();

  parallel_for<size_t>(0, size, [&](size_t i) { data[i] = std::tanh(data[i]); });
}

template <typename T>
void Tanh<T>::compute_gradient_inplace(const Tensor<T> &input, Tensor<T> &upstream_gradient) const {
  if (upstream_gradient.shape() != input.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }

  const T *input_data = input.data();
  T *grad_data = upstream_gradient.data();
  size_t size = input.size();

  parallel_for<size_t>(0, size, [&](size_t i) {
    T tanh_val = std::tanh(input_data[i]);
    T local_grad = T(1) - tanh_val * tanh_val;
    grad_data[i] *= local_grad;
  });
}

template <typename T> std::string Tanh<T>::name() const { return "tanh"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> Tanh<T>::clone() const {
  return std::make_unique<Tanh<T>>(*this);
}

// Explicit template instantiations
template class Tanh<float>;
template class Tanh<double>;

} // namespace tnn