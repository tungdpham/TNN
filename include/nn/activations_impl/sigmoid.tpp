/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/activations_impl/base_activation.hpp"
#include "nn/activations_impl/sigmoid.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tnn {
template <typename T> void Sigmoid<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  size_t size = tensor.size();

  parallel_for<size_t>(0, size, [&](size_t i) { data[i] = T(1) / (T(1) + std::exp(-data[i])); });
}

template <typename T>
void Sigmoid<T>::compute_gradient_inplace(const Tensor<T> &input,
                                          Tensor<T> &upstream_gradient) const {
  if (upstream_gradient.shape() != input.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }

  const T *input_data = input.data();
  T *grad_data = upstream_gradient.data();
  size_t size = input.size();

  parallel_for<size_t>(0, size, [&](size_t i) {
    T sigmoid_val = T(1) / (T(1) + std::exp(-input_data[i]));
    T local_grad = sigmoid_val * (T(1) - sigmoid_val);
    grad_data[i] *= local_grad;
  });
}

template <typename T> std::string Sigmoid<T>::name() const { return "sigmoid"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> Sigmoid<T>::clone() const {
  return std::make_unique<Sigmoid<T>>(*this);
}

// Explicit template instantiations
template class Sigmoid<float>;
template class Sigmoid<double>;

} // namespace tnn