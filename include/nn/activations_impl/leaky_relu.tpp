/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/activations_impl/leaky_relu.hpp"

namespace tnn {
template <typename T> LeakyReLU<T>::LeakyReLU(T negative_slope) : negative_slope_(negative_slope) {}

template <typename T> void LeakyReLU<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  const size_t size = tensor.size();

  parallel_for<size_t>(
      0, size, [&](size_t i) { data[i] = data[i] > T(0) ? data[i] : negative_slope_ * data[i]; });
}

template <typename T>
void LeakyReLU<T>::compute_gradient_inplace(const Tensor<T> &input,
                                            Tensor<T> &upstream_gradient) const {
  if (upstream_gradient.shape() != input.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }

  const T *input_data = input.data();
  T *grad_data = upstream_gradient.data();
  size_t size = input.size();

  parallel_for<size_t>(0, size, [&](size_t i) {
    T local_grad = input_data[i] > T(0) ? T(1) : negative_slope_;
    grad_data[i] *= local_grad;
  });
}

template <typename T> std::string LeakyReLU<T>::name() const { return "leaky_relu"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> LeakyReLU<T>::clone() const {
  return std::make_unique<LeakyReLU<T>>(*this);
}

template class LeakyReLU<float>;
template class LeakyReLU<double>;

} // namespace tnn