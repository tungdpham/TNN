/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/activations_impl/relu.hpp"
#include "ops/ops.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"
#include <cassert>
#include <memory>
#include <string>

namespace tnn {
template <typename T> ReLU<T>::ReLU(T negative_slope) : negative_slope_(negative_slope) {}

template <typename T> void ReLU<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  const size_t size = tensor.size();

  if (negative_slope_ == T(0)) {

    const size_t num_threads = get_num_threads();
    const size_t block_size = size / num_threads;
    parallel_for<size_t>(0, num_threads, [&](size_t i) {
      size_t start = i * block_size;
      size_t end = std::min(start + block_size, size);
      ops::cpu::scalar_max(data + start, T(0), data + start, end - start);
    });
  } else {
    parallel_for<size_t>(
        0, size, [&](size_t i) { data[i] = data[i] > T(0) ? data[i] : negative_slope_ * data[i]; });
  }
}

template <typename T>
void ReLU<T>::compute_gradient_inplace(const Tensor<T> &input, Tensor<T> &upstream_gradient) const {
  assert(input.shape() == upstream_gradient.shape() &&
         "Shapes must match for in-place gradient computation");

  const T *input_data = input.data();
  T *grad_data = upstream_gradient.data();
  const size_t size = input.size();

  parallel_for<size_t>(0, size, [&](size_t i) {
    T local_grad = input_data[i] > T(0) ? T(1) : negative_slope_;
    grad_data[i] *= local_grad;
  });
}

template <typename T> std::string ReLU<T>::name() const {
  return negative_slope_ == T(0) ? "relu" : "leaky_relu";
}

template <typename T> std::unique_ptr<ActivationFunction<T>> ReLU<T>::clone() const {
  return std::make_unique<ReLU<T>>(*this);
}

// Explicit template instantiations
template class ReLU<float>;
template class ReLU<double>;

} // namespace tnn