/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/activations_impl/linear.hpp"
#include "nn/activations_impl/base_activation.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tnn {
template <typename T> void Linear<T>::apply(Tensor<T> &tensor) const { (void)tensor; }

template <typename T>
void Linear<T>::apply_with_bias(Tensor<T> &tensor, const Tensor<T> &bias) const {
  if (tensor.shape() != bias.shape()) {
    throw std::invalid_argument("Tensor and bias must have the same shape");
  }

  T *data = tensor.data();
  const T *bias_data = bias.data();
  const size_t size = tensor.size();

  tthreads::parallel_for<size_t>(0, size, [&](size_t i) { data[i] += bias_data[i]; });
}

template <typename T> void Linear<T>::apply_with_scalar_bias(Tensor<T> &tensor, T bias) const {
  if (bias != T(0)) {
    T *data = tensor.data();
    size_t size = tensor.size();

    tthreads::parallel_for<size_t>(0, size, [&](size_t i) { data[i] += bias; });
  }
}

template <typename T>
Tensor<T> Linear<T>::compute_gradient(const Tensor<T> &pre_activation_values,
                                      const Tensor<T> *upstream_gradient) const {
  if (upstream_gradient == nullptr) {
    throw std::runtime_error("Upstream gradient must be provided for Linear activation");
  }
  return upstream_gradient->clone();
}

template <typename T>
void Linear<T>::compute_gradient_inplace(const Tensor<T> &pre_activation_values,
                                         Tensor<T> &upstream_gradient) const {
  if (upstream_gradient.shape() != pre_activation_values.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }
}

template <typename T> void Linear<T>::apply_channel_wise(Tensor<T> &tensor, int channel) const {
  (void)tensor;
  (void)channel;
}

template <typename T>
void Linear<T>::apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                                             const std::vector<T> &bias) const {
  if (channel < 0 || channel >= static_cast<int>(tensor.channels())) {
    throw std::invalid_argument("Channel index out of bounds");
  }

  size_t batch_size = tensor.batch_size();
  size_t height = tensor.height();
  size_t width = tensor.width();
  size_t spatial_size = height * width;

  if (bias.size() != spatial_size) {
    throw std::invalid_argument("Bias size must match spatial dimensions");
  }

  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        tensor(n, channel, h, w) += bias[h * width + w];
      }
    }
  }
}

template <typename T> void Linear<T>::apply_batch_wise(Tensor<T> &tensor, int batch_idx) const {
  (void)tensor;
  (void)batch_idx;
}

template <typename T> std::string Linear<T>::name() const { return "linear"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> Linear<T>::clone() const {
  return std::make_unique<Linear<T>>(*this);
}

template <typename T> void Linear<T>::apply_single_value(T &value) const { (void)value; }

template <typename T> T Linear<T>::compute_single_gradient(T pre_activation_value) const {
  (void)pre_activation_value;
  return T(1);
}

// Explicit template instantiations
template class Linear<float>;
template class Linear<double>;

} // namespace tnn