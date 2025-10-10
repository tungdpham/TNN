/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/activations_impl/elu.hpp"
#include "nn/activations_impl/base_activation.hpp"
#include "threading/thread_handler.hpp"

namespace tnn {
template <typename T> ELU<T>::ELU(T alpha) : alpha_(alpha) {}

template <typename T> void ELU<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  const size_t size = tensor.size();

  tthreads::parallel_for<size_t>(0, size, [&](size_t i) {
    data[i] = data[i] > T(0) ? data[i] : alpha_ * (std::exp(data[i]) - T(1));
  });
}

template <typename T> void ELU<T>::apply_with_bias(Tensor<T> &tensor, const Tensor<T> &bias) const {
  if (tensor.shape() != bias.shape()) {
    throw std::invalid_argument("Tensor and bias must have the same shape");
  }

  T *data = tensor.data();
  const T *bias_data = bias.data();
  size_t size = tensor.size();

  tthreads::parallel_for<size_t>(0, size, [&](size_t i) {
    T val = data[i] + bias_data[i];
    data[i] = val > T(0) ? val : alpha_ * (std::exp(val) - T(1));
  });
}

template <typename T> void ELU<T>::apply_with_scalar_bias(Tensor<T> &tensor, T bias) const {
  T *data = tensor.data();
  size_t size = tensor.size();

  tthreads::parallel_for<size_t>(0, size, [&](size_t i) {
    T val = data[i] + bias;
    data[i] = val > T(0) ? val : alpha_ * (std::exp(val) - T(1));
  });
}

template <typename T>
Tensor<T> ELU<T>::compute_gradient(const Tensor<T> &pre_activation_values,
                                   const Tensor<T> *upstream_gradient) const {
  Tensor<T> gradient;
  if (upstream_gradient != nullptr) {
    gradient = upstream_gradient->clone();
  } else {
    gradient = Tensor<T>(pre_activation_values.shape());
    gradient.fill(T(1));
  }
  compute_gradient_inplace(pre_activation_values, gradient);
  return gradient;
}

template <typename T>
void ELU<T>::compute_gradient_inplace(const Tensor<T> &pre_activation_values,
                                      Tensor<T> &upstream_gradient) const {
  if (upstream_gradient.shape() != pre_activation_values.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }

  const T *input_data = pre_activation_values.data();
  T *grad_data = upstream_gradient.data();
  size_t size = pre_activation_values.size();

  tthreads::parallel_for<size_t>(0, size, [&](size_t i) {
    T local_grad = input_data[i] > T(0) ? T(1) : alpha_ * std::exp(input_data[i]);
    grad_data[i] *= local_grad;
  });
}

template <typename T> void ELU<T>::apply_channel_wise(Tensor<T> &tensor, int channel) const {
  if (channel < 0 || channel >= static_cast<int>(tensor.channels())) {
    throw std::invalid_argument("Channel index out of bounds");
  }

  size_t batch_size = tensor.batch_size();
  size_t height = tensor.height();
  size_t width = tensor.width();

  const size_t total = batch_size * height * width;
  tthreads::parallel_for<size_t>(0, total, [&](size_t idx) {
    size_t n = idx / (height * width);
    size_t rem = idx % (height * width);
    size_t h = rem / width;
    size_t w = rem % width;
    T &val = tensor(n, channel, h, w);
    val = val > T(0) ? val : alpha_ * (std::exp(val) - T(1));
  });
}

template <typename T>
void ELU<T>::apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
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

  const size_t total = batch_size * height * width;
  tthreads::parallel_for<size_t>(0, total, [&](size_t idx) {
    size_t n = idx / (height * width);
    size_t rem = idx % (height * width);
    size_t h = rem / width;
    size_t w = rem % width;
    T val = tensor(n, channel, h, w) + bias[h * width + w];
    tensor(n, channel, h, w) = val > T(0) ? val : alpha_ * (std::exp(val) - T(1));
  });
}

template <typename T> void ELU<T>::apply_batch_wise(Tensor<T> &tensor, int batch_idx) const {
  if (batch_idx < 0 || batch_idx >= static_cast<int>(tensor.batch_size())) {
    throw std::invalid_argument("Batch index out of bounds");
  }

  size_t channels = tensor.channels();
  size_t height = tensor.height();
  size_t width = tensor.width();

  const size_t total = channels * height * width;
  tthreads::parallel_for<size_t>(0, total, [&](size_t idx) {
    size_t c = idx / (height * width);
    size_t rem = idx % (height * width);
    size_t h = rem / width;
    size_t w = rem % width;
    T &val = tensor(batch_idx, c, h, w);
    val = val > T(0) ? val : alpha_ * (std::exp(val) - T(1));
  });
}

template <typename T> std::string ELU<T>::name() const { return "elu"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> ELU<T>::clone() const {
  return std::make_unique<ELU<T>>(*this);
}

template <typename T> void ELU<T>::apply_single_value(T &value) const {
  value = value > T(0) ? value : alpha_ * (std::exp(value) - T(1));
}

template <typename T> T ELU<T>::compute_single_gradient(T pre_activation_value) const {
  return pre_activation_value > T(0) ? T(1) : alpha_ * std::exp(pre_activation_value);
}

// Explicit template instantiations
template class ELU<float>;
template class ELU<double>;

} // namespace tnn