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

  tthreads::parallel_for<size_t>(
      0, size, [&](size_t i) { data[i] = data[i] > T(0) ? data[i] : negative_slope_ * data[i]; });
}

template <typename T>
void LeakyReLU<T>::apply_with_bias(Tensor<T> &tensor, const Tensor<T> &bias) const {
  if (tensor.shape() != bias.shape()) {
    throw std::invalid_argument("Tensor and bias must have the same shape");
  }

  T *data = tensor.data();
  const T *bias_data = bias.data();
  size_t size = tensor.size();

  tthreads::parallel_for<size_t>(0, size, [&](size_t i) {
    T val = data[i] + bias_data[i];
    data[i] = val > T(0) ? val : negative_slope_ * val;
  });
}

template <typename T> void LeakyReLU<T>::apply_with_scalar_bias(Tensor<T> &tensor, T bias) const {
  T *data = tensor.data();
  size_t size = tensor.size();

  tthreads::parallel_for<size_t>(0, size, [&](size_t i) {
    T val = data[i] + bias;
    data[i] = val > T(0) ? val : negative_slope_ * val;
  });
}

template <typename T>
Tensor<T> LeakyReLU<T>::compute_gradient(const Tensor<T> &input,
                                         const Tensor<T> *upstream_gradient) const {
  Tensor<T> gradient;
  if (upstream_gradient != nullptr) {
    gradient = upstream_gradient->clone();
  } else {
    gradient = Tensor<T>(input.shape());
    gradient.fill(T(1));
  }
  compute_gradient_inplace(input, gradient);
  return gradient;
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

  tthreads::parallel_for<size_t>(0, size, [&](size_t i) {
    T local_grad = input_data[i] > T(0) ? T(1) : negative_slope_;
    grad_data[i] *= local_grad;
  });
}

template <typename T> void LeakyReLU<T>::apply_channel_wise(Tensor<T> &tensor, int channel) const {
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
    val = val > T(0) ? val : negative_slope_ * val;
  });
}

template <typename T>
void LeakyReLU<T>::apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
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
    tensor(n, channel, h, w) = val > T(0) ? val : negative_slope_ * val;
  });
}

template <typename T> void LeakyReLU<T>::apply_batch_wise(Tensor<T> &tensor, int batch_idx) const {
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
    val = val > T(0) ? val : negative_slope_ * val;
  });
}

template <typename T> std::string LeakyReLU<T>::name() const { return "leaky_relu"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> LeakyReLU<T>::clone() const {
  return std::make_unique<LeakyReLU<T>>(*this);
}

template <typename T> void LeakyReLU<T>::apply_single_value(T &value) const {
  value = value > T(0) ? value : negative_slope_ * value;
}

template <typename T> T LeakyReLU<T>::compute_single_gradient(T pre_activation_value) const {
  return pre_activation_value > T(0) ? T(1) : negative_slope_;
}

template class LeakyReLU<float>;
template class LeakyReLU<double>;

} // namespace tnn