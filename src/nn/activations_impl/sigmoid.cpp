/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/activations_impl/sigmoid.hpp"
#include "nn/activations_impl/base_activation.hpp"
#include "tensor/tensor.hpp"
#include "utils/parallel_for.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tnn {
template <typename T> void Sigmoid<T>::apply(Tensor<T> &tensor) const {
  T *data = tensor.data();
  size_t size = tensor.size();

  utils::parallel_for<size_t>(0, size,
                              [&](size_t i) { data[i] = T(1) / (T(1) + std::exp(-data[i])); });
}

template <typename T>
void Sigmoid<T>::apply_with_bias(Tensor<T> &tensor, const Tensor<T> &bias) const {
  if (tensor.shape() != bias.shape()) {
    throw std::invalid_argument("Tensor and bias must have the same shape");
  }

  T *data = tensor.data();
  const T *bias_data = bias.data();
  size_t size = tensor.size();

  utils::parallel_for<size_t>(0, size, [&](size_t i) {
    T val = data[i] + bias_data[i];
    data[i] = T(1) / (T(1) + std::exp(-val));
  });
}

template <typename T> void Sigmoid<T>::apply_with_scalar_bias(Tensor<T> &tensor, T bias) const {
  T *data = tensor.data();
  size_t size = tensor.size();

  utils::parallel_for<size_t>(0, size, [&](size_t i) {
    T val = data[i] + bias;
    data[i] = T(1) / (T(1) + std::exp(-val));
  });
}

template <typename T>
Tensor<T> Sigmoid<T>::compute_gradient(const Tensor<T> &pre_activation_values,
                                       const Tensor<T> *upstream_gradient) const {
  Tensor<T> gradient;
  if (upstream_gradient != nullptr) {
    if (upstream_gradient->shape() != pre_activation_values.shape()) {
      throw std::invalid_argument("Upstream gradient must have the same "
                                  "shape as pre-activation values");
    }
    gradient = upstream_gradient->clone();
  } else {
    gradient = Tensor<T>(pre_activation_values.shape());
    gradient.fill(T(1));
  }
  compute_gradient_inplace(pre_activation_values, gradient);
  return gradient;
}

template <typename T>
void Sigmoid<T>::compute_gradient_inplace(const Tensor<T> &pre_activation_values,
                                          Tensor<T> &upstream_gradient) const {
  if (upstream_gradient.shape() != pre_activation_values.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }

  const T *input_data = pre_activation_values.data();
  T *grad_data = upstream_gradient.data();
  size_t size = pre_activation_values.size();

  utils::parallel_for<size_t>(0, size, [&](size_t i) {
    T sigmoid_val = T(1) / (T(1) + std::exp(-input_data[i]));
    T local_grad = sigmoid_val * (T(1) - sigmoid_val);
    grad_data[i] *= local_grad;
  });
}

template <typename T> void Sigmoid<T>::apply_channel_wise(Tensor<T> &tensor, int channel) const {
  if (channel < 0 || channel >= static_cast<int>(tensor.channels())) {
    throw std::invalid_argument("Channel index out of bounds");
  }

  size_t batch_size = tensor.batch_size();
  size_t height = tensor.height();
  size_t width = tensor.width();

  utils::parallel_for<size_t>(0, batch_size, [&](size_t n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        T &val = tensor(n, channel, h, w);
        val = T(1) / (T(1) + std::exp(-val));
      }
    }
  });
}

template <typename T>
void Sigmoid<T>::apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
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

  utils::parallel_for<size_t>(0, batch_size, [&](size_t n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        T val = tensor(n, channel, h, w) + bias[h * width + w];
        tensor(n, channel, h, w) = T(1) / (T(1) + std::exp(-val));
      }
    }
  });
}

template <typename T> void Sigmoid<T>::apply_batch_wise(Tensor<T> &tensor, int batch_idx) const {
  if (batch_idx < 0 || batch_idx >= static_cast<int>(tensor.batch_size())) {
    throw std::invalid_argument("Batch index out of bounds");
  }

  size_t channels = tensor.channels();
  size_t height = tensor.height();
  size_t width = tensor.width();

  utils::parallel_for<size_t>(0, channels, [&](size_t c) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        T &val = tensor(batch_idx, c, h, w);
        val = T(1) / (T(1) + std::exp(-val));
      }
    }
  });
}

template <typename T> std::string Sigmoid<T>::name() const { return "sigmoid"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> Sigmoid<T>::clone() const {
  return std::make_unique<Sigmoid<T>>(*this);
}

template <typename T> void Sigmoid<T>::apply_single_value(T &value) const {
  value = T(1) / (T(1) + std::exp(-value));
}

template <typename T> T Sigmoid<T>::compute_single_gradient(T pre_activation_value) const {
  T sigmoid_val = T(1) / (T(1) + std::exp(-pre_activation_value));
  return sigmoid_val * (T(1) - sigmoid_val);
}

// Explicit template instantiations
template class Sigmoid<float>;
template class Sigmoid<double>;

} // namespace tnn