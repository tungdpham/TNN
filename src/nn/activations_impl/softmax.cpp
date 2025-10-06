/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/activations_impl/softmax.hpp"
#include "nn/activations_impl/base_activation.hpp"
#include "tensor/tensor.hpp"
#include "utils/parallel_for.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tnn {
template <typename T> void Softmax<T>::apply(Tensor<T> &tensor) const {
  size_t batch_size = tensor.batch_size();
  size_t height = tensor.height();
  size_t width = tensor.width();

  utils::parallel_for<size_t>(0, batch_size, [&](size_t n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        apply_softmax_spatial(tensor, n, h, w);
      }
    }
  });
}

template <typename T>
void Softmax<T>::apply_with_bias(Tensor<T> &tensor, const Tensor<T> &bias) const {
  if (tensor.shape() != bias.shape()) {
    throw std::invalid_argument("Tensor and bias must have the same shape");
  }

  T *data = tensor.data();
  const T *bias_data = bias.data();
  size_t size = tensor.size();

  utils::parallel_for<size_t>(0, size, [&](size_t i) { data[i] += bias_data[i]; });

  apply(tensor);
}

template <typename T> void Softmax<T>::apply_with_scalar_bias(Tensor<T> &tensor, T bias) const {
  if (bias != T(0)) {
    T *data = tensor.data();
    size_t size = tensor.size();

    utils::parallel_for<size_t>(0, size, [&](size_t i) { data[i] += bias; });
  }

  apply(tensor);
}

template <typename T>
Tensor<T> Softmax<T>::compute_gradient(const Tensor<T> &pre_activation_values,
                                       const Tensor<T> *upstream_gradient) const {
  if (upstream_gradient == nullptr) {
    throw std::invalid_argument("Upstream gradient must be provided for "
                                "softmax gradient computation");
  }

  if (upstream_gradient->shape() != pre_activation_values.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }

  Tensor<T> gradient = upstream_gradient->clone();
  compute_gradient_inplace(pre_activation_values, gradient);
  return gradient;
}

template <typename T>
void Softmax<T>::compute_gradient_inplace(const Tensor<T> &pre_activation_values,
                                          Tensor<T> &upstream_gradient) const {
  size_t batch_size = pre_activation_values.batch_size();
  size_t channels = pre_activation_values.channels();
  size_t height = pre_activation_values.height();
  size_t width = pre_activation_values.width();

  if (upstream_gradient.shape() != pre_activation_values.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }

  Tensor<T> softmax_values = pre_activation_values;
  apply(softmax_values);

  utils::parallel_for<size_t>(0, batch_size, [&](size_t n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        T dot_product = T(0);
        for (size_t j = 0; j < channels; ++j) {
          dot_product += softmax_values(n, j, h, w) * upstream_gradient(n, j, h, w);
        }

        for (size_t i = 0; i < channels; ++i) {
          T s_i = softmax_values(n, i, h, w);
          T upstream_i = upstream_gradient(n, i, h, w);
          upstream_gradient(n, i, h, w) = s_i * (upstream_i - dot_product);
        }
      }
    }
  });
}

template <typename T> void Softmax<T>::apply_channel_wise(Tensor<T> &tensor, int channel) const {
  (void)tensor;
  (void)channel;
  throw std::runtime_error("Channel-wise softmax is not supported. Use full "
                           "tensor softmax instead.");
}

template <typename T>
void Softmax<T>::apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                                              const std::vector<T> &bias) const {
  (void)tensor;
  (void)channel;
  (void)bias;
  throw std::runtime_error("Channel-wise softmax is not supported. Use full "
                           "tensor softmax instead.");
}

template <typename T> void Softmax<T>::apply_batch_wise(Tensor<T> &tensor, int batch_idx) const {
  if (batch_idx < 0 || batch_idx >= static_cast<int>(tensor.batch_size())) {
    throw std::invalid_argument("Batch index out of bounds");
  }

  size_t height = tensor.height();
  size_t width = tensor.width();

  utils::parallel_for<size_t>(0, height, [&](size_t h) {
    for (size_t w = 0; w < width; ++w) {
      apply_softmax_spatial(tensor, batch_idx, h, w);
    }
  });
}

template <typename T>
void Softmax<T>::apply_spatial(Tensor<T> &tensor, int batch, int channel, int height,
                               int width) const {
  (void)channel;
  apply_softmax_spatial(tensor, batch, height, width);
}

template <typename T> std::string Softmax<T>::name() const { return "softmax"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> Softmax<T>::clone() const {
  return std::make_unique<Softmax<T>>(*this);
}

template <typename T> void Softmax<T>::apply_single_value(T &value) const {
  (void)value;
  throw std::runtime_error("Single value softmax is not supported. Softmax "
                           "requires normalization across channels.");
}

template <typename T> T Softmax<T>::compute_single_gradient(T pre_activation_value) const {
  (void)pre_activation_value;
  throw std::runtime_error("Single value softmax gradient is not supported. "
                           "Use compute_gradient instead.");
}

template <typename T>
void Softmax<T>::apply_softmax_spatial(Tensor<T> &tensor, size_t n, size_t h, size_t w) const {
  size_t channels = tensor.channels();

  T max_val = tensor(n, 0, h, w);
  for (size_t c = 1; c < channels; ++c) {
    T val = tensor(n, c, h, w);
    if (val > max_val)
      max_val = val;
  }

  T sum = T(0);
  for (size_t c = 0; c < channels; ++c) {
    T val = tensor(n, c, h, w);
    tensor(n, c, h, w) = std::exp(val - max_val);
    sum += tensor(n, c, h, w);
  }

  sum = std::max(sum, T(1e-10));
  for (size_t c = 0; c < channels; ++c) {
    tensor(n, c, h, w) /= sum;
  }
}

// Explicit template instantiations
template class Softmax<float>;
template class Softmax<double>;

} // namespace tnn