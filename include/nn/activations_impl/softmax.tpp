/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/base_activation.hpp"
#include "nn/activations_impl/softmax.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_handler.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace tnn {
template <typename T> void Softmax<T>::apply(Tensor<T> &tensor) const {
  size_t batch_size = tensor.batch_size();
  size_t height = tensor.height();
  size_t width = tensor.width();

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        T max_val = tensor(n, 0, h, w);
        size_t channels = tensor.channels();
        for (size_t c = 1; c < channels; ++c) {
          T val = tensor(n, c, h, w);
          if (val > max_val) {
            max_val = val;
          }
        }

        T sum_exp = T(0);
        for (size_t c = 0; c < channels; ++c) {
          T exp_val = std::exp(tensor(n, c, h, w) - max_val);
          tensor(n, c, h, w) = exp_val;
          sum_exp += exp_val;
        }

        for (size_t c = 0; c < channels; ++c) {
          tensor(n, c, h, w) /= sum_exp;
        }
      }
    }
  });
}

template <typename T>
void Softmax<T>::compute_gradient_inplace(const Tensor<T> &input,
                                          Tensor<T> &upstream_gradient) const {
  size_t batch_size = input.batch_size();
  size_t channels = input.channels();
  size_t height = input.height();
  size_t width = input.width();

  if (upstream_gradient.shape() != input.shape()) {
    throw std::invalid_argument("Upstream gradient must have the same "
                                "shape as pre-activation values");
  }

  Tensor<T> softmax_values = input;
  apply(softmax_values);

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
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

template <typename T> std::string Softmax<T>::name() const { return "softmax"; }

template <typename T> std::unique_ptr<ActivationFunction<T>> Softmax<T>::clone() const {
  return std::make_unique<Softmax<T>>(*this);
}

// Explicit template instantiations
template class Softmax<float>;
template class Softmax<double>;

} // namespace tnn