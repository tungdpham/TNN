/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "utils/parallel_for.hpp"

// Sigmoid Activation for Tensors
template <typename T = float> class Sigmoid : public ActivationFunction<T> {
public:
  void apply(Tensor<T> &tensor) const override {
    T *data = tensor.data();
    size_t size = tensor.size();

    utils::parallel_for_range<size_t>(0, size, [&](size_t i) {
      data[i] = T(1) / (T(1) + std::exp(-data[i]));
    });
  }

  void apply_with_bias(Tensor<T> &tensor,
                       const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument("Tensor and bias must have the same shape");
    }

    T *data = tensor.data();
    const T *bias_data = bias.data();
    size_t size = tensor.size();

    utils::parallel_for_range<size_t>(0, size, [&](size_t i) {
      T val = data[i] + bias_data[i];
      data[i] = T(1) / (T(1) + std::exp(-val));
    });
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    T *data = tensor.data();
    size_t size = tensor.size();

    utils::parallel_for_range<size_t>(0, size, [&](size_t i) {
      T val = data[i] + bias;
      data[i] = T(1) / (T(1) + std::exp(-val));
    });
  }

  Tensor<T> compute_gradient(
      const Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    Tensor<T> gradient(pre_activation_values.shape());
    const T *input_data = pre_activation_values.data();
    T *grad_data = gradient.data();
    size_t size = pre_activation_values.size();

    utils::parallel_for_range<size_t>(0, size, [&](size_t i) {
      // Compute sigmoid and its gradient from pre-activation
      T sigmoid_val = T(1) / (T(1) + std::exp(-input_data[i]));
      grad_data[i] = sigmoid_val * (T(1) - sigmoid_val);
    });

    // If upstream gradient is provided, multiply element-wise
    if (upstream_gradient != nullptr) {
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }
      const T *upstream_data = upstream_gradient->data();
      
      utils::parallel_for_range<size_t>(0, size, [&](size_t i) {
        grad_data[i] *= upstream_data[i];
      });
    }

    return gradient;
  }

  void compute_gradient_inplace(const Tensor<T> &pre_activation_values,
                                Tensor<T> &upstream_gradient) const override {
    if (upstream_gradient.shape() != pre_activation_values.shape()) {
      throw std::invalid_argument("Upstream gradient must have the same "
                                  "shape as pre-activation values");
    }

    const T *input_data = pre_activation_values.data();
    T *grad_data = upstream_gradient.data();
    size_t size = pre_activation_values.size();

    utils::parallel_for_range<size_t>(0, size, [&](size_t i) {
      // Compute sigmoid and its gradient from pre-activation
      T sigmoid_val = T(1) / (T(1) + std::exp(-input_data[i]));
      T local_grad = sigmoid_val * (T(1) - sigmoid_val);
      grad_data[i] *= local_grad;
    });
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    if (channel < 0 || channel >= static_cast<int>(tensor.channels())) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    size_t batch_size = tensor.batch_size();
    size_t height = tensor.height();
    size_t width = tensor.width();

    utils::parallel_for_range<size_t>(0, batch_size, [&](size_t n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T &val = tensor(n, channel, h, w);
          val = T(1) / (T(1) + std::exp(-val));
        }
      }
    });
  }

  void apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                                    const std::vector<T> &bias) const override {
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

    utils::parallel_for_range<size_t>(0, batch_size, [&](size_t n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T val = tensor(n, channel, h, w) + bias[h * width + w];
          tensor(n, channel, h, w) = T(1) / (T(1) + std::exp(-val));
        }
      }
    });
  }

  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override {
    if (batch_idx < 0 || batch_idx >= static_cast<int>(tensor.batch_size())) {
      throw std::invalid_argument("Batch index out of bounds");
    }

    size_t channels = tensor.channels();
    size_t height = tensor.height();
    size_t width = tensor.width();

    utils::parallel_for_range<size_t>(0, channels, [&](size_t c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T &val = tensor(batch_idx, c, h, w);
          val = T(1) / (T(1) + std::exp(-val));
        }
      }
    });
  }

  std::string name() const override { return "sigmoid"; }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<Sigmoid<T>>();
  }

protected:
  void apply_single_value(T &value) const override {
    value = T(1) / (T(1) + std::exp(-value));
  }

  T compute_single_gradient(T pre_activation_value) const override {
    T sigmoid_val = T(1) / (T(1) + std::exp(-pre_activation_value));
    return sigmoid_val * (T(1) - sigmoid_val);
  }
};