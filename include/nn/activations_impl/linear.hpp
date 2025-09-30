/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

template <typename T = float> class Linear : public ActivationFunction<T> {
public:
  void apply(Tensor<T> &tensor) const override { (void)tensor; }

  void apply_with_bias(Tensor<T> &tensor, const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument("Tensor and bias must have the same shape");
    }

    T *data = tensor.data();
    const T *bias_data = bias.data();
    const size_t size = tensor.size();

    utils::parallel_for<size_t>(0, size, [&](size_t i) { data[i] += bias_data[i]; });
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    if (bias != T(0)) {
      T *data = tensor.data();
      size_t size = tensor.size();

      utils::parallel_for<size_t>(0, size, [&](size_t i) { data[i] += bias; });
    }
  }

  Tensor<T> compute_gradient(const Tensor<T> &pre_activation_values,
                             const Tensor<T> *upstream_gradient = nullptr) const override {
    if (upstream_gradient == nullptr) {
      throw std::runtime_error("Upstream gradient must be provided for Linear activation");
    }
    return upstream_gradient->clone();
  }

  void compute_gradient_inplace(const Tensor<T> &pre_activation_values,
                                Tensor<T> &upstream_gradient) const override {
    if (upstream_gradient.shape() != pre_activation_values.shape()) {
      throw std::invalid_argument("Upstream gradient must have the same "
                                  "shape as pre-activation values");
    }
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    (void)tensor;
    (void)channel;
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

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          tensor(n, channel, h, w) += bias[h * width + w];
        }
      }
    }
  }

  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override {
    (void)tensor;
    (void)batch_idx;
  }

  std::string name() const override { return "linear"; }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<Linear<T>>();
  }

protected:
  void apply_single_value(T &value) const override { (void)value; }

  T compute_single_gradient(T pre_activation_value) const override {
    (void)pre_activation_value;
    return T(1);
  }
};