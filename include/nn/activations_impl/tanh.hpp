/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

// Tanh Activation for Tensors
template <typename T = float>
class Tanh : public ActivationFunction<T> {
public:
  void apply(Tensor<T> &tensor) const override {
    T *data = tensor.data();
    size_t size = tensor.size();

#ifdef USE_TBB 
    tnn::parallel_for_range<size_t>(0, size, [&](size_t i) {
      data[i] = std::tanh(data[i]);
    });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      data[i] = std::tanh(data[i]);
    }
#endif
  }

  void apply_with_bias(Tensor<T> &tensor,
                       const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument("Tensor and bias must have the same shape");
    }

    T *data = tensor.data();
    const T *bias_data = bias.data();
    size_t size = tensor.size();

#ifdef USE_TBB 
    tnn::parallel_for_range<size_t>(0, size, [&](size_t i) {
      T val = data[i] + bias_data[i];
      data[i] = std::tanh(val);
    });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      T val = data[i] + bias_data[i];
      data[i] = std::tanh(val);
    }
#endif
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    T *data = tensor.data();
    size_t size = tensor.size();

#ifdef USE_TBB 
    tnn::parallel_for_range<size_t>(0, size, [&](size_t i) {
      T val = data[i] + bias;
      data[i] = std::tanh(val);
    });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      T val = data[i] + bias;
      data[i] = std::tanh(val);
    }
#endif
  }

  Tensor<T> compute_gradient(
      const Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    Tensor<T> gradient(pre_activation_values.shape());
    const T *input_data = pre_activation_values.data();
    T *grad_data = gradient.data();
    size_t size = pre_activation_values.size();

#ifdef USE_TBB 
    tnn::parallel_for_range<size_t>(0, size, [&](size_t i) {
      // Compute tanh and its gradient from pre-activation: 1 - tanh²(x)
      T tanh_val = std::tanh(input_data[i]);
      grad_data[i] = T(1) - tanh_val * tanh_val;
    });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      // Compute tanh and its gradient from pre-activation: 1 - tanh²(x)
      T tanh_val = std::tanh(input_data[i]);
      grad_data[i] = T(1) - tanh_val * tanh_val;
    }
#endif

    // If upstream gradient is provided, multiply element-wise
    if (upstream_gradient != nullptr) {
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }
      const T *upstream_data = upstream_gradient->data();

#ifdef USE_TBB 
      tnn::parallel_for_range<size_t>(0, size, [&](size_t i) {
        grad_data[i] *= upstream_data[i];
      });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < size; ++i) {
        grad_data[i] *= upstream_data[i];
      }
#endif
    }

    return gradient;
  }

  void compute_gradient_inplace(
      const Tensor<T> &pre_activation_values,
      Tensor<T> &upstream_gradient) const override {
    if (upstream_gradient.shape() != pre_activation_values.shape()) {
      throw std::invalid_argument("Upstream gradient must have the same "
                                  "shape as pre-activation values");
    }

    const T *input_data = pre_activation_values.data();
    T *grad_data = upstream_gradient.data();
    size_t size = pre_activation_values.size();

#ifdef USE_TBB 
    tnn::parallel_for_range<size_t>(0, size, [&](size_t i) {
      // Compute tanh and its gradient from pre-activation: 1 - tanh²(x)
      T tanh_val = std::tanh(input_data[i]);
      T local_grad = T(1) - tanh_val * tanh_val;
      grad_data[i] *= local_grad;
    });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      // Compute tanh and its gradient from pre-activation: 1 - tanh²(x)
      T tanh_val = std::tanh(input_data[i]);
      T local_grad = T(1) - tanh_val * tanh_val;
      grad_data[i] *= local_grad;
    }
#endif
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    if (channel < 0 || channel >= static_cast<int>(tensor.channels())) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    size_t batch_size = tensor.batch_size();
    size_t height = tensor.height();
    size_t width = tensor.width();

#ifdef USE_TBB 
    {
      const size_t total = batch_size * height * width;
      tnn::parallel_for_range<size_t>(0, total, [&](size_t idx) {
        size_t n = idx / (height * width);
        size_t rem = idx % (height * width);
        size_t h = rem / width;
        size_t w = rem % width;
        T &val = tensor(n, channel, h, w);
        val = std::tanh(val);
      });
    }
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T &val = tensor(n, channel, h, w);
          val = std::tanh(val);
        }
      }
    }
#endif
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

#ifdef USE_TBB 
    {
      const size_t total = batch_size * height * width;
      tnn::parallel_for_range<size_t>(0, total, [&](size_t idx) {
        size_t n = idx / (height * width);
        size_t rem = idx % (height * width);
        size_t h = rem / width;
        size_t w = rem % width;
        T val = tensor(n, channel, h, w) + bias[h * width + w];
        tensor(n, channel, h, w) = std::tanh(val);
      });
    }
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T val = tensor(n, channel, h, w) + bias[h * width + w];
          tensor(n, channel, h, w) = std::tanh(val);
        }
      }
    }
#endif
  }

  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override {
    if (batch_idx < 0 || batch_idx >= static_cast<int>(tensor.batch_size())) {
      throw std::invalid_argument("Batch index out of bounds");
    }

    size_t channels = tensor.channels();
    size_t height = tensor.height();
    size_t width = tensor.width();

#ifdef USE_TBB 
    {
      const size_t total = channels * height * width;
      tnn::parallel_for_range<size_t>(0, total, [&](size_t idx) {
        size_t c = idx / (height * width);
        size_t rem = idx % (height * width);
        size_t h = rem / width;
        size_t w = rem % width;
        T &val = tensor(batch_idx, c, h, w);
        val = std::tanh(val);
      });
    }
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T &val = tensor(batch_idx, c, h, w);
          val = std::tanh(val);
        }
      }
    }
#endif
  }

  std::string name() const override { return "tanh"; }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<Tanh<T>>();
  }

protected:
  void apply_single_value(T &value) const override {
    value = std::tanh(value);
  }

  T compute_single_gradient(T pre_activation_value) const override {
    T tanh_val = std::tanh(pre_activation_value);
    return T(1) - tanh_val * tanh_val;
  }
};