#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../tensor/tensor.hpp"

// Abstract base class for tensor-based activation functions
// Designed for 4D tensors with NCHW layout (Batch, Channels, Height, Width)
template <typename T = float> class ActivationFunction {
public:
  virtual ~ActivationFunction() = default;

  // Core activation methods
  virtual void apply(Tensor<T> &tensor) const = 0;
  virtual void apply_with_bias(Tensor<T> &tensor,
                               const Tensor<T> &bias) const = 0;
  virtual void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const = 0;

  // Gradient computation for backpropagation
  virtual Tensor<T>
  compute_gradient(const Tensor<T> &pre_activation_values,
                   const Tensor<T> *upstream_gradient = nullptr) const = 0;
  virtual void compute_gradient_inplace(
      Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const = 0;

  // Channel-wise operations (useful for batch normalization, etc.)
  virtual void apply_channel_wise(Tensor<T> &tensor, int channel) const = 0;
  virtual void
  apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                               const std::vector<T> &bias) const = 0;

  // Batch-wise operations
  virtual void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const = 0;

  // Utility methods
  virtual std::string name() const = 0;
  virtual std::unique_ptr<ActivationFunction<T>> clone() const = 0;

  // Optional: spatial-wise operations (apply to specific spatial locations)
  virtual void apply_spatial(Tensor<T> &tensor, int batch, int channel,
                             int height, int width) const {
    T &value = tensor(batch, channel, height, width);
    apply_single_value(value);
  }

protected:
  // Helper for single value activation (to be implemented by derived classes)
  virtual void apply_single_value(T &value) const = 0;
  virtual T compute_single_gradient(T pre_activation_value) const = 0;
};

// ReLU Activation for Tensors
template <typename T = float>
class TensorReLU : public ActivationFunction<T> {
private:
  T negative_slope_;

public:
  explicit TensorReLU(T negative_slope = T(0))
      : negative_slope_(negative_slope) {}

  void apply(Tensor<T> &tensor) const override {
    T *data = tensor.data();
    size_t size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      data[i] = data[i] > T(0) ? data[i] : negative_slope_ * data[i];
    }
  }

  void apply_with_bias(Tensor<T> &tensor,
                       const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument("Tensor and bias must have the same shape");
    }

    T *data = tensor.data();
    const T *bias_data = bias.data();
    size_t size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      T val = data[i] + bias_data[i];
      data[i] = val > T(0) ? val : negative_slope_ * val;
    }
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    T *data = tensor.data();
    size_t size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      T val = data[i] + bias;
      data[i] = val > T(0) ? val : negative_slope_ * val;
    }
  }

  Tensor<T> compute_gradient(
      const Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    Tensor<T> gradient(pre_activation_values.shape());
    const T *input_data = pre_activation_values.data();
    T *grad_data = gradient.data();
    size_t size = pre_activation_values.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      grad_data[i] = input_data[i] > T(0) ? T(1) : negative_slope_;
    }

    // If upstream gradient is provided, multiply element-wise
    if (upstream_gradient != nullptr) {
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }
      const T *upstream_data = upstream_gradient->data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < size; ++i) {
        grad_data[i] *= upstream_data[i];
      }
    }

    return gradient;
  }

  void compute_gradient_inplace(
      Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    T *data = pre_activation_values.data();
    size_t size = pre_activation_values.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      data[i] = data[i] > T(0) ? T(1) : negative_slope_;
    }

    // If upstream gradient is provided, multiply element-wise
    if (upstream_gradient != nullptr) {
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }
      const T *upstream_data = upstream_gradient->data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < size; ++i) {
        data[i] *= upstream_data[i];
      }
    }
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    if (channel < 0 || channel >= static_cast<int>(tensor.channels())) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    size_t batch_size = tensor.batch_size();
    size_t height = tensor.height();
    size_t width = tensor.width();

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T &val = tensor(n, channel, h, w);
          val = val > T(0) ? val : negative_slope_ * val;
        }
      }
    }
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

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T val = tensor(n, channel, h, w) + bias[h * width + w];
          tensor(n, channel, h, w) = val > T(0) ? val : negative_slope_ * val;
        }
      }
    }
  }

  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override {
    if (batch_idx < 0 || batch_idx >= static_cast<int>(tensor.batch_size())) {
      throw std::invalid_argument("Batch index out of bounds");
    }

    size_t channels = tensor.channels();
    size_t height = tensor.height();
    size_t width = tensor.width();

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T &val = tensor(batch_idx, c, h, w);
          val = val > T(0) ? val : negative_slope_ * val;
        }
      }
    }
  }

  std::string name() const override {
    return negative_slope_ == T(0) ? "relu" : "leaky_relu";
  }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<TensorReLU<T>>(negative_slope_);
  }

protected:
  void apply_single_value(T &value) const override {
    value = value > T(0) ? value : negative_slope_ * value;
  }

  T compute_single_gradient(T pre_activation_value) const override {
    return pre_activation_value > T(0) ? T(1) : negative_slope_;
  }
};

// Sigmoid Activation for Tensors
template <typename T = float>
class TensorSigmoid : public ActivationFunction<T> {
public:
  void apply(Tensor<T> &tensor) const override {
    T *data = tensor.data();
    size_t size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      data[i] = T(1) / (T(1) + std::exp(-data[i]));
    }
  }

  void apply_with_bias(Tensor<T> &tensor,
                       const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument("Tensor and bias must have the same shape");
    }

    T *data = tensor.data();
    const T *bias_data = bias.data();
    size_t size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      T val = data[i] + bias_data[i];
      data[i] = T(1) / (T(1) + std::exp(-val));
    }
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    T *data = tensor.data();
    size_t size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      T val = data[i] + bias;
      data[i] = T(1) / (T(1) + std::exp(-val));
    }
  }

  Tensor<T> compute_gradient(
      const Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    Tensor<T> gradient(pre_activation_values.shape());
    const T *input_data = pre_activation_values.data();
    T *grad_data = gradient.data();
    size_t size = pre_activation_values.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      // Compute sigmoid and its gradient from pre-activation
      T sigmoid_val = T(1) / (T(1) + std::exp(-input_data[i]));
      grad_data[i] = sigmoid_val * (T(1) - sigmoid_val);
    }

    // If upstream gradient is provided, multiply element-wise
    if (upstream_gradient != nullptr) {
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }
      const T *upstream_data = upstream_gradient->data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < size; ++i) {
        grad_data[i] *= upstream_data[i];
      }
    }

    return gradient;
  }

  void compute_gradient_inplace(
      Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    T *data = pre_activation_values.data();
    size_t size = pre_activation_values.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      // Compute sigmoid and its gradient from pre-activation
      T sigmoid_val = T(1) / (T(1) + std::exp(-data[i]));
      data[i] = sigmoid_val * (T(1) - sigmoid_val);
    }

    // If upstream gradient is provided, multiply element-wise
    if (upstream_gradient != nullptr) {
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }
      const T *upstream_data = upstream_gradient->data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < size; ++i) {
        data[i] *= upstream_data[i];
      }
    }
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    if (channel < 0 || channel >= static_cast<int>(tensor.channels())) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    size_t batch_size = tensor.batch_size();
    size_t height = tensor.height();
    size_t width = tensor.width();

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T &val = tensor(n, channel, h, w);
          val = T(1) / (T(1) + std::exp(-val));
        }
      }
    }
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

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T val = tensor(n, channel, h, w) + bias[h * width + w];
          tensor(n, channel, h, w) = T(1) / (T(1) + std::exp(-val));
        }
      }
    }
  }

  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override {
    if (batch_idx < 0 || batch_idx >= static_cast<int>(tensor.batch_size())) {
      throw std::invalid_argument("Batch index out of bounds");
    }

    size_t channels = tensor.channels();
    size_t height = tensor.height();
    size_t width = tensor.width();

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (size_t c = 0; c < channels; ++c) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T &val = tensor(batch_idx, c, h, w);
          val = T(1) / (T(1) + std::exp(-val));
        }
      }
    }
  }

  std::string name() const override { return "sigmoid"; }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<TensorSigmoid<T>>();
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

// Softmax Activation for Tensors
template <typename T = float>
class TensorSoftmax : public ActivationFunction<T> {
public:
  void apply(Tensor<T> &tensor) const override {
    size_t batch_size = tensor.batch_size();
    // size_t channels = tensor.channels();
    size_t height = tensor.height();
    size_t width = tensor.width();

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    // Apply softmax across channels for each spatial location
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          apply_softmax_spatial(tensor, n, h, w);
        }
      }
    }
  }

  void apply_with_bias(Tensor<T> &tensor,
                       const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument("Tensor and bias must have the same shape");
    }

    // Add bias first
    T *data = tensor.data();
    const T *bias_data = bias.data();
    size_t size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      data[i] += bias_data[i];
    }

    // Then apply softmax
    apply(tensor);
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    if (bias != T(0)) {
      T *data = tensor.data();
      size_t size = tensor.size();

      for (size_t i = 0; i < size; ++i) {
        data[i] += bias;
      }
    }

    apply(tensor);
  }

  Tensor<T> compute_gradient(
      const Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    Tensor<T> gradient(pre_activation_values.shape());

    // First compute softmax from pre-activation values
    Tensor<T> softmax_values = pre_activation_values; // Copy
    apply(softmax_values); // Apply softmax to get activated values

    size_t batch_size = pre_activation_values.batch_size();
    size_t channels = pre_activation_values.channels();
    size_t height = pre_activation_values.height();
    size_t width = pre_activation_values.width();

    if (upstream_gradient == nullptr) {
      // If no upstream gradient provided, assume uniform gradient of 1
      // (simplified case)
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            // Compute sum of all softmax outputs at this spatial location
            T sum_outputs = T(0);
            for (size_t c = 0; c < channels; ++c) {
              sum_outputs += softmax_values(n, c, h, w);
            }

            // Compute gradient for each channel at this spatial location
            for (size_t i = 0; i < channels; ++i) {
              T s_i = softmax_values(n, i, h, w);
              gradient(n, i, h, w) = s_i * (T(1) - sum_outputs);
            }
          }
        }
      }
    } else {
      // Proper softmax gradient computation with upstream gradient
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            // Compute the dot product of softmax outputs and upstream gradients
            T dot_product = T(0);
            for (size_t j = 0; j < channels; ++j) {
              dot_product +=
                  softmax_values(n, j, h, w) * (*upstream_gradient)(n, j, h, w);
            }

            // Compute gradient for each channel at this spatial location
            for (size_t i = 0; i < channels; ++i) {
              T s_i = softmax_values(n, i, h, w);
              T upstream_i = (*upstream_gradient)(n, i, h, w);
              gradient(n, i, h, w) = s_i * (upstream_i - dot_product);
            }
          }
        }
      }
    }

    return gradient;
  }

  void compute_gradient_inplace(
      Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    size_t batch_size = pre_activation_values.batch_size();
    size_t channels = pre_activation_values.channels();
    size_t height = pre_activation_values.height();
    size_t width = pre_activation_values.width();

    // First compute softmax from pre-activation values
    Tensor<T> softmax_values = pre_activation_values; // Copy
    apply(softmax_values); // Apply softmax to get activated values

    if (upstream_gradient == nullptr) {
      // If no upstream gradient provided, assume uniform gradient of 1
      // (simplified case)
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            // Compute sum of all softmax outputs at this spatial location
            T sum_outputs = T(0);
            for (size_t c = 0; c < channels; ++c) {
              sum_outputs += softmax_values(n, c, h, w);
            }

            // Compute gradient for each channel at this spatial location
            for (size_t i = 0; i < channels; ++i) {
              T s_i = softmax_values(n, i, h, w);
              pre_activation_values(n, i, h, w) = s_i * (T(1) - sum_outputs);
            }
          }
        }
      }
    } else {
      // Proper softmax gradient computation with upstream gradient
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            // Compute the dot product of softmax outputs and upstream gradients
            T dot_product = T(0);
            for (size_t j = 0; j < channels; ++j) {
              dot_product +=
                  softmax_values(n, j, h, w) * (*upstream_gradient)(n, j, h, w);
            }

            // Compute gradient for each channel at this spatial location
            for (size_t i = 0; i < channels; ++i) {
              T s_i = softmax_values(n, i, h, w);
              T upstream_i = (*upstream_gradient)(n, i, h, w);
              pre_activation_values(n, i, h, w) =
                  s_i * (upstream_i - dot_product);
            }
          }
        }
      }
    }
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    // For softmax, channel-wise application doesn't make much sense
    // as softmax needs to see all channels to normalize properly
    throw std::runtime_error("Channel-wise softmax is not supported. Use full "
                             "tensor softmax instead.");
  }

  void apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                                    const std::vector<T> &bias) const override {
    throw std::runtime_error("Channel-wise softmax is not supported. Use full "
                             "tensor softmax instead.");
  }

  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override {
    if (batch_idx < 0 || batch_idx >= static_cast<int>(tensor.batch_size())) {
      throw std::invalid_argument("Batch index out of bounds");
    }

    // size_t channels = tensor.channels();
    size_t height = tensor.height();
    size_t width = tensor.width();

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        apply_softmax_spatial(tensor, batch_idx, h, w);
      }
    }
  }

  // Override apply_spatial to properly handle softmax across channels at a
  // specific spatial location
  void apply_spatial(Tensor<T> &tensor, int batch, int channel, int height,
                     int width) const override {
    // For softmax, we need to apply it across all channels at this spatial
    // location We can't just apply it to a single value - that's mathematically
    // meaningless
    apply_softmax_spatial(tensor, batch, height, width);
  }

  std::string name() const override { return "softmax"; }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<TensorSoftmax<T>>();
  }

protected:
  void apply_single_value(T &value) const override {
    // Single value softmax is mathematically meaningless - softmax requires
    // normalization across a group
    throw std::runtime_error("Single value softmax is not supported. Softmax "
                             "requires normalization across channels.");
  }

  T compute_single_gradient(T pre_activation_value) const override {
    // Single value softmax gradient is not well-defined without the full
    // softmax context
    throw std::runtime_error("Single value softmax gradient is not supported. "
                             "Use compute_gradient instead.");
  }

private:
  void apply_softmax_spatial(Tensor<T> &tensor, size_t n, size_t h,
                             size_t w) const {
    size_t channels = tensor.channels();

    // Find max for numerical stability
    T max_val = tensor(n, 0, h, w);
    for (size_t c = 1; c < channels; ++c) {
      T val = tensor(n, c, h, w);
      if (val > max_val)
        max_val = val;
    }

    // Compute exp and sum
    T sum = T(0);
    for (size_t c = 0; c < channels; ++c) {
      T val = tensor(n, c, h, w);
      tensor(n, c, h, w) = std::exp(val - max_val);
      sum += tensor(n, c, h, w);
    }

    // Normalize
    for (size_t c = 0; c < channels; ++c) {
      tensor(n, c, h, w) /= sum;
    }
  }
};

// Linear (Identity) Activation for Tensors
template <typename T = float>
class TensorLinear : public ActivationFunction<T> {
public:
  void apply(Tensor<T> &tensor) const override {
    // Linear activation does nothing to the values
  }

  void apply_with_bias(Tensor<T> &tensor,
                       const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument("Tensor and bias must have the same shape");
    }

    T *data = tensor.data();
    const T *bias_data = bias.data();
    size_t size = tensor.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      data[i] += bias_data[i];
    }
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    if (bias != T(0)) {
      T *data = tensor.data();
      size_t size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < size; ++i) {
        data[i] += bias;
      }
    }
  }

  Tensor<T> compute_gradient(
      const Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    if (upstream_gradient != nullptr) {
      // For linear activation, gradient is just the upstream gradient
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }
      return *upstream_gradient;
    } else {
      // If no upstream gradient, return tensor of ones (derivative of linear
      // function is 1)
      Tensor<T> gradient(pre_activation_values.shape());
      gradient.fill(T(1));
      return gradient;
    }
  }

  void compute_gradient_inplace(
      Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    if (upstream_gradient != nullptr) {
      // For linear activation, gradient is just the upstream gradient
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }
      pre_activation_values = *upstream_gradient;
    } else {
      // If no upstream gradient, fill with ones (derivative of linear function
      // is 1)
      pre_activation_values.fill(T(1));
    }
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    // Linear activation does nothing
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
    // Linear activation does nothing
  }

  std::string name() const override { return "linear"; }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<TensorLinear<T>>();
  }

protected:
  void apply_single_value(T &value) const override {
    // Linear activation does nothing
  }

  T compute_single_gradient(T pre_activation_value) const override {
    return T(1);
  }
};

// Factory for creating tensor activation functions
template <typename T = float> class ActivationFactory {
private:
  static std::unordered_map<
      std::string,
      std::function<std::unique_ptr<ActivationFunction<T>>()>>
      creators_;

public:
  static void register_activation(
      const std::string &name,
      std::function<std::unique_ptr<ActivationFunction<T>>()> creator) {
    creators_[name] = creator;
  }

  static std::unique_ptr<ActivationFunction<T>>
  create(const std::string &name) {
    auto it = creators_.find(name);
    if (it != creators_.end()) {
      return it->second();
    }
    throw std::invalid_argument("Unknown activation function: " + name);
  }

  static void register_defaults() {
    register_activation("relu",
                        []() { return std::make_unique<TensorReLU<T>>(); });
    register_activation("leaky_relu", []() {
      return std::make_unique<TensorReLU<T>>(T(0.01));
    });
    register_activation("sigmoid",
                        []() { return std::make_unique<TensorSigmoid<T>>(); });
    register_activation("softmax",
                        []() { return std::make_unique<TensorSoftmax<T>>(); });
    register_activation("linear",
                        []() { return std::make_unique<TensorLinear<T>>(); });
  }

  static std::vector<std::string> get_available_activations() {
    std::vector<std::string> names;
    for (const auto &pair : creators_) {
      names.push_back(pair.first);
    }
    return names;
  }
};

template <typename T>
std::unordered_map<
    std::string, std::function<std::unique_ptr<ActivationFunction<T>>()>>
    ActivationFactory<T>::creators_;

// Convenience namespace for direct function calls
namespace TensorActivations {

template <typename T = float>
void apply_relu(Tensor<T> &tensor, T negative_slope = T(0)) {
  TensorReLU<T> relu(negative_slope);
  relu.apply(tensor);
}

template <typename T = float> void apply_sigmoid(Tensor<T> &tensor) {
  TensorSigmoid<T> sigmoid;
  sigmoid.apply(tensor);
}

template <typename T = float> void apply_softmax(Tensor<T> &tensor) {
  TensorSoftmax<T> softmax;
  softmax.apply(tensor);
}

template <typename T = float> void apply_linear(Tensor<T> &tensor) {
  TensorLinear<T> linear;
  linear.apply(tensor);
}

// Channel-wise operations
template <typename T = float>
void apply_relu_channel(Tensor<T> &tensor, int channel,
                        T negative_slope = T(0)) {
  TensorReLU<T> relu(negative_slope);
  relu.apply_channel_wise(tensor, channel);
}

template <typename T = float>
void apply_sigmoid_channel(Tensor<T> &tensor, int channel) {
  TensorSigmoid<T> sigmoid;
  sigmoid.apply_channel_wise(tensor, channel);
}

// Batch-wise operations
template <typename T = float>
void apply_relu_batch(Tensor<T> &tensor, int batch_idx,
                      T negative_slope = T(0)) {
  TensorReLU<T> relu(negative_slope);
  relu.apply_batch_wise(tensor, batch_idx);
}
} // namespace TensorActivations

/* Usage Examples:

// Create a 4D tensor (batch=2, channels=64, height=32, width=32)
TensorF feature_maps(2, 64, 32, 32);
feature_maps.fill_random_normal(0.1f);

// Method 1: Direct function calls
TensorActivations::apply_relu(feature_maps);

// Method 2: Using factory
ActivationFactory<float> factory;
factory.register_defaults();
auto relu = factory.create("relu");
relu->apply(feature_maps);

// Method 3: With bias
TensorF bias(2, 64, 32, 32);
bias.fill(0.01f);
relu->apply_with_bias(feature_maps, bias);

// Method 4: Channel-wise operations (useful for batch normalization)
for (int c = 0; c < static_cast<int>(feature_maps.channels()); ++c) {
    TensorActivations::apply_relu_channel(feature_maps, c);
}

// Method 5: Compute gradients for backpropagation
TensorF gradients = relu->compute_gradient(feature_maps);

// Method 6: Softmax across channels for each spatial location
TensorF logits(1, 10, 1, 1);  // Classification logits
TensorActivations::apply_softmax(logits);

*/
