#pragma once

#include <algorithm>

#include "tensor_activations.hpp"

// Advanced activation functions for CNNs
namespace TensorActivations {

// Swish activation: f(x) = x * sigmoid(x)
template <typename T = double>
class TensorSwish : public TensorActivationFunction<T> {
private:
  mutable Tensor<T>
      cached_inputs_; // Store original inputs for gradient computation
  mutable bool cache_valid_;

public:
  TensorSwish() : cache_valid_(false) {}

  void apply(Tensor<T> &tensor) const override {
    // Cache the original inputs for gradient computation
    cached_inputs_ = tensor; // Store pre-activated values
    cache_valid_ = true;

    T *data = tensor.data();
    int size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T x = data[i];
      T sigmoid_x = T(1) / (T(1) + std::exp(-x));
      data[i] = x * sigmoid_x;
    }
  }

  void apply_with_bias(Tensor<T> &tensor,
                       const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument(
          "Tensor<T> and bias must have the same shape");
    }

    // Add bias first, then cache the pre-activated values
    T *data = tensor.data();
    const T *bias_data = bias.data();
    int size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      data[i] += bias_data[i];
    }

    // Cache the pre-activated values (after bias addition)
    cached_inputs_ = tensor;
    cache_valid_ = true;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T x = data[i];
      T sigmoid_x = T(1) / (T(1) + std::exp(-x));
      data[i] = x * sigmoid_x;
    }
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    T *data = tensor.data();
    int size = tensor.size();

    // Add bias first
    if (bias != T(0)) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < size; ++i) {
        data[i] += bias;
      }
    }

    // Cache the pre-activated values (after bias addition)
    cached_inputs_ = tensor;
    cache_valid_ = true;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T x = data[i];
      T sigmoid_x = T(1) / (T(1) + std::exp(-x));
      data[i] = x * sigmoid_x;
    }
  }

  Tensor<T> compute_gradient(
      const Tensor<T> &activated_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    if (!cache_valid_) {
      throw std::runtime_error("Swish gradient computation requires cached "
                               "input values. Call apply() first.");
    }

    if (cached_inputs_.shape() != activated_values.shape()) {
      throw std::runtime_error(
          "Cached inputs shape mismatch with activated values");
    }

    Tensor<T> gradient(activated_values.shape(), activated_values.layout());
    const T *input_data =
        cached_inputs_.data(); // Use original pre-activated values
    T *grad_data = gradient.data();
    int size = activated_values.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T x = input_data[i]; // Original pre-activated value
      T sigmoid_x = T(1) / (T(1) + std::exp(-x));

      // Correct Swish derivative: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 -
      // sigmoid(x))
      grad_data[i] = sigmoid_x + x * sigmoid_x * (T(1) - sigmoid_x);
    }

    // Apply upstream gradient if provided
    if (upstream_gradient != nullptr) {
      if (upstream_gradient->shape() != activated_values.shape()) {
        throw std::invalid_argument(
            "Upstream gradient must have the same shape as activated values");
      }
      const T *upstream_data = upstream_gradient->data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < size; ++i) {
        grad_data[i] *= upstream_data[i];
      }
    }

    return gradient;
  }

  void compute_gradient_inplace(
      Tensor<T> &activated_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    if (!cache_valid_) {
      throw std::runtime_error("Swish gradient computation requires cached "
                               "input values. Call apply() first.");
    }

    if (cached_inputs_.shape() != activated_values.shape()) {
      throw std::runtime_error(
          "Cached inputs shape mismatch with activated values");
    }

    const T *input_data =
        cached_inputs_.data(); // Use original pre-activated values
    T *data = activated_values.data();
    int size = activated_values.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T x = input_data[i]; // Original pre-activated value
      T sigmoid_x = T(1) / (T(1) + std::exp(-x));

      // Correct Swish derivative: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 -
      // sigmoid(x))
      data[i] = sigmoid_x + x * sigmoid_x * (T(1) - sigmoid_x);
    }

    // Apply upstream gradient if provided
    if (upstream_gradient != nullptr) {
      if (upstream_gradient->shape() != activated_values.shape()) {
        throw std::invalid_argument(
            "Upstream gradient must have the same shape as activated values");
      }
      const T *upstream_data = upstream_gradient->data();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < size; ++i) {
        data[i] *= upstream_data[i];
      }
    }
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    if (channel < 0 || channel >= tensor.channels()) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    int batch_size = tensor.batch_size();
    int height = tensor.height();
    int width = tensor.width();

    for (int n = 0; n < batch_size; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          T &x = tensor(n, channel, h, w);
          T sigmoid_x = T(1) / (T(1) + std::exp(-x));
          x = x * sigmoid_x;
        }
      }
    }
  }

  void apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                                    const std::vector<T> &bias) const override {
    if (channel < 0 || channel >= tensor.channels()) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    int batch_size = tensor.batch_size();
    int height = tensor.height();
    int width = tensor.width();
    int spatial_size = height * width;

    if ((int)bias.size() != spatial_size) {
      throw std::invalid_argument("Bias size must match spatial dimensions");
    }

    for (int n = 0; n < batch_size; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          T &x = tensor(n, channel, h, w);
          x += bias[h * width + w];
          T sigmoid_x = T(1) / (T(1) + std::exp(-x));
          x = x * sigmoid_x;
        }
      }
    }
  }

  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override {
    if (batch_idx < 0 || batch_idx >= tensor.batch_size()) {
      throw std::invalid_argument("Batch index out of bounds");
    }

    int channels = tensor.channels();
    int height = tensor.height();
    int width = tensor.width();

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          T &x = tensor(batch_idx, c, h, w);
          T sigmoid_x = T(1) / (T(1) + std::exp(-x));
          x = x * sigmoid_x;
        }
      }
    }
  }

  std::string name() const override { return "tensor_swish"; }

  std::unique_ptr<TensorActivationFunction<T>> clone() const override {
    return std::make_unique<TensorSwish<T>>();
  }

protected:
  void apply_single_value(T &value) const override {
    T sigmoid_val = T(1) / (T(1) + std::exp(-value));
    value = value * sigmoid_val;
  }

  T compute_single_gradient(T activated_value) const override {
    return std::max(T(0), activated_value) + T(0.1) * activated_value;
  }
};

// GELU activation: f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 *
// x^3)))
template <typename T = double>
class TensorGELU : public TensorActivationFunction<T> {
private:
  static constexpr T SQRT_2_PI = T(0.7978845608028654); // sqrt(2/π)
  static constexpr T GELU_COEFF = T(0.044715);

public:
  void apply(Tensor<T> &tensor) const override {
    T *data = tensor.data();
    int size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T x = data[i];
      T x3 = x * x * x;
      T tanh_arg = SQRT_2_PI * (x + GELU_COEFF * x3);
      data[i] = T(0.5) * x * (T(1) + std::tanh(tanh_arg));
    }
  }

  void apply_with_bias(Tensor<T> &tensor,
                       const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument(
          "Tensor<T> and bias must have the same shape");
    }

    T *data = tensor.data();
    const T *bias_data = bias.data();
    int size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T x = data[i] + bias_data[i];
      T x3 = x * x * x;
      T tanh_arg = SQRT_2_PI * (x + GELU_COEFF * x3);
      data[i] = T(0.5) * x * (T(1) + std::tanh(tanh_arg));
    }
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    T *data = tensor.data();
    int size = tensor.size();

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
      T x = data[i] + bias;
      T x3 = x * x * x;
      T tanh_arg = SQRT_2_PI * (x + GELU_COEFF * x3);
      data[i] = T(0.5) * x * (T(1) + std::tanh(tanh_arg));
    }
  }

  Tensor<T> compute_gradient(const Tensor<T> &activated_values) const override {
    // GELU gradient is complex - simplified approximation
    Tensor<T> gradient(activated_values.shape(), activated_values.layout());
    const T *input_data = activated_values.data();
    T *grad_data = gradient.data();
    int size = activated_values.size();

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
      // Approximate gradient (in practice, store intermediate values)
      T val = input_data[i];
      grad_data[i] = val > T(0) ? T(1) : T(0.1); // Simplified
    }

    return gradient;
  }

  void compute_gradient_inplace(Tensor<T> &activated_values) const override {
    T *data = activated_values.data();
    int size = activated_values.size();

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
      T val = data[i];
      data[i] = val > T(0) ? T(1) : T(0.1);
    }
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    if (channel < 0 || channel >= tensor.channels()) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    int batch_size = tensor.batch_size();
    int height = tensor.height();
    int width = tensor.width();

    for (int n = 0; n < batch_size; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          T &x = tensor(n, channel, h, w);
          T x3 = x * x * x;
          T tanh_arg = SQRT_2_PI * (x + GELU_COEFF * x3);
          x = T(0.5) * x * (T(1) + std::tanh(tanh_arg));
        }
      }
    }
  }

  void apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                                    const std::vector<T> &bias) const override {
    if (channel < 0 || channel >= tensor.channels()) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    int batch_size = tensor.batch_size();
    int height = tensor.height();
    int width = tensor.width();
    int spatial_size = height * width;

    if ((int)bias.size() != spatial_size) {
      throw std::invalid_argument("Bias size must match spatial dimensions");
    }

    for (int n = 0; n < batch_size; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          T &x = tensor(n, channel, h, w);
          x += bias[h * width + w];
          T x3 = x * x * x;
          T tanh_arg = SQRT_2_PI * (x + GELU_COEFF * x3);
          x = T(0.5) * x * (T(1) + std::tanh(tanh_arg));
        }
      }
    }
  }

  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override {
    if (batch_idx < 0 || batch_idx >= tensor.batch_size()) {
      throw std::invalid_argument("Batch index out of bounds");
    }

    int channels = tensor.channels();
    int height = tensor.height();
    int width = tensor.width();

    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          T &x = tensor(batch_idx, c, h, w);
          T x3 = x * x * x;
          T tanh_arg = SQRT_2_PI * (x + GELU_COEFF * x3);
          x = T(0.5) * x * (T(1) + std::tanh(tanh_arg));
        }
      }
    }
  }

  std::string name() const override { return "tensor_gelu"; }

  std::unique_ptr<TensorActivationFunction<T>> clone() const override {
    return std::make_unique<TensorGELU<T>>();
  }

protected:
  void apply_single_value(T &value) const override {
    T x3 = value * value * value;
    T tanh_arg = SQRT_2_PI * (value + GELU_COEFF * x3);
    value = T(0.5) * value * (T(1) + std::tanh(tanh_arg));
  }

  T compute_single_gradient(T activated_value) const override {
    return activated_value > T(0) ? T(1) : T(0.1);
  }
};

// ELU activation: f(x) = x if x > 0, α(e^x - 1) if x <= 0
template <typename T = double>
class TensorELU : public TensorActivationFunction<T> {
private:
  T alpha_;

public:
  explicit TensorELU(T alpha = T(1.0)) : alpha_(alpha) {}

  void apply(Tensor<T> &tensor) const override {
    T *data = tensor.data();
    int size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T x = data[i];
      data[i] = x > T(0) ? x : alpha_ * (std::exp(x) - T(1));
    }
  }

  void apply_with_bias(Tensor<T> &tensor,
                       const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument(
          "Tensor<T> and bias must have the same shape");
    }

    T *data = tensor.data();
    const T *bias_data = bias.data();
    int size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T x = data[i] + bias_data[i];
      data[i] = x > T(0) ? x : alpha_ * (std::exp(x) - T(1));
    }
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    T *data = tensor.data();
    int size = tensor.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T x = data[i] + bias;
      data[i] = x > T(0) ? x : alpha_ * (std::exp(x) - T(1));
    }
  }

  Tensor<T> compute_gradient(const Tensor<T> &activated_values) const override {
    Tensor<T> gradient(activated_values.shape(), activated_values.layout());
    const T *input_data = activated_values.data();
    T *grad_data = gradient.data();
    int size = activated_values.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T val = input_data[i];
      grad_data[i] = val > T(0) ? T(1) : val + alpha_;
    }

    return gradient;
  }

  void compute_gradient_inplace(Tensor<T> &activated_values) const override {
    T *data = activated_values.data();
    int size = activated_values.size();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      T val = data[i];
      data[i] = val > T(0) ? T(1) : val + alpha_;
    }
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    if (channel < 0 || channel >= tensor.channels()) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    int batch_size = tensor.batch_size();
    int height = tensor.height();
    int width = tensor.width();

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int n = 0; n < batch_size; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          T &x = tensor(n, channel, h, w);
          x = x > T(0) ? x : alpha_ * (std::exp(x) - T(1));
        }
      }
    }
  }

  void apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                                    const std::vector<T> &bias) const override {
    if (channel < 0 || channel >= tensor.channels()) {
      throw std::invalid_argument("Channel index out of bounds");
    }

    int batch_size = tensor.batch_size();
    int height = tensor.height();
    int width = tensor.width();
    int spatial_size = height * width;

    if ((int)bias.size() != spatial_size) {
      throw std::invalid_argument("Bias size must match spatial dimensions");
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int n = 0; n < batch_size; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          T &x = tensor(n, channel, h, w);
          x += bias[h * width + w];
          x = x > T(0) ? x : alpha_ * (std::exp(x) - T(1));
        }
      }
    }
  }

  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override {
    if (batch_idx < 0 || batch_idx >= tensor.batch_size()) {
      throw std::invalid_argument("Batch index out of bounds");
    }

    int channels = tensor.channels();
    int height = tensor.height();
    int width = tensor.width();

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          T &x = tensor(batch_idx, c, h, w);
          x = x > T(0) ? x : alpha_ * (std::exp(x) - T(1));
        }
      }
    }
  }

  std::string name() const override { return "tensor_elu"; }

  std::unique_ptr<TensorActivationFunction<T>> clone() const override {
    return std::make_unique<TensorELU<T>>(alpha_);
  }

protected:
  void apply_single_value(T &value) const override {
    value = value > T(0) ? value : alpha_ * (std::exp(value) - T(1));
  }

  T compute_single_gradient(T activated_value) const override {
    return activated_value > T(0) ? T(1) : activated_value + alpha_;
  }
};

// Convenience functions for advanced activations
template <typename T = double> void apply_swish(Tensor<T> &tensor) {
  TensorSwish<T> swish;
  swish.apply(tensor);
}

template <typename T = double> void apply_gelu(Tensor<T> &tensor) {
  TensorGELU<T> gelu;
  gelu.apply(tensor);
}

template <typename T = double>
void apply_elu(Tensor<T> &tensor, T alpha = T(1.0)) {
  TensorELU<T> elu(alpha);
  elu.apply(tensor);
}

} // namespace TensorActivations

// Extended factory registration
template <typename T = double>
class ExtendedTensorActivationFactory : public TensorActivationFactory<T> {
public:
  static void register_extended() {
    TensorActivationFactory<T>::register_defaults();
    TensorActivationFactory<T>::register_activation("swish", []() {
      return std::make_unique<TensorActivations::TensorSwish<T>>();
    });
    TensorActivationFactory<T>::register_activation("gelu", []() {
      return std::make_unique<TensorActivations::TensorGELU<T>>();
    });
    TensorActivationFactory<T>::register_activation("elu", []() {
      return std::make_unique<TensorActivations::TensorELU<T>>();
    });
  }
};

/* Usage Examples:

// Using advanced activations
Tensor<T> feature_maps(8, 256, 16, 16);
feature_maps.fill_random_normal(0.1);

// Swish activation (often better than ReLU for deep networks)
TensorActivations::apply_swish(feature_maps);

// GELU activation (popular in transformers, also good for CNNs)
TensorActivations::apply_gelu(feature_maps);

// ELU activation (smoother than ReLU, better gradients)
TensorActivations::apply_elu(feature_maps, 1.0);

// Using factory
ExtendedTensorActivationFactory<double> factory;
factory.register_extended();

auto swish = factory.create("swish");
swish->apply(feature_maps);

auto gelu = factory.create("gelu");
gelu->apply(feature_maps);

*/
