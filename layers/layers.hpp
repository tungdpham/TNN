#pragma once

#include <any>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../tensor/tensor.hpp"
#include "activations.hpp"

// BLAS optimization support
#ifdef USE_OPENBLAS
#include <cblas.h>
#elif defined(USE_MKL)
#include <mkl_cblas.h>
#elif defined(USE_ATLAS)
extern "C" {
#include <cblas.h>
}
#endif

// Forward declarations
class Optimizer;

namespace layers {
// Convenience function for creating tensor activation functions
template <typename T = float>
std::unique_ptr<ActivationFunction<T>>
create_activation(const std::string &name) {
  // Ensure factory has default activations registered
  ActivationFactory<T>::register_defaults();
  return ActivationFactory<T>::create(name);
}

// Base configuration for all layers
struct LayerConfig {
  std::string name;
  std::unordered_map<std::string, std::any> parameters;

  template <typename T>
  T get(const std::string &key, const T &default_value = T{}) const {
    auto it = parameters.find(key);
    if (it != parameters.end()) {
      try {
        return std::any_cast<T>(it->second);
      } catch (const std::bad_any_cast &) {
        return default_value;
      }
    }
    return default_value;
  }
};

// Abstract base layer interface
template <typename T = float> class Layer {
public:
  virtual ~Layer() = default;

  // Core forward/backward operations
  virtual Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) = 0;
  virtual Tensor<T> backward(const Tensor<T> &grad_output,
                             int micro_batch_id = 0) = 0;

  // Parameter management
  virtual std::vector<Tensor<T> *> parameters() { return {}; }

  virtual std::vector<Tensor<T> *> gradients() { return {}; }

  virtual bool has_parameters() const { return false; }

  // Configuration and introspection
  virtual std::string type() const = 0;
  virtual LayerConfig get_config() const = 0;
  virtual std::unique_ptr<Layer<T>> clone() const = 0;

  // Training state
  virtual void set_training(bool training) { is_training_ = training; }

  virtual bool is_training() const { return is_training_; }

  // Output shape inference for different Tensor<T> types
  virtual std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const = 0;

  // Optional: custom parameter update (for layers that need special handling)
  virtual void update_parameters(const Optimizer &optimizer) {}

  std::string name() const { return name_; }

protected:
  bool is_training_ = true;
  std::string name_;
};

// Base class for layers without parameters (activation, pooling, etc.)
template <typename T = float> class StatelessLayer : public Layer<T> {
public:
  explicit StatelessLayer(const std::string &name = "") { this->name_ = name; }

  std::vector<Tensor<T> *> parameters() override { return {}; }

  std::vector<Tensor<T> *> gradients() override { return {}; }

  bool has_parameters() const override { return false; }
};

// Base class for layers with parameters (dense, conv, etc.)
template <typename T = float> class ParameterizedLayer : public Layer<T> {
public:
  explicit ParameterizedLayer(const std::string &name = "") {
    this->name_ = name;
  }

  std::vector<Tensor<T> *> parameters() override {
    std::vector<Tensor<T> *> params;
    collect_parameters(params);
    return params;
  }

  std::vector<Tensor<T> *> gradients() override {
    std::vector<Tensor<T> *> grads;
    collect_gradients(grads);
    return grads;
  }

  bool has_parameters() const override { return true; }

  void update_parameters(const Optimizer &optimizer) override {
    update_parameters_impl(optimizer);
  }

protected:
  virtual void collect_parameters(std::vector<Tensor<T> *> &params) = 0;
  virtual void collect_gradients(std::vector<Tensor<T> *> &grads) = 0;
  virtual void update_parameters_impl(const Optimizer &optimizer) = 0;
};

// BLAS-optimized Dense/Fully Connected Layer
template <typename T = float>
class BLASDenseLayer : public ParameterizedLayer<T> {
private:
  size_t input_features_;
  size_t output_features_;
  bool use_bias_;
  std::unique_ptr<ActivationFunction<T>> activation_;
  Tensor<T> weights_;
  Tensor<T> bias_;
  Tensor<T> weight_gradients_;
  Tensor<T> bias_gradients_;

  // Per-micro-batch state
  std::unordered_map<int, Tensor<T>> micro_batch_inputs_;
  std::unordered_map<int, Tensor<T>> micro_batch_pre_activations_;

  // BLAS helper functions
  void gemm_forward(const T *input_data, const T *weight_data, T *output_data,
                    size_t batch_size, size_t input_features,
                    size_t output_features) const {
#if defined(USE_OPENBLAS) || defined(USE_MKL) || defined(USE_ATLAS)
    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size,
                  output_features, input_features, 1.0f, input_data,
                  input_features, weight_data, input_features, 0.0f,
                  output_data, output_features);
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size,
                  output_features, input_features, 1.0, input_data,
                  input_features, weight_data, input_features, 0.0, output_data,
                  output_features);
    } else {
      fallback_gemm(input_data, weight_data, output_data, batch_size,
                    input_features, output_features);
    } 
#else
      fallback_gemm(input_data, weight_data, output_data, batch_size,
                    input_features, output_features);
#endif
  }

  void gemm_weight_gradients(const T *input_data, const T *grad_output_data,
                             T *weight_grad_data, size_t batch_size,
                             size_t input_features,
                             size_t output_features) const {
#if defined(USE_OPENBLAS) || defined(USE_MKL) || defined(USE_ATLAS)
    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_features,
                  input_features, batch_size, 1.0f, grad_output_data,
                  output_features, input_data, input_features, 0.0f,
                  weight_grad_data, input_features);
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, output_features,
                  input_features, batch_size, 1.0, grad_output_data,
                  output_features, input_data, input_features, 0.0,
                  weight_grad_data, input_features);
    } else {
      fallback_weight_gradients(input_data, grad_output_data, weight_grad_data,
                                batch_size, input_features, output_features);
    }
#else
    fallback_weight_gradients(input_data, grad_output_data, weight_grad_data,
                              batch_size, input_features, output_features);
#endif
  }

  void gemm_input_gradients(const T *grad_output_data, const T *weight_data,
                            T *grad_input_data, size_t batch_size,
                            size_t input_features,
                            size_t output_features) const {
#if defined(USE_OPENBLAS) || defined(USE_MKL) || defined(USE_ATLAS)
    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size,
                  input_features, output_features, 1.0f, grad_output_data,
                  output_features, weight_data, input_features, 0.0f,
                  grad_input_data, input_features);
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size,
                  input_features, output_features, 1.0, grad_output_data,
                  output_features, weight_data, input_features, 0.0,
                  grad_input_data, input_features);
    } else {
      fallback_input_gradients(grad_output_data, weight_data, grad_input_data,
                               batch_size, input_features, output_features);
    }
#else
    fallback_input_gradients(grad_output_data, weight_data, grad_input_data,
                             batch_size, input_features, output_features);
#endif
  }

  void add_bias_vector(T *output_data, const T *bias_data, size_t batch_size,
                       size_t output_features) const {
#if defined(USE_OPENBLAS) || defined(USE_MKL) || defined(USE_ATLAS)
    if constexpr (std::is_same_v<T, float>) {
      for (size_t n = 0; n < batch_size; ++n) {
        cblas_saxpy(output_features, 1.0f, bias_data, 1,
                    output_data + n * output_features, 1);
      }
    } else if constexpr (std::is_same_v<T, double>) {
      for (size_t n = 0; n < batch_size; ++n) {
        cblas_daxpy(output_features, 1.0, bias_data, 1,
                    output_data + n * output_features, 1);
      }
    } else {
      fallback_add_bias(output_data, bias_data, batch_size, output_features);
    }
#else
    fallback_add_bias(output_data, bias_data, batch_size, output_features);
#endif
  }

  // Fallback implementations when BLAS is not available
  void fallback_gemm(const T *input_data, const T *weight_data, T *output_data,
                     size_t batch_size, size_t input_features,
                     size_t output_features) const {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t out_f = 0; out_f < output_features; ++out_f) {
        T sum = T(0);
        for (size_t in_f = 0; in_f < input_features; ++in_f) {
          sum += input_data[n * input_features + in_f] *
                 weight_data[out_f * input_features + in_f];
        }
        output_data[n * output_features + out_f] = sum;
      }
    }
  }

  void fallback_weight_gradients(const T *input_data, const T *grad_output_data,
                                 T *weight_grad_data, size_t batch_size,
                                 size_t input_features,
                                 size_t output_features) const {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t out_f = 0; out_f < output_features; ++out_f) {
      for (size_t in_f = 0; in_f < input_features; ++in_f) {
        T sum = T(0);
        for (size_t n = 0; n < batch_size; ++n) {
          sum += grad_output_data[n * output_features + out_f] *
                 input_data[n * input_features + in_f];
        }
        weight_grad_data[out_f * input_features + in_f] = sum;
      }
    }
  }

  void fallback_input_gradients(const T *grad_output_data, const T *weight_data,
                                T *grad_input_data, size_t batch_size,
                                size_t input_features,
                                size_t output_features) const {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t in_f = 0; in_f < input_features; ++in_f) {
        T sum = T(0);
        for (size_t out_f = 0; out_f < output_features; ++out_f) {
          sum += grad_output_data[n * output_features + out_f] *
                 weight_data[out_f * input_features + in_f];
        }
        grad_input_data[n * input_features + in_f] = sum;
      }
    }
  }

  void fallback_add_bias(T *output_data, const T *bias_data, size_t batch_size,
                         size_t output_features) const {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t out_f = 0; out_f < output_features; ++out_f) {
        output_data[n * output_features + out_f] += bias_data[out_f];
      }
    }
  }

public:
  BLASDenseLayer(size_t input_features, size_t output_features,
                 std::unique_ptr<ActivationFunction<T>> activation = nullptr,
                 bool use_bias = true, const std::string &name = "blas_dense")
      : ParameterizedLayer<T>(name), input_features_(input_features),
        output_features_(output_features), use_bias_(use_bias),
        activation_(std::move(activation)) {
    weights_ =
        Tensor<T>(std::vector<size_t>{output_features, input_features, 1, 1});
    weight_gradients_ =
        Tensor<T>(std::vector<size_t>{output_features, input_features, 1, 1});

    if (use_bias_) {
      bias_ = Tensor<T>(std::vector<size_t>{output_features, 1, 1, 1});
      bias_gradients_ =
          Tensor<T>(std::vector<size_t>{output_features, 1, 1, 1});
      bias_.fill(T(0));
    }

    // Xavier initialization
    T fan_in = static_cast<T>(input_features);
    T fan_out = static_cast<T>(output_features);
    T std_dev = std::sqrt(T(2.0) / (fan_in + fan_out));
    weights_.fill_random_normal(std_dev);
  }

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    micro_batch_inputs_[micro_batch_id] = input;

    size_t batch_size = input.batch_size();
    size_t total_input_features =
        input.channels() * input.height() * input.width();

    if (total_input_features != input_features_) {
      printf("Input shape: %zu features, expected: %zu features\n",
             total_input_features, input_features_);
      throw std::invalid_argument(
          "Input feature size mismatch in BLASDenseLayer");
    }

    Tensor<T> output(std::vector<size_t>{batch_size, output_features_, 1, 1});

    // Get flattened input data for BLAS operations
    std::vector<T> input_flat = input.to_rm_vector();
    std::vector<T> output_flat(batch_size * output_features_);

    // Get weight data in contiguous format
    std::vector<T> weight_flat = weights_.to_rm_vector();

    // Perform BLAS matrix multiplication
    gemm_forward(input_flat.data(), weight_flat.data(), output_flat.data(),
                 batch_size, input_features_, output_features_);

    // Add bias using BLAS if available
    if (use_bias_) {
      std::vector<T> bias_flat = bias_.to_rm_vector();
      add_bias_vector(output_flat.data(), bias_flat.data(), batch_size,
                      output_features_);
    }

    // Convert back to tensor format
    output.from_rm_vector(output_flat);

    // Store pre-activation output
    micro_batch_pre_activations_[micro_batch_id] = output;

    // Apply activation if provided
    if (activation_) {
      activation_->apply(output);
    }

    return output;
  }

  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override {
    auto it_input = micro_batch_inputs_.find(micro_batch_id);
    auto it_pre_act = micro_batch_pre_activations_.find(micro_batch_id);

    if (it_input == micro_batch_inputs_.end()) {
      throw std::runtime_error("No cached input found for micro-batch ID: " +
                               std::to_string(micro_batch_id));
    }
    if (activation_ && it_pre_act == micro_batch_pre_activations_.end()) {
      throw std::runtime_error(
          "No cached pre-activation output found for micro-batch ID: " +
          std::to_string(micro_batch_id));
    }

    const Tensor<T> &last_input = it_input->second;
    size_t batch_size = last_input.batch_size();
    Tensor<T> grad_input(last_input.shape());

    Tensor<T> current_grad = grad_output;

    // Backprop through activation
    if (activation_) {
      Tensor<T> activation_grad =
          activation_->compute_gradient(it_pre_act->second, &current_grad);
      current_grad = activation_grad;
    }

    // Compute weight gradients using BLAS
    gemm_weight_gradients(last_input.data(), current_grad.data(),
                          weight_gradients_.data(), batch_size, input_features_,
                          output_features_);

    // Compute bias gradients
    if (use_bias_) {
      bias_gradients_.fill(T(0));
      for (size_t out_f = 0; out_f < output_features_; ++out_f) {
        T grad_sum = T(0);
        for (size_t n = 0; n < batch_size; ++n) {
          grad_sum += current_grad(n, out_f, 0, 0);
        }
        bias_gradients_(out_f, 0, 0, 0) = grad_sum;
      }
    }

    // Compute input gradients using BLAS
    gemm_input_gradients(current_grad.data(), weights_.data(),
                         grad_input.data(), batch_size, input_features_,
                         output_features_);

    // Clean up cached data for this micro-batch
    micro_batch_inputs_.erase(it_input);
    if (activation_) {
      micro_batch_pre_activations_.erase(it_pre_act);
    }

    return grad_input;
  }

  std::string type() const override { return "blas_dense"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["input_features"] = input_features_;
    config.parameters["output_features"] = output_features_;
    config.parameters["use_bias"] = use_bias_;
    config.parameters["activation"] =
        activation_ ? activation_->name() : std::string("none");
    config.parameters["optimized"] = std::string("blas");
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    auto activation_clone = activation_ ? activation_->clone() : nullptr;
    return std::make_unique<BLASDenseLayer<T>>(
        input_features_, output_features_, std::move(activation_clone),
        use_bias_, this->name_);
  }

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override {
    if (input_shape.size() != 4) {
      throw std::invalid_argument("BLASDenseLayer expects 4D input");
    }
    return {input_shape[0], output_features_, 1, 1};
  }

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override {
    params.push_back(&weights_);
    if (use_bias_) {
      params.push_back(&bias_);
    }
  }

  void collect_gradients(std::vector<Tensor<T> *> &grads) override {
    grads.push_back(&weight_gradients_);
    if (use_bias_) {
      grads.push_back(&bias_gradients_);
    }
  }

  void update_parameters_impl(const Optimizer &optimizer) override {
    // To be implemented with optimizer interface
  }

  static std::unique_ptr<Layer<T>>
  create_from_config(const LayerConfig &config) {
    size_t input_features = config.get<size_t>("input_features");
    size_t output_features = config.get<size_t>("output_features");
    bool use_bias = config.get<bool>("use_bias");
    std::string activation_name = config.get<std::string>("activation");

    std::unique_ptr<ActivationFunction<T>> activation;
    if (activation_name != "none") {
      // Ensure factory has default activations registered
      ActivationFactory<T>::register_defaults();
      activation = ActivationFactory<T>::create(activation_name);
    }

    return std::make_unique<BLASDenseLayer<T>>(input_features, output_features,
                                               std::move(activation), use_bias,
                                               config.name);
  }
};

// Activation Layer (stateless)
template <typename T = double>
class ActivationLayer : public StatelessLayer<T> {
private:
  std::unique_ptr<ActivationFunction<T>> activation_;
  std::unordered_map<int, Tensor<T>> micro_batch_inputs_;

public:
  explicit ActivationLayer(std::unique_ptr<ActivationFunction<T>> activation,
                           const std::string &name = "activation")
      : StatelessLayer<T>(name), activation_(std::move(activation)) {
    if (!activation_) {
      throw std::invalid_argument("Activation function cannot be null");
    }
  }

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    micro_batch_inputs_[micro_batch_id] = input;
    Tensor<T> output = input; // Copy
    activation_->apply(output);
    return output;
  }

  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override {
    auto it = micro_batch_inputs_.find(micro_batch_id);
    if (it == micro_batch_inputs_.end()) {
      throw std::runtime_error(
          "No cached input found for micro-batch ID in ActivationLayer: " +
          std::to_string(micro_batch_id));
    }
    const Tensor<T> &last_input = it->second;
    Tensor<T> grad = activation_->compute_gradient(last_input, &grad_output);
    micro_batch_inputs_.erase(it);
    return grad;
  }

  std::string type() const override { return "activation"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["activation"] = activation_->name();
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<ActivationLayer<T>>(activation_->clone(),
                                                this->name_);
  }

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override {
    return input_shape; // Activation doesn't change shape
  }
};

// BLAS-optimized 2D Convolutional Layer using im2col + GEMM
template <typename T = double>
class BLASConv2DLayer : public ParameterizedLayer<T> {
private:
  size_t in_channels_;
  size_t out_channels_;
  size_t kernel_h_;
  size_t kernel_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;
  bool use_bias_;
  std::unique_ptr<ActivationFunction<T>> activation_;

  Tensor<T> weights_;          // [out_channels, in_channels, kernel_h, kernel_w]
  Tensor<T> bias_;             // [out_channels, 1, 1, 1]
  Tensor<T> weight_gradients_; // Same shape as weights
  Tensor<T> bias_gradients_;   // Same shape as bias

  // Per-micro-batch state
  std::unordered_map<int, Tensor<T>> micro_batch_inputs_;
  std::unordered_map<int, Tensor<T>> micro_batch_pre_activations_;
  mutable std::unordered_map<int, Matrix<T>> micro_batch_im2col_matrices_;

  // BLAS helper functions for convolution
  void conv_gemm_forward(const T *col_data, const T *weight_data,
                         T *output_data, size_t output_size, size_t kernel_size,
                         size_t out_channels) const {
#if defined(USE_OPENBLAS) || defined(USE_MKL) || defined(USE_ATLAS)
    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_channels,
                  output_size, kernel_size, 1.0f, weight_data, kernel_size,
                  col_data, output_size, 0.0f, output_data, output_size);
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, out_channels,
                  output_size, kernel_size, 1.0, weight_data, kernel_size,
                  col_data, output_size, 0.0, output_data, output_size);
    } else {
      fallback_conv_gemm_forward(col_data, weight_data, output_data,
                                 output_size, kernel_size, out_channels);
    }
#else
    fallback_conv_gemm_forward(col_data, weight_data, output_data,
                               output_size, kernel_size, out_channels);
#endif
  }

  void conv_gemm_weight_gradients(const T *col_data, const T *grad_output_data,
                                  T *weight_grad_data, size_t output_size,
                                  size_t kernel_size,
                                  size_t out_channels) const {
#if defined(USE_OPENBLAS) || defined(USE_MKL) || defined(USE_ATLAS)
    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, out_channels,
                  kernel_size, output_size, 1.0f, grad_output_data, output_size,
                  col_data, output_size, 0.0f, weight_grad_data, kernel_size);
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, out_channels,
                  kernel_size, output_size, 1.0, grad_output_data, output_size,
                  col_data, output_size, 0.0, weight_grad_data, kernel_size);
    } else {
      fallback_conv_gemm_weight_gradients(col_data, grad_output_data,
                                          weight_grad_data, output_size,
                                          kernel_size, out_channels);
    }
#else
    fallback_conv_gemm_weight_gradients(col_data, grad_output_data,
                                        weight_grad_data, output_size,
                                        kernel_size, out_channels);
#endif
  }

  void conv_gemm_input_gradients(const T *grad_output_data,
                                 const T *weight_data, T *col_grad_data,
                                 size_t output_size, size_t kernel_size,
                                 size_t out_channels) const {
#if defined(USE_OPENBLAS) || defined(USE_MKL) || defined(USE_ATLAS)
    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, kernel_size,
                  output_size, out_channels, 1.0f, weight_data, kernel_size,
                  grad_output_data, output_size, 0.0f, col_grad_data,
                  output_size);
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, kernel_size,
                  output_size, out_channels, 1.0, weight_data, kernel_size,
                  grad_output_data, output_size, 0.0, col_grad_data,
                  output_size);
    } else {
      fallback_conv_gemm_input_gradients(grad_output_data, weight_data,
                                         col_grad_data, output_size,
                                         kernel_size, out_channels);
    }
#else
    fallback_conv_gemm_input_gradients(grad_output_data, weight_data,
                                       col_grad_data, output_size,
                                       kernel_size, out_channels);
#endif
  }

  // Fallback implementations when BLAS is not available
  void fallback_conv_gemm_forward(const T *col_data, const T *weight_data,
                                  T *output_data, size_t output_size,
                                  size_t kernel_size,
                                  size_t out_channels) const {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t oc = 0; oc < out_channels; ++oc) {
      for (size_t os = 0; os < output_size; ++os) {
        T sum = T(0);
        for (size_t ks = 0; ks < kernel_size; ++ks) {
          sum += weight_data[oc * kernel_size + ks] *
                 col_data[ks * output_size + os];
        }
        output_data[oc * output_size + os] = sum;
      }
    }
  }

  void fallback_conv_gemm_weight_gradients(
      const T *col_data, const T *grad_output_data, T *weight_grad_data,
      size_t output_size, size_t kernel_size, size_t out_channels) const {
    // Add profiling information
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t oc = 0; oc < out_channels; ++oc) {
      for (size_t ks = 0; ks < kernel_size; ++ks) {
        T sum = T(0);
        for (size_t os = 0; os < output_size; ++os) {
          sum += grad_output_data[oc * output_size + os] *
                 col_data[ks * output_size + os];
        }
        weight_grad_data[oc * kernel_size + ks] = sum;
      }
    }
  }

  void fallback_conv_gemm_input_gradients(const T *grad_output_data,
                                          const T *weight_data,
                                          T *col_grad_data, size_t output_size,
                                          size_t kernel_size,
                                          size_t out_channels) const {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t ks = 0; ks < kernel_size; ++ks) {
      for (size_t os = 0; os < output_size; ++os) {
        T sum = T(0);
        for (size_t oc = 0; oc < out_channels; ++oc) {
          sum += weight_data[oc * kernel_size + ks] *
                 grad_output_data[oc * output_size + os];
        }
        col_grad_data[ks * output_size + os] = sum;
      }
    }
  }

  // // Helper to reshape weights for BLAS operations
  // std::vector<T> get_flattened_weights() const {
  //   std::vector<T> flattened = weights_.to_rm_vector();
  //   return flattened;
  // }

  void set_flattened_weight_gradients(const std::vector<T> &flattened) {
    weight_gradients_.from_rm_vector(flattened);
  }

public:
  BLASConv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h,
                  size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
                  size_t pad_h = 0, size_t pad_w = 0, bool use_bias = true,
                  std::unique_ptr<ActivationFunction<T>> activation = nullptr,
                  const std::string &name = "blas_conv2d")
      : ParameterizedLayer<T>(name), in_channels_(in_channels),
        out_channels_(out_channels), kernel_h_(kernel_h), kernel_w_(kernel_w),
        stride_h_(stride_h), stride_w_(stride_w), pad_h_(pad_h), pad_w_(pad_w),
        use_bias_(use_bias), activation_(std::move(activation)),
        micro_batch_im2col_matrices_() {
    weights_ = Tensor<T>(
        std::vector<size_t>{out_channels, in_channels, kernel_h, kernel_w});
    weight_gradients_ = Tensor<T>(
        std::vector<size_t>{out_channels, in_channels, kernel_h, kernel_w});

    if (use_bias_) {
      bias_ = Tensor<T>(std::vector<size_t>{out_channels, 1, 1, 1});
      bias_gradients_ = Tensor<T>(std::vector<size_t>{out_channels, 1, 1, 1});
      bias_.fill(0.0);
    }

    // Xavier/Glorot initialization
    T fan_in = static_cast<T>(in_channels * kernel_h * kernel_w);
    T fan_out = static_cast<T>(out_channels * kernel_h * kernel_w);
    T std_dev = std::sqrt(T(2.0) / (fan_in + fan_out));
    weights_.fill_random_normal(std_dev);
  }

  // Forward and backward implementations moved from separate file
  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    if (input.channels() != in_channels_) {
      printf("Input shape: %zu channels, expected: %zu channels\n",
             input.channels(), in_channels_);
      throw std::invalid_argument(
          "Input channel size mismatch in BLASConv2DLayer");
    }

    micro_batch_inputs_[micro_batch_id] = input;

    size_t batch_size = input.batch_size();
    size_t input_h = input.height();
    size_t input_w = input.width();

    // Calculate output dimensions
    size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

    // Perform im2col transformation
    Matrix<T> col_matrix = input.im2col(kernel_h_, kernel_w_, stride_h_,
                                        stride_w_, pad_h_, pad_w_);
    micro_batch_im2col_matrices_[micro_batch_id] =
        col_matrix; // Cache for backward pass

    // Create output tensor
    Tensor<T> output(batch_size, out_channels_, output_h, output_w);

    // Get flattened weights for BLAS
    std::vector<T> weight_flat = weights_.to_rm_vector();

    // Convert col_matrix to contiguous format for BLAS
    std::vector<T> col_data = col_matrix.to_vector();

    // Prepare data for BLAS operation
    size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
    size_t output_size = batch_size * output_h * output_w;

    // Perform convolution using BLAS GEMM
    std::vector<T> output_flat(out_channels_ * output_size);
    conv_gemm_forward(col_data.data(), weight_flat.data(), output_flat.data(),
                      output_size, kernel_size, out_channels_);

    // Reshape output back to tensor format
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t oc = 0; oc < out_channels_; ++oc) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            size_t flat_idx = oc * output_size + n * (output_h * output_w) +
                              oh * output_w + ow;
            output(n, oc, oh, ow) = output_flat[flat_idx];
          }
        }
      }
    }

    // Add bias if enabled
    if (use_bias_) {
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t oc = 0; oc < out_channels_; ++oc) {
          T bias_val = bias_(oc, 0, 0, 0);
          for (size_t oh = 0; oh < output_h; ++oh) {
            for (size_t ow = 0; ow < output_w; ++ow) {
              output(n, oc, oh, ow) += bias_val;
            }
          }
        }
      }
    }

    // Store pre-activation output
    micro_batch_pre_activations_[micro_batch_id] = output;

    // Apply activation if provided
    if (activation_) {
      activation_->apply(output);
    }

    return output;
  }

  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override {
    auto it_input = micro_batch_inputs_.find(micro_batch_id);
    auto it_pre_act = micro_batch_pre_activations_.find(micro_batch_id);
    auto it_im2col = micro_batch_im2col_matrices_.find(micro_batch_id);

    if (it_input == micro_batch_inputs_.end()) {
      throw std::runtime_error("No cached input found for micro-batch ID: " +
                               std::to_string(micro_batch_id));
    }
    if (it_im2col == micro_batch_im2col_matrices_.end()) {
      throw std::runtime_error(
          "No cached im2col matrix found for micro-batch ID: " +
          std::to_string(micro_batch_id));
    }
    if (activation_ && it_pre_act == micro_batch_pre_activations_.end()) {
      throw std::runtime_error(
          "No cached pre-activation output found for micro-batch ID: " +
          std::to_string(micro_batch_id));
    }

    const Tensor<T> &last_input = it_input->second;
    const Matrix<T> &cached_im2col_matrix = it_im2col->second;

    size_t batch_size = last_input.batch_size();
    size_t input_h = last_input.height();
    size_t input_w = last_input.width();
    size_t output_h = grad_output.height();
    size_t output_w = grad_output.width();

    Tensor<T> current_grad = grad_output;

    // Backprop through activation
    if (activation_) {
      current_grad =
          activation_->compute_gradient(it_pre_act->second, &current_grad);
    }

    // Initialize gradients
    weight_gradients_.fill(T(0));
    if (use_bias_) {
      bias_gradients_.fill(T(0));
    }

    // Prepare data for BLAS operations
    size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
    size_t output_size = batch_size * output_h * output_w;

    // Flatten gradient output for BLAS
    std::vector<T> grad_output_flat(out_channels_ * output_size);

#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t oc = 0; oc < out_channels_; ++oc) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            grad_output_flat[oc * output_size + n * (output_h * output_w) +
                             oh * output_w + ow] = current_grad(n, oc, oh, ow);
          }
        }
      }
    }

    // Convert cached im2col matrix to contiguous format
    std::vector<T> col_data(kernel_size * output_size);
    for (size_t i = 0; i < kernel_size; ++i) {
      for (size_t j = 0; j < output_size; ++j) {
        col_data[i * output_size + j] = cached_im2col_matrix(i, j);
      }
    }

    // Compute weight gradients using BLAS
    std::vector<T> weight_grad_flat(out_channels_ * kernel_size);
    conv_gemm_weight_gradients(col_data.data(), grad_output_flat.data(),
                               weight_grad_flat.data(), output_size,
                               kernel_size, out_channels_);

    // Set weight gradients back to tensor format
    set_flattened_weight_gradients(weight_grad_flat);

    // Compute bias gradients
    if (use_bias_) {
      for (size_t oc = 0; oc < out_channels_; ++oc) {
        T grad_sum = T(0);
        for (size_t n = 0; n < batch_size; ++n) {
          for (size_t oh = 0; oh < output_h; ++oh) {
            for (size_t ow = 0; ow < output_w; ++ow) {
              grad_sum += current_grad(n, oc, oh, ow);
            }
          }
        }
        bias_gradients_(oc, 0, 0, 0) = grad_sum;
      }
    }

    // Compute input gradients using BLAS
    std::vector<T> weight_flat = weights_.to_rm_vector();
    std::vector<T> col_grad_flat(kernel_size * output_size);
    conv_gemm_input_gradients(grad_output_flat.data(), weight_flat.data(),
                              col_grad_flat.data(), output_size, kernel_size,
                              out_channels_);

    // Convert col_grad back to Matrix format
    Matrix<T> col_grad_matrix(kernel_size, output_size);
    for (size_t i = 0; i < kernel_size; ++i) {
      for (size_t j = 0; j < output_size; ++j) {
        col_grad_matrix(i, j) = col_grad_flat[i * output_size + j];
      }
    }

    // Use col2im to convert back to input gradient tensor
    Tensor<T> grad_input = Tensor<T>::col2im(
        col_grad_matrix, batch_size, in_channels_, input_h, input_w, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);

    // Clean up cache
    micro_batch_inputs_.erase(it_input);
    micro_batch_im2col_matrices_.erase(it_im2col);
    if (activation_) {
      micro_batch_pre_activations_.erase(it_pre_act);
    }

    return grad_input;
  }

  std::string type() const override { return "blas_conv2d"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["in_channels"] = in_channels_;
    config.parameters["out_channels"] = out_channels_;
    config.parameters["kernel_h"] = kernel_h_;
    config.parameters["kernel_w"] = kernel_w_;
    config.parameters["stride_h"] = stride_h_;
    config.parameters["stride_w"] = stride_w_;
    config.parameters["pad_h"] = pad_h_;
    config.parameters["pad_w"] = pad_w_;
    config.parameters["use_bias"] = use_bias_;
    config.parameters["activation"] =
        activation_ ? activation_->name() : std::string("none");
    config.parameters["optimized"] = std::string("blas");
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    auto activation_clone = activation_ ? activation_->clone() : nullptr;
    return std::make_unique<BLASConv2DLayer<T>>(
        in_channels_, out_channels_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, use_bias_, std::move(activation_clone), this->name_);
  }

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override {
    if (input_shape.size() != 4) {
      throw std::invalid_argument("BLASConv2DLayer expects 4D input");
    }

    size_t output_h = (input_shape[2] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    size_t output_w = (input_shape[3] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

    return {input_shape[0], out_channels_, output_h, output_w};
  }

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override {
    params.push_back(&weights_);
    if (use_bias_) {
      params.push_back(&bias_);
    }
  }

  void collect_gradients(std::vector<Tensor<T> *> &grads) override {
    grads.push_back(&weight_gradients_);
    if (use_bias_) {
      grads.push_back(&bias_gradients_);
    }
  }

  void update_parameters_impl(const Optimizer &optimizer) override {
    // To be implemented with optimizer interface
  }
};

// MaxPooling Layer for 2D tensors
template <typename T = double> class MaxPool2DLayer : public StatelessLayer<T> {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  std::unordered_map<int, Tensor<T>> micro_batch_masks_;
  std::unordered_map<int, Tensor<T>> micro_batch_inputs_;

public:
  MaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h = 0,
                 size_t stride_w = 0, size_t pad_h = 0, size_t pad_w = 0,
                 const std::string &name = "maxpool2d")
      : StatelessLayer<T>(name), pool_h_(pool_h), pool_w_(pool_w),
        stride_h_(stride_h == 0 ? pool_h : stride_h),
        stride_w_(stride_w == 0 ? pool_w : stride_w), pad_h_(pad_h),
        pad_w_(pad_w) {}

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    // Store input for backward pass
    micro_batch_inputs_[micro_batch_id] = input;

    size_t batch_size = input.batch_size();
    size_t channels = input.channels();
    size_t input_h = input.height();
    size_t input_w = input.width();

    // Calculate output dimensions
    size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
    size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

    // Create output tensor
    Tensor<T> output(
        std::vector<size_t>{batch_size, channels, output_h, output_w});

    // Create mask for backward pass (store which input position had max value)
    Tensor<T> mask(
        std::vector<size_t>{batch_size, channels, output_h, output_w});

    // Apply padding if needed
    Tensor<T> padded_input =
        (pad_h_ > 0 || pad_w_ > 0)
            ? input.pad(pad_h_, pad_w_, -std::numeric_limits<T>::infinity())
            : input;

    // Perform max pooling
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        for (size_t out_h = 0; out_h < output_h; ++out_h) {
          for (size_t out_w = 0; out_w < output_w; ++out_w) {
            T max_val = -std::numeric_limits<T>::infinity();
            size_t max_h = 0, max_w = 0;

            // Find maximum in pooling window
            for (size_t ph = 0; ph < pool_h_; ++ph) {
              for (size_t pw = 0; pw < pool_w_; ++pw) {
                size_t input_h_idx = out_h * stride_h_ + ph;
                size_t input_w_idx = out_w * stride_w_ + pw;

                if (input_h_idx < padded_input.height() &&
                    input_w_idx < padded_input.width()) {
                  T val = padded_input(n, c, input_h_idx, input_w_idx);
                  if (val > max_val) {
                    max_val = val;
                    max_h = input_h_idx;
                    max_w = input_w_idx;
                  }
                }
              }
            }

            output(n, c, out_h, out_w) = max_val;
            // Store flattened index of max position for backward pass
            mask(n, c, out_h, out_w) =
                static_cast<T>(max_h * padded_input.width() + max_w);
          }
        }
      }
    }

    micro_batch_masks_[micro_batch_id] = mask;
    return output;
  }

  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override {
    auto it_input = micro_batch_inputs_.find(micro_batch_id);
    auto it_mask = micro_batch_masks_.find(micro_batch_id);

    if (it_input == micro_batch_inputs_.end()) {
      throw std::runtime_error(
          "No cached input found for micro-batch ID in MaxPool2DLayer: " +
          std::to_string(micro_batch_id));
    }
    if (it_mask == micro_batch_masks_.end()) {
      throw std::runtime_error(
          "No cached mask found for micro-batch ID in MaxPool2DLayer: " +
          std::to_string(micro_batch_id));
    }

    const Tensor<T> &last_input = it_input->second;
    const Tensor<T> &mask = it_mask->second;

    // Use stored input dimensions instead of trying to calculate them
    size_t batch_size = last_input.batch_size();
    size_t channels = last_input.channels();
    size_t input_h = last_input.height();
    size_t input_w = last_input.width();
    size_t output_h = grad_output.height();
    size_t output_w = grad_output.width();

    Tensor<T> grad_input(
        std::vector<size_t>{batch_size, channels, input_h, input_w});
    grad_input.fill(0.0);

    // Create padded gradient input if padding was used
    Tensor<T> padded_grad_input = (pad_h_ > 0 || pad_w_ > 0)
                                      ? grad_input.pad(pad_h_, pad_w_)
                                      : grad_input;

    // Distribute gradients to max positions
#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        for (size_t out_h = 0; out_h < output_h; ++out_h) {
          for (size_t out_w = 0; out_w < output_w; ++out_w) {
            // Get the position of the max value
            size_t flat_idx = static_cast<size_t>(mask(n, c, out_h, out_w));
            size_t max_h = flat_idx / padded_grad_input.width();
            size_t max_w = flat_idx % padded_grad_input.width();

            // Add gradient to the max position
            if (max_h < padded_grad_input.height() &&
                max_w < padded_grad_input.width()) {
#ifdef _OPENMP
#pragma omp atomic
#endif
              padded_grad_input(n, c, max_h, max_w) +=
                  grad_output(n, c, out_h, out_w);
            }
          }
        }
      }
    }

    // Remove padding if it was applied
    if (pad_h_ > 0 || pad_w_ > 0) {
      grad_input = padded_grad_input.crop(
          pad_h_, pad_w_, padded_grad_input.height() - pad_h_ - 1,
          padded_grad_input.width() - pad_w_ - 1);
    } else {
      grad_input = padded_grad_input;
    }

    // Clean up cache
    micro_batch_inputs_.erase(it_input);
    micro_batch_masks_.erase(it_mask);

    return grad_input;
  }

  std::string type() const override { return "maxpool2d"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["pool_h"] = pool_h_;
    config.parameters["pool_w"] = pool_w_;
    config.parameters["stride_h"] = stride_h_;
    config.parameters["stride_w"] = stride_w_;
    config.parameters["pad_h"] = pad_h_;
    config.parameters["pad_w"] = pad_w_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<MaxPool2DLayer<T>>(
        pool_h_, pool_w_, stride_h_, stride_w_, pad_h_, pad_w_, this->name_);
  }

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override {
    if (input_shape.size() != 4) {
      throw std::invalid_argument("MaxPool2DLayer expects 4D input");
    }

    size_t output_h = (input_shape[2] + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
    size_t output_w = (input_shape[3] + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

    return {input_shape[0], input_shape[1], output_h, output_w};
  }
};

// Dropout Layer
template <typename T = double> class DropoutLayer : public StatelessLayer<T> {
private:
  T dropout_rate_;
  std::unordered_map<int, Tensor<T>> micro_batch_masks_;
  mutable std::mt19937 generator_;

public:
  explicit DropoutLayer(T dropout_rate, const std::string &name = "dropout")
      : StatelessLayer<T>(name), dropout_rate_(dropout_rate),
        generator_(std::random_device{}()) {
    if (dropout_rate < T(0) || dropout_rate >= T(1)) {
      throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
  }

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    if (!this->is_training_) {
      return input; // No dropout during inference
    }

    Tensor<T> mask(input.shape());
    Tensor<T> output = input;

    std::uniform_real_distribution<T> distribution(T(0), T(1));

    // Generate random mask with proper scaling
    T scale = T(1) / (T(1) - dropout_rate_);

#ifdef _OPENMP
#pragma omp parallel for collapse(                                             \
        2) if (input.batch_size() * input.channels() > 1000)
#endif
    for (size_t n = 0; n < input.batch_size(); ++n) {
      for (size_t c = 0; c < input.channels(); ++c) {
#ifdef _OPENMP
        thread_local std::mt19937 local_generator(std::random_device{}());
        thread_local std::uniform_real_distribution<T> local_distribution(T(0),
                                                                          T(1));
#else
        std::uniform_real_distribution<T> local_distribution(T(0), T(1));
#endif
        for (size_t h = 0; h < input.height(); ++h) {
          for (size_t w = 0; w < input.width(); ++w) {
#ifdef _OPENMP
            if (local_distribution(local_generator) < dropout_rate_) {
#else
            if (local_distribution(generator_) < dropout_rate_) {
#endif
              mask(n, c, h, w) = T(0);
              output(n, c, h, w) = T(0);
            } else {
              mask(n, c, h, w) = scale;
              output(n, c, h, w) *= scale;
            }
          }
        }
      }
    }

    micro_batch_masks_[micro_batch_id] = mask;
    return output;
  }

  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override {
    if (!this->is_training_) {
      return grad_output;
    }

    auto it_mask = micro_batch_masks_.find(micro_batch_id);
    if (it_mask == micro_batch_masks_.end()) {
      throw std::runtime_error(
          "No cached mask found for micro-batch ID in DropoutLayer: " +
          std::to_string(micro_batch_id));
    }
    const Tensor<T> &mask = it_mask->second;

    Tensor<T> grad_input = grad_output;

#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
    for (size_t n = 0; n < grad_output.batch_size(); ++n) {
      for (size_t c = 0; c < grad_output.channels(); ++c) {
        for (size_t h = 0; h < grad_output.height(); ++h) {
          for (size_t w = 0; w < grad_output.width(); ++w) {
            grad_input(n, c, h, w) *= mask(n, c, h, w);
          }
        }
      }
    }

    micro_batch_masks_.erase(it_mask);
    return grad_input;
  }

  std::string type() const override { return "dropout"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["dropout_rate"] = dropout_rate_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<DropoutLayer<T>>(dropout_rate_, this->name_);
  }

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override {
    return input_shape; // Dropout doesn't change shape
  }
};

// Flatten Layer for converting from 4D to 2D tensors (for compatibility with
// dense layers)
template <typename T = double> class FlattenLayer : public StatelessLayer<T> {
private:
  std::unordered_map<int, std::vector<size_t>> micro_batch_original_shapes_;

public:
  explicit FlattenLayer(const std::string &name = "flatten")
      : StatelessLayer<T>(name) {}

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    micro_batch_original_shapes_[micro_batch_id] = input.shape();

    size_t batch_size = input.batch_size();
    size_t features = input.channels() * input.height() * input.width();

    // Create flattened output: [batch_size, features, 1, 1]
    Tensor<T> output(std::vector<size_t>{batch_size, features, 1, 1});

    // Copy data in CHW order
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      size_t feature_idx = 0;
      for (size_t c = 0; c < input.channels(); ++c) {
        for (size_t h = 0; h < input.height(); ++h) {
          for (size_t w = 0; w < input.width(); ++w) {
            output(n, feature_idx, 0, 0) = input(n, c, h, w);
            feature_idx++;
          }
        }
      }
    }

    return output;
  }

  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override {
    auto it = micro_batch_original_shapes_.find(micro_batch_id);
    if (it == micro_batch_original_shapes_.end()) {
      throw std::runtime_error(
          "No cached shape found for micro-batch ID in FlattenLayer: " +
          std::to_string(micro_batch_id));
    }
    const std::vector<size_t> &original_shape = it->second;
    // Reshape back to original shape
    Tensor<T> grad_input(original_shape);

    size_t batch_size = grad_input.batch_size();

    // Copy data back to CHW order
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      size_t feature_idx = 0;
      for (size_t c = 0; c < grad_input.channels(); ++c) {
        for (size_t h = 0; h < grad_input.height(); ++h) {
          for (size_t w = 0; w < grad_input.width(); ++w) {
            grad_input(n, c, h, w) = grad_output(n, feature_idx, 0, 0);
            feature_idx++;
          }
        }
      }
    }

    micro_batch_original_shapes_.erase(it);
    return grad_input;
  }

  std::string type() const override { return "flatten"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<FlattenLayer<T>>(this->name_);
  }

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override {
    if (input_shape.size() != 4) {
      throw std::invalid_argument("FlattenLayer expects 4D input");
    }

    size_t features = input_shape[1] * input_shape[2] * input_shape[3];
    return {input_shape[0], features, 1, 1};
  }
};

// Layer Factory
template <typename T = double> class LayerFactory {
private:
  static std::unordered_map<
      std::string,
      std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)>>
      creators_;

public:
  static void register_layer(
      const std::string &type,
      std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)> creator) {
    creators_[type] = creator;
  }

  static std::unique_ptr<Layer<T>> create(const std::string &type,
                                          const LayerConfig &config) {
    auto it = creators_.find(type);
    if (it != creators_.end()) {
      return it->second(config);
    }
    throw std::invalid_argument("Unknown layer type: " + type);
  }

  static std::unique_ptr<Layer<T>> create(const LayerConfig &config) {
    return create(config.get<std::string>("type"), config);
  }

  static void register_defaults() {
    // BLAS-optimized Dense layer
    register_layer(
        "blas_dense",
        [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
          size_t input_features = config.get<size_t>("input_features");
          size_t output_features = config.get<size_t>("output_features");
          bool use_bias = config.get<bool>("use_bias", true);
          std::string activation_name =
              config.get<std::string>("activation", "none");

          std::unique_ptr<ActivationFunction<T>> activation = nullptr;
          if (activation_name != "none") {
            auto factory = ActivationFactory<T>();
            factory.register_defaults();
            activation = factory.create(activation_name);
          }

          return std::make_unique<BLASDenseLayer<T>>(
              input_features, output_features, std::move(activation), use_bias,
              config.name);
        });

    // BLAS-optimized Conv2D layer
    register_layer(
        "blas_conv2d",
        [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
          size_t in_channels = config.get<size_t>("in_channels");
          size_t out_channels = config.get<size_t>("out_channels");
          size_t kernel_h = config.get<size_t>("kernel_h");
          size_t kernel_w = config.get<size_t>("kernel_w");
          size_t stride_h = config.get<size_t>("stride_h", 1);
          size_t stride_w = config.get<size_t>("stride_w", 1);
          size_t pad_h = config.get<size_t>("pad_h", 0);
          size_t pad_w = config.get<size_t>("pad_w", 0);
          bool use_bias = config.get<bool>("use_bias", true);
          std::string activation_name =
              config.get<std::string>("activation", "none");

          std::unique_ptr<ActivationFunction<T>> activation = nullptr;
          if (activation_name != "none") {
            auto factory = ActivationFactory<T>();
            factory.register_defaults();
            activation = factory.create(activation_name);
          }

          return std::make_unique<BLASConv2DLayer<T>>(
              in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
              pad_h, pad_w, use_bias, std::move(activation), config.name);
        });

    // Activation layer
    register_layer("activation",
                   [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
                     std::string activation_name =
                         config.get<std::string>("activation");
                     auto factory = ActivationFactory<T>();
                     factory.register_defaults();
                     auto activation = factory.create(activation_name);
                     if (!activation) {
                       throw std::invalid_argument(
                           "Failed to create activation: " + activation_name);
                     }
                     return std::make_unique<ActivationLayer<T>>(
                         std::move(activation), config.name);
                   });

    // MaxPool2D layer
    register_layer("maxpool2d",
                   [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
                     size_t pool_h = config.get<size_t>("pool_h");
                     size_t pool_w = config.get<size_t>("pool_w");
                     size_t stride_h = config.get<size_t>("stride_h", 0);
                     size_t stride_w = config.get<size_t>("stride_w", 0);
                     size_t pad_h = config.get<size_t>("pad_h", 0);
                     size_t pad_w = config.get<size_t>("pad_w", 0);

                     return std::make_unique<MaxPool2DLayer<T>>(
                         pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                         config.name);
                   });

    // Dropout layer
    register_layer(
        "dropout", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
          T dropout_rate = config.get<T>("dropout_rate");
          return std::make_unique<DropoutLayer<T>>(dropout_rate, config.name);
        });

    // Flatten layer
    register_layer("flatten",
                   [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
                     return std::make_unique<FlattenLayer<T>>(config.name);
                   });
  }

  static std::vector<std::string> available_types() {
    std::vector<std::string> types;
    for (const auto &pair : creators_) {
      types.push_back(pair.first);
    }
    return types;
  }
};

template <typename T>
std::unordered_map<
    std::string, std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)>>
    LayerFactory<T>::creators_;

} // namespace layers

// Convenience functions for creating layers
namespace Layers {

template <typename T = double>
std::unique_ptr<layers::Layer<T>>
blas_dense(size_t input_features, size_t output_features,
           const std::string &activation = "none", bool use_bias = true,
           const std::string &name = "blas_dense") {
  std::unique_ptr<ActivationFunction<T>> act = nullptr;
  if (activation != "none") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    act = factory.create(activation);
  }

  return std::make_unique<layers::BLASDenseLayer<T>>(
      input_features, output_features, std::move(act), use_bias, name);
}

template <typename T = double>
std::unique_ptr<layers::Layer<T>>
blas_conv2d(size_t in_channels, size_t out_channels, size_t kernel_h,
            size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
            size_t pad_h = 0, size_t pad_w = 0,
            const std::string &activation = "none", bool use_bias = true,
            const std::string &name = "blas_conv2d") {
  std::unique_ptr<ActivationFunction<T>> act = nullptr;
  if (activation != "none") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    act = factory.create(activation);
  }
  return std::make_unique<layers::BLASConv2DLayer<T>>(
      in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h,
      pad_w, use_bias, std::move(act), name);
}

template <typename T = double>
std::unique_ptr<layers::Layer<T>>
activation(const std::string &activation_name,
           const std::string &name = "activation") {
  auto factory = ActivationFactory<T>();
  factory.register_defaults();
  auto act = factory.create(activation_name);
  return std::make_unique<layers::ActivationLayer<T>>(std::move(act), name);
}

template <typename T = double>
std::unique_ptr<layers::Layer<T>>
maxpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 0,
          size_t stride_w = 0, size_t pad_h = 0, size_t pad_w = 0,
          const std::string &name = "maxpool2d") {
  return std::make_unique<layers::MaxPool2DLayer<T>>(
      pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, name);
}

template <typename T = double>
std::unique_ptr<layers::Layer<T>> dropout(T dropout_rate,
                                          const std::string &name = "dropout") {
  return std::make_unique<layers::DropoutLayer<T>>(dropout_rate, name);
}

template <typename T = double>
std::unique_ptr<layers::Layer<T>> flatten(const std::string &name = "flatten") {
  return std::make_unique<layers::FlattenLayer<T>>(name);
}
} // namespace Layers