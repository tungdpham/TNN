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
#include <type_traits>

// SIMD intrinsics
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#include <smmintrin.h>
#endif

#include "../tensor/tensor.hpp"
#include "activations.hpp"
#include "optimizers.hpp"

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
  virtual void update_parameters(const Optimizer<T> &optimizer) {}

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

  void update_parameters(const Optimizer<T> &optimizer) override {
    update_parameters_impl(optimizer);
  }

protected:
  virtual void collect_parameters(std::vector<Tensor<T> *> &params) = 0;
  virtual void collect_gradients(std::vector<Tensor<T> *> &grads) = 0;
  virtual void update_parameters_impl(const Optimizer<T> &optimizer) = 0;
};

// Dense/Fully Connected Layer
template <typename T = float> class DenseLayer : public ParameterizedLayer<T> {
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

  // Helper functions
  void gemm_forward(const T *input_data, const T *weight_data, T *output_data,
                    size_t batch_size, size_t input_features,
                    size_t output_features) const {
    gemm_impl(input_data, weight_data, output_data, batch_size, input_features,
              output_features);
  }

  void gemm_weight_gradients(const T *input_data, const T *grad_output_data,
                             T *weight_grad_data, size_t batch_size,
                             size_t input_features,
                             size_t output_features) const {
    weight_gradients_impl(input_data, grad_output_data, weight_grad_data,
                          batch_size, input_features, output_features);
  }

  void gemm_input_gradients(const T *grad_output_data, const T *weight_data,
                            T *grad_input_data, size_t batch_size,
                            size_t input_features,
                            size_t output_features) const {
    input_gradients_impl(grad_output_data, weight_data, grad_input_data,
                         batch_size, input_features, output_features);
  }

  void add_bias_vector(T *output_data, const T *bias_data, size_t batch_size,
                       size_t output_features) const {
    add_bias_impl(output_data, bias_data, batch_size, output_features);
  }

  // Default implementations
  void gemm_impl(const T *input_data, const T *weight_data, T *output_data,
                 const size_t batch_size, const size_t input_features,
                 const size_t output_features) const {
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

  void weight_gradients_impl(const T *input_data, const T *grad_output_data,
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

  void input_gradients_impl(const T *grad_output_data, const T *weight_data,
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

  void add_bias_impl(T *output_data, const T *bias_data, size_t batch_size,
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
  DenseLayer(size_t input_features, size_t output_features,
             std::unique_ptr<ActivationFunction<T>> activation = nullptr,
             bool use_bias = true, const std::string &name = "dense")
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
    }

    // Xavier initialization
    T fan_in = static_cast<T>(input_features);
    T fan_out = static_cast<T>(output_features);
    T std_dev = std::sqrt(T(2.0) / (fan_in + fan_out));
    weights_.fill_random_normal(T(0), std_dev);
  }

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    // printf("Forward pass for micro-batch ID: %d\n", micro_batch_id);
    micro_batch_inputs_[micro_batch_id] = input;

    const size_t batch_size = input.batch_size();
    const size_t total_input_features =
        input.channels() * input.height() * input.width();

    if (total_input_features != input_features_) {
      printf("Input shape: %zu features, expected: %zu features\n",
             total_input_features, input_features_);
      throw std::invalid_argument("Input feature size mismatch in DenseLayer");
    }

    Tensor<T> output(std::vector<size_t>{batch_size, output_features_, 1, 1});

    // Perform matrix multiplication
    gemm_forward(input.data(), weights_.data(), output.data(), batch_size,
                 input_features_, output_features_);

    // Add bias
    if (use_bias_) {
      add_bias_vector(output.data(), bias_.data(), batch_size,
                      output_features_);
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
    // printf("Backward pass for micro-batch ID: %d\n", micro_batch_id);
    auto it_input = micro_batch_inputs_.find(micro_batch_id);
    auto it_pre_act = micro_batch_pre_activations_.find(micro_batch_id);

    if (it_input == micro_batch_inputs_.end()) {
      for (const auto &pair : micro_batch_inputs_) {
        printf("Cached micro-batch IDs: %d\n", pair.first);
      }
      throw std::runtime_error("No cached input found for micro-batch ID: " +
                               std::to_string(micro_batch_id));
    }
    if (it_pre_act == micro_batch_pre_activations_.end()) {
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

    // Compute weight gradients
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

    // Compute input gradients
    gemm_input_gradients(current_grad.data(), weights_.data(),
                         grad_input.data(), batch_size, input_features_,
                         output_features_);

    // // Clean up cached data for this micro-batch
    micro_batch_inputs_.erase(it_input);
    micro_batch_pre_activations_.erase(it_pre_act);
    return grad_input;
  }

  std::string type() const override { return "dense"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["input_features"] = input_features_;
    config.parameters["output_features"] = output_features_;
    config.parameters["use_bias"] = use_bias_;
    config.parameters["activation"] =
        activation_ ? activation_->name() : std::string("none");
    config.parameters["optimized"] = std::string("native");
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    auto activation_clone = activation_ ? activation_->clone() : nullptr;
    return std::make_unique<DenseLayer<T>>(input_features_, output_features_,
                                           std::move(activation_clone),
                                           use_bias_, this->name_);
  }

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override {
    if (input_shape.size() != 4) {
      throw std::invalid_argument("DenseLayer expects 4D input");
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

  void update_parameters_impl(const Optimizer<T> &optimizer) override {
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

    return std::make_unique<DenseLayer<T>>(input_features, output_features,
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
    // micro_batch_inputs_.erase(it);
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

// 2D Convolutional Layer using im2col + GEMM
template <typename T = double>
class Conv2DLayer : public ParameterizedLayer<T> {
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

  Tensor<T> weights_; // [out_channels, in_channels, kernel_h, kernel_w]
  Tensor<T> bias_;    // [out_channels, 1, 1, 1]
  Tensor<T> weight_gradients_; 
  Tensor<T> bias_gradients_;   

  // Per-micro-batch state
  std::unordered_map<int, Tensor<T>> micro_batch_inputs_;
  std::unordered_map<int, Tensor<T>> micro_batch_pre_activations_;
  mutable std::unordered_map<int, Matrix<T>> micro_batch_im2col_matrices_;

  // Helper functions for convolution
  void conv_gemm_forward(const T *col_data, const T *weight_data,
                         T *output_data, size_t output_size, size_t kernel_size,
                         size_t out_channels) const {
    conv_gemm_forward_impl(col_data, weight_data, output_data, output_size,
                           kernel_size, out_channels);
  }

  void conv_gemm_weight_gradients(const T *col_data, const T *grad_output_data,
                                  T *weight_grad_data, size_t output_size,
                                  size_t kernel_size,
                                  size_t out_channels) const {
    conv_gemm_weight_gradients_impl(col_data, grad_output_data,
                                    weight_grad_data, output_size, kernel_size,
                                    out_channels);
  }

  void conv_gemm_input_gradients(const T *grad_output_data,
                                 const T *weight_data, T *col_grad_data,
                                 size_t output_size, size_t kernel_size,
                                 size_t out_channels) const {
    conv_gemm_input_gradients_impl(grad_output_data, weight_data, col_grad_data,
                                   output_size, kernel_size, out_channels);
  }

  void conv_gemm_forward_impl(const T *col_data, const T *weight_data,
                              T *output_data, const size_t output_size,
                              const size_t kernel_size,
                              const size_t out_channels) const {
    
    // Transpose im2col matrix for better memory access patterns
    // Original: col_data[kernel_size x output_size]
    // Transposed: col_data_T[output_size x kernel_size]
    std::vector<T> col_data_transposed(kernel_size * output_size);
    transpose_matrix(col_data, col_data_transposed.data(), kernel_size, output_size);
    
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t oc = 0; oc < out_channels; ++oc) {
      for (size_t os = 0; os < output_size; ++os) {
        output_data[oc * output_size + os] = simd_dot_product_contiguous(
            &weight_data[oc * kernel_size], 
            &col_data_transposed[os * kernel_size], 
            kernel_size
        );
      }
    }
  }

  // Fast matrix transpose utility
  void transpose_matrix(const T *src, T *dst, size_t rows, size_t cols) const {
    // Use cache-friendly blocking for large matrices
    const size_t block_size = 64; // Tuned for typical L1 cache
    
    if (rows * cols < 1024) {
      // Simple transpose for small matrices
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
          dst[j * rows + i] = src[i * cols + j];
        }
      }
    } else {
      // Blocked transpose for larger matrices
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
      for (size_t i = 0; i < rows; i += block_size) {
        for (size_t j = 0; j < cols; j += block_size) {
          size_t max_i = std::min(i + block_size, rows);
          size_t max_j = std::min(j + block_size, cols);
          
          for (size_t ii = i; ii < max_i; ++ii) {
            for (size_t jj = j; jj < max_j; ++jj) {
              dst[jj * rows + ii] = src[ii * cols + jj];
            }
          }
        }
      }
    }
  }

  // Optimized SIMD dot product for contiguous memory access
  T simd_dot_product_contiguous(const T *weights, const T *col_data, 
                                size_t kernel_size) const {
    T sum = T(0);
    
    // Use SIMD for float type only
    if constexpr (std::is_same_v<T, float>) {
#if defined(__AVX2__)
      // AVX2 implementation - process 8 floats at once
      __m256 sum_vec = _mm256_setzero_ps();
      size_t simd_end = kernel_size - (kernel_size % 8);
      
      for (size_t ks = 0; ks < simd_end; ks += 8) {
        // Load 8 weights (contiguous)
        __m256 w_vec = _mm256_loadu_ps(&weights[ks]);
        
        // Load 8 col_data values (now contiguous!)
        __m256 c_vec = _mm256_loadu_ps(&col_data[ks]);
        
        // Fused multiply-add
        sum_vec = _mm256_fmadd_ps(w_vec, c_vec, sum_vec);
      }
      
      // Horizontal sum of the vector
      __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
      __m128 sum_low = _mm256_castps256_ps128(sum_vec);
      __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
      
      // Sum the 4 elements in the 128-bit vector
      sum_128 = _mm_hadd_ps(sum_128, sum_128);
      sum_128 = _mm_hadd_ps(sum_128, sum_128);
      sum = _mm_cvtss_f32(sum_128);
      
      // Handle remaining elements
      for (size_t ks = simd_end; ks < kernel_size; ++ks) {
        sum += weights[ks] * col_data[ks];
      }
      
#elif defined(__SSE2__)
      // SSE2 implementation - process 4 floats at once
      __m128 sum_vec = _mm_setzero_ps();
      size_t simd_end = kernel_size - (kernel_size % 4);
      
      for (size_t ks = 0; ks < simd_end; ks += 4) {
        // Load 4 weights (contiguous)
        __m128 w_vec = _mm_loadu_ps(&weights[ks]);
        
        // Load 4 col_data values (now contiguous!)
        __m128 c_vec = _mm_loadu_ps(&col_data[ks]);
        
        // Multiply and add
        __m128 prod = _mm_mul_ps(w_vec, c_vec);
        sum_vec = _mm_add_ps(sum_vec, prod);
      }
      
      // Horizontal sum of the vector
      sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
      sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
      sum = _mm_cvtss_f32(sum_vec);
      
      // Handle remaining elements
      for (size_t ks = simd_end; ks < kernel_size; ++ks) {
        sum += weights[ks] * col_data[ks];
      }
      
#else
      // Fallback scalar implementation
      for (size_t ks = 0; ks < kernel_size; ++ks) {
        sum += weights[ks] * col_data[ks];
      }
#endif
    } else {
      // For non-float types, use scalar implementation
      for (size_t ks = 0; ks < kernel_size; ++ks) {
        sum += weights[ks] * col_data[ks];
      }
    }
    
    return sum;
  }

  // Overloaded version for strided access (used in input gradients)
  T simd_dot_product_contiguous(const T *weights, const T *col_data, 
                                size_t length, size_t weight_stride) const {
    T sum = T(0);
    
    // Use SIMD for float type only
    if constexpr (std::is_same_v<T, float>) {
#if defined(__AVX2__)
      // AVX2 implementation - process 8 floats at once
      __m256 sum_vec = _mm256_setzero_ps();
      size_t simd_end = length - (length % 8);
      
      for (size_t i = 0; i < simd_end; i += 8) {
        // Load 8 weights with stride
        __m256 w_vec = _mm256_set_ps(
            weights[(i + 7) * weight_stride],
            weights[(i + 6) * weight_stride],
            weights[(i + 5) * weight_stride],
            weights[(i + 4) * weight_stride],
            weights[(i + 3) * weight_stride],
            weights[(i + 2) * weight_stride],
            weights[(i + 1) * weight_stride],
            weights[i * weight_stride]
        );
        
        // Load 8 col_data values (contiguous)
        __m256 c_vec = _mm256_loadu_ps(&col_data[i]);
        
        // Fused multiply-add
        sum_vec = _mm256_fmadd_ps(w_vec, c_vec, sum_vec);
      }
      
      // Horizontal sum of the vector
      __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
      __m128 sum_low = _mm256_castps256_ps128(sum_vec);
      __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
      
      // Sum the 4 elements in the 128-bit vector
      sum_128 = _mm_hadd_ps(sum_128, sum_128);
      sum_128 = _mm_hadd_ps(sum_128, sum_128);
      sum = _mm_cvtss_f32(sum_128);
      
      // Handle remaining elements
      for (size_t i = simd_end; i < length; ++i) {
        sum += weights[i * weight_stride] * col_data[i];
      }
      
#elif defined(__SSE2__)
      // SSE2 implementation - process 4 floats at once
      __m128 sum_vec = _mm_setzero_ps();
      size_t simd_end = length - (length % 4);
      
      for (size_t i = 0; i < simd_end; i += 4) {
        // Load 4 weights with stride
        __m128 w_vec = _mm_set_ps(
            weights[(i + 3) * weight_stride],
            weights[(i + 2) * weight_stride],
            weights[(i + 1) * weight_stride],
            weights[i * weight_stride]
        );
        
        // Load 4 col_data values (contiguous)
        __m128 c_vec = _mm_loadu_ps(&col_data[i]);
        
        // Multiply and add
        __m128 prod = _mm_mul_ps(w_vec, c_vec);
        sum_vec = _mm_add_ps(sum_vec, prod);
      }
      
      // Horizontal sum of the vector
      sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
      sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
      sum = _mm_cvtss_f32(sum_vec);
      
      // Handle remaining elements
      for (size_t i = simd_end; i < length; ++i) {
        sum += weights[i * weight_stride] * col_data[i];
      }
      
#else
      // Fallback scalar implementation
      for (size_t i = 0; i < length; ++i) {
        sum += weights[i * weight_stride] * col_data[i];
      }
#endif
    } else {
      // For non-float types, use scalar implementation
      for (size_t i = 0; i < length; ++i) {
        sum += weights[i * weight_stride] * col_data[i];
      }
    }
    
    return sum;
  }

  void conv_gemm_weight_gradients_impl(const T *col_data,
                                       const T *grad_output_data,
                                       T *weight_grad_data,
                                       const size_t output_size,
                                       const size_t kernel_size,
                                       const size_t out_channels) const {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t oc = 0; oc < out_channels; ++oc) {
      for (size_t ks = 0; ks < kernel_size; ++ks) {
        // SIMD-optimized dot product for weight gradients
        weight_grad_data[oc * kernel_size + ks] = simd_dot_product_contiguous(
            &grad_output_data[oc * output_size], 
            &col_data[ks * output_size], 
            output_size
        );
      }
    }
  }

  void conv_gemm_input_gradients_impl(const T *grad_output_data,
                                      const T *weight_data, T *col_grad_data,
                                      const size_t output_size,
                                      const size_t kernel_size,
                                      const size_t out_channels) const {
    
    // Transpose grad_output matrix for better memory access patterns
    // Original: grad_output_data[out_channels x output_size]
    // Transposed: grad_output_T[output_size x out_channels]
    std::vector<T> grad_output_transposed(out_channels * output_size);
    transpose_matrix(grad_output_data, grad_output_transposed.data(), out_channels, output_size);
    
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t ks = 0; ks < kernel_size; ++ks) {
      for (size_t os = 0; os < output_size; ++os) {
        // SIMD-optimized dot product for input gradients
        col_grad_data[ks * output_size + os] = simd_dot_product_contiguous(
            &weight_data[ks], // weight column for this kernel position
            &grad_output_transposed[os * out_channels], // grad_output row for this output position
            out_channels,
            kernel_size // stride for weight data
        );
      }
    }
  }

  void set_flattened_weight_gradients(const std::vector<T> &flattened) {
    weight_gradients_.from_vector(flattened);
  }

public:
  Conv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h,
              size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
              size_t pad_h = 0, size_t pad_w = 0, bool use_bias = true,
              std::unique_ptr<ActivationFunction<T>> activation = nullptr,
              const std::string &name = "conv2d")
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
    }

    // Xavier/Glorot initialization
    T fan_in = static_cast<T>(in_channels * kernel_h * kernel_w);
    T fan_out = static_cast<T>(out_channels * kernel_h * kernel_w);
    T std_dev = std::sqrt(T(2.0) / (fan_in + fan_out));
    weights_.fill_random_normal(T(0), std_dev);
  }

  // Forward and backward implementations moved from separate file
  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    if (input.channels() != in_channels_) {
      printf("Input shape: %zu channels, expected: %zu channels\n",
             input.channels(), in_channels_);
      throw std::invalid_argument("Input channel size mismatch in Conv2DLayer");
    }

    micro_batch_inputs_[micro_batch_id] = input;

    const size_t batch_size = input.batch_size();
    const size_t input_h = input.height();
    const size_t input_w = input.width();

    // Calculate output dimensions
    const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

    // Perform im2col transformation
    Matrix<T> col_matrix = input.im2col(kernel_h_, kernel_w_, stride_h_,
                                        stride_w_, pad_h_, pad_w_);
    micro_batch_im2col_matrices_[micro_batch_id] =
        col_matrix; // Cache for backward pass

    // Create output tensor
    Tensor<T> output(batch_size, out_channels_, output_h, output_w);

    // Prepare data for GEMM operation
    size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
    size_t output_size = batch_size * output_h * output_w;

    // Perform convolution using GEMM
    std::vector<T> output_flat(out_channels_ * output_size);
    conv_gemm_forward(col_matrix.data(), weights_.data(), output_flat.data(),
                      output_size, kernel_size, out_channels_);

    T *output_data = output.data();

    const size_t N_stride = output.stride(0);
    const size_t C_stride = output.stride(1);
    const size_t H_stride = output.stride(2);
    const size_t W_stride = output.stride(3);

#ifdef _OPENMP
#pragma omp parallel for collapse(4)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t oc = 0; oc < out_channels_; ++oc) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            size_t flat_idx = oc * output_size + n * (output_h * output_w) +
                              oh * output_w + ow;
            output_data[n * N_stride + oc * C_stride + oh * H_stride +
                        ow * W_stride] = output_flat[flat_idx];
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

    // Prepare data for GEMM operations
    size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
    size_t output_size = batch_size * output_h * output_w;

    // Flatten gradient output for GEMM
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

    // Compute weight gradients
    conv_gemm_weight_gradients(cached_im2col_matrix.data(), grad_output_flat.data(),
                               weight_gradients_.data(), output_size,
                               kernel_size, out_channels_);

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

    // Compute input gradients
    Matrix<T> col_grad_matrix(kernel_size, output_size);
    conv_gemm_input_gradients(grad_output_flat.data(), weights_.data(),
                              col_grad_matrix.data(), output_size, kernel_size,
                              out_channels_);

    // Use col2im to convert back to input gradient tensor
    Tensor<T> grad_input = Tensor<T>::col2im(
        col_grad_matrix, batch_size, in_channels_, input_h, input_w, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);

    return grad_input;
  }

  std::string type() const override { return "conv2d"; }

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
    config.parameters["optimized"] = std::string("native");
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    auto activation_clone = activation_ ? activation_->clone() : nullptr;
    return std::make_unique<Conv2DLayer<T>>(
        in_channels_, out_channels_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, use_bias_, std::move(activation_clone), this->name_);
  }

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override {
    if (input_shape.size() != 4) {
      throw std::invalid_argument("Conv2DLayer expects 4D input");
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

  void update_parameters_impl(const Optimizer<T> &optimizer) override {
    // To be implemented with optimizer interface
  }
};

// Optimized MaxPooling Layer for 2D tensors
template <typename T = double> class MaxPool2DLayer : public StatelessLayer<T> {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  // Use more efficient storage for mask indices
  std::unordered_map<int, std::vector<size_t>> micro_batch_mask_indices_;
  std::unordered_map<int, Tensor<T>> micro_batch_inputs_;

  // Pre-computed stride values for faster access
  mutable size_t input_stride_n_, input_stride_c_, input_stride_h_,
      input_stride_w_;
  mutable size_t output_stride_n_, output_stride_c_, output_stride_h_,
      output_stride_w_;

  // Helper function for vectorized max finding
  inline std::pair<T, size_t> find_max_in_window(const T *data, size_t start_h,
                                                 size_t start_w, size_t input_h,
                                                 size_t input_w,
                                                 size_t stride_h,
                                                 size_t stride_w) const {
    T max_val = -std::numeric_limits<T>::infinity();
    size_t max_idx = 0;

    for (size_t ph = 0; ph < pool_h_; ++ph) {
      for (size_t pw = 0; pw < pool_w_; ++pw) {
        size_t h_idx = start_h + ph;
        size_t w_idx = start_w + pw;

        if (h_idx < input_h && w_idx < input_w) {
          size_t linear_idx = h_idx * stride_h + w_idx * stride_w;
          T val = data[linear_idx];
          if (val > max_val) {
            max_val = val;
            max_idx = linear_idx;
          }
        }
      }
    }

    return {max_val, max_idx};
  }

public:
  MaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h = 0,
                 size_t stride_w = 0, size_t pad_h = 0, size_t pad_w = 0,
                 const std::string &name = "maxpool2d")
      : StatelessLayer<T>(name), pool_h_(pool_h), pool_w_(pool_w),
        stride_h_(stride_h == 0 ? pool_h : stride_h),
        stride_w_(stride_w == 0 ? pool_w : stride_w), pad_h_(pad_h),
        pad_w_(pad_w), input_stride_n_(0), input_stride_c_(0),
        input_stride_h_(0), input_stride_w_(0), output_stride_n_(0),
        output_stride_c_(0), output_stride_h_(0), output_stride_w_(0) {

    // Validate parameters
    if (pool_h_ == 0 || pool_w_ == 0) {
      throw std::invalid_argument("Pool dimensions must be positive");
    }
    if (stride_h_ == 0 || stride_w_ == 0) {
      throw std::invalid_argument("Stride dimensions must be positive");
    }
  }

  // Add method to clear cached data for memory management
  void clear_cache(int micro_batch_id = -1) {
    if (micro_batch_id < 0) {
      // Clear all cached data
      micro_batch_inputs_.clear();
      micro_batch_mask_indices_.clear();
    } else {
      // Clear specific micro-batch data
      micro_batch_inputs_.erase(micro_batch_id);
      micro_batch_mask_indices_.erase(micro_batch_id);
    }
  }

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override {
    // Store input for backward pass
    micro_batch_inputs_[micro_batch_id] = input;

    const size_t batch_size = input.batch_size();
    const size_t channels = input.channels();
    const size_t input_h = input.height();
    const size_t input_w = input.width();

    // Calculate output dimensions
    const size_t output_h = (input_h + 2 * pad_h_ - pool_h_) / stride_h_ + 1;
    const size_t output_w = (input_w + 2 * pad_w_ - pool_w_) / stride_w_ + 1;

    // Create output tensor
    Tensor<T> output(
        std::vector<size_t>{batch_size, channels, output_h, output_w});

    // Pre-compute strides for efficient memory access
    input_stride_n_ = input.stride(0);
    input_stride_c_ = input.stride(1);
    input_stride_h_ = input.stride(2);
    input_stride_w_ = input.stride(3);

    output_stride_n_ = output.stride(0);
    output_stride_c_ = output.stride(1);
    output_stride_h_ = output.stride(2);
    output_stride_w_ = output.stride(3);

    // Store mask indices more efficiently
    const size_t total_outputs = batch_size * channels * output_h * output_w;
    std::vector<size_t> mask_indices(total_outputs);

    const T *input_data = input.data();
    T *output_data = output.data();

    // Optimized pooling with better memory access patterns
    if (pad_h_ == 0 && pad_w_ == 0) {
      // No padding case - most common and fastest path
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
          const T *input_channel =
              input_data + n * input_stride_n_ + c * input_stride_c_;
          T *output_channel =
              output_data + n * output_stride_n_ + c * output_stride_c_;

          for (size_t out_h = 0; out_h < output_h; ++out_h) {
            for (size_t out_w = 0; out_w < output_w; ++out_w) {
              T max_val = -std::numeric_limits<T>::infinity();
              size_t max_idx = 0;

              const size_t start_h = out_h * stride_h_;
              const size_t start_w = out_w * stride_w_;

              // Unrolled inner loops for small kernel sizes
              if (pool_h_ == 2 && pool_w_ == 2) {
                // Special case for 2x2 pooling - most common
                const T *pool_start = input_channel +
                                      start_h * input_stride_h_ +
                                      start_w * input_stride_w_;

                T val0 = pool_start[0];
                T val1 = pool_start[input_stride_w_];
                T val2 = pool_start[input_stride_h_];
                T val3 = pool_start[input_stride_h_ + input_stride_w_];

                max_val = val0;
                max_idx = start_h * input_w + start_w;

                if (val1 > max_val) {
                  max_val = val1;
                  max_idx = start_h * input_w + start_w + 1;
                }
                if (val2 > max_val) {
                  max_val = val2;
                  max_idx = (start_h + 1) * input_w + start_w;
                }
                if (val3 > max_val) {
                  max_val = val3;
                  max_idx = (start_h + 1) * input_w + start_w + 1;
                }
              } else {
                // General case
                for (size_t ph = 0; ph < pool_h_; ++ph) {
                  for (size_t pw = 0; pw < pool_w_; ++pw) {
                    const size_t h_idx = start_h + ph;
                    const size_t w_idx = start_w + pw;

                    if (h_idx < input_h && w_idx < input_w) {
                      T val = input_channel[h_idx * input_stride_h_ +
                                            w_idx * input_stride_w_];
                      if (val > max_val) {
                        max_val = val;
                        max_idx = h_idx * input_w + w_idx;
                      }
                    }
                  }
                }
              }

              output_channel[out_h * output_stride_h_ +
                             out_w * output_stride_w_] = max_val;
              const size_t output_idx =
                  ((n * channels + c) * output_h + out_h) * output_w + out_w;
              mask_indices[output_idx] = max_idx;
            }
          }
        }
      }
    } else {
      // Padding case - use existing implementation but with optimizations
      Tensor<T> padded_input =
          input.pad(pad_h_, pad_w_, -std::numeric_limits<T>::infinity());
      const T *padded_data = padded_input.data();
      const size_t padded_h = padded_input.height();
      const size_t padded_w = padded_input.width();
      const size_t padded_stride_h = padded_input.stride(2);
      const size_t padded_stride_w = padded_input.stride(3);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
          const T *padded_channel = padded_data + n * padded_input.stride(0) +
                                    c * padded_input.stride(1);
          T *output_channel =
              output_data + n * output_stride_n_ + c * output_stride_c_;

          for (size_t out_h = 0; out_h < output_h; ++out_h) {
            for (size_t out_w = 0; out_w < output_w; ++out_w) {
              T max_val = -std::numeric_limits<T>::infinity();
              size_t max_idx = 0;

              for (size_t ph = 0; ph < pool_h_; ++ph) {
                for (size_t pw = 0; pw < pool_w_; ++pw) {
                  const size_t h_idx = out_h * stride_h_ + ph;
                  const size_t w_idx = out_w * stride_w_ + pw;

                  if (h_idx < padded_h && w_idx < padded_w) {
                    T val = padded_channel[h_idx * padded_stride_h +
                                           w_idx * padded_stride_w];
                    if (val > max_val) {
                      max_val = val;
                      max_idx = h_idx * padded_w + w_idx;
                    }
                  }
                }
              }

              output_channel[out_h * output_stride_h_ +
                             out_w * output_stride_w_] = max_val;
              const size_t output_idx =
                  ((n * channels + c) * output_h + out_h) * output_w + out_w;
              mask_indices[output_idx] = max_idx;
            }
          }
        }
      }
    }

    micro_batch_mask_indices_[micro_batch_id] = std::move(mask_indices);
    return output;
  }

  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override {
    auto it_input = micro_batch_inputs_.find(micro_batch_id);
    auto it_mask = micro_batch_mask_indices_.find(micro_batch_id);

    if (it_input == micro_batch_inputs_.end()) {
      throw std::runtime_error(
          "No cached input found for micro-batch ID in MaxPool2DLayer: " +
          std::to_string(micro_batch_id));
    }
    if (it_mask == micro_batch_mask_indices_.end()) {
      throw std::runtime_error(
          "No cached mask found for micro-batch ID in MaxPool2DLayer: " +
          std::to_string(micro_batch_id));
    }

    const Tensor<T> &last_input = it_input->second;
    const std::vector<size_t> &mask_indices = it_mask->second;

    const size_t batch_size = last_input.batch_size();
    const size_t channels = last_input.channels();
    const size_t input_h = last_input.height();
    const size_t input_w = last_input.width();
    const size_t output_h = grad_output.height();
    const size_t output_w = grad_output.width();

    Tensor<T> grad_input(
        std::vector<size_t>{batch_size, channels, input_h, input_w});
    grad_input.fill(0.0);

    const T *grad_output_data = grad_output.data();
    T *grad_input_data = grad_input.data();

    if (pad_h_ == 0 && pad_w_ == 0) {
      // No padding case - direct indexing
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
          const T *grad_out_channel =
              grad_output_data + n * output_stride_n_ + c * output_stride_c_;
          T *grad_in_channel =
              grad_input_data + n * input_stride_n_ + c * input_stride_c_;

          for (size_t out_h = 0; out_h < output_h; ++out_h) {
            for (size_t out_w = 0; out_w < output_w; ++out_w) {
              const size_t output_idx =
                  ((n * channels + c) * output_h + out_h) * output_w + out_w;
              const size_t max_idx = mask_indices[output_idx];
              const size_t max_h = max_idx / input_w;
              const size_t max_w = max_idx % input_w;

              const T grad_val = grad_out_channel[out_h * output_stride_h_ +
                                                  out_w * output_stride_w_];

              // Use atomic add to handle potential race conditions
#ifdef _OPENMP
#pragma omp atomic
#endif
              grad_in_channel[max_h * input_stride_h_ +
                              max_w * input_stride_w_] += grad_val;
            }
          }
        }
      }
    } else {
      // Padding case - need to handle coordinate transformation
      const size_t padded_h = input_h + 2 * pad_h_;
      const size_t padded_w = input_w + 2 * pad_w_;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
          const T *grad_out_channel =
              grad_output_data + n * output_stride_n_ + c * output_stride_c_;
          T *grad_in_channel =
              grad_input_data + n * input_stride_n_ + c * input_stride_c_;

          for (size_t out_h = 0; out_h < output_h; ++out_h) {
            for (size_t out_w = 0; out_w < output_w; ++out_w) {
              const size_t output_idx =
                  ((n * channels + c) * output_h + out_h) * output_w + out_w;
              const size_t padded_max_idx = mask_indices[output_idx];
              const size_t padded_max_h = padded_max_idx / padded_w;
              const size_t padded_max_w = padded_max_idx % padded_w;

              // Convert back to unpadded coordinates
              if (padded_max_h >= pad_h_ && padded_max_h < input_h + pad_h_ &&
                  padded_max_w >= pad_w_ && padded_max_w < input_w + pad_w_) {
                const size_t max_h = padded_max_h - pad_h_;
                const size_t max_w = padded_max_w - pad_w_;

                const T grad_val = grad_out_channel[out_h * output_stride_h_ +
                                                    out_w * output_stride_w_];

#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_in_channel[max_h * input_stride_h_ +
                                max_w * input_stride_w_] += grad_val;
              }
            }
          }
        }
      }
    }

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

    // micro_batch_masks_.erase(it_mask);
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

    Tensor<T> output = input.reshape({batch_size, features, 1, 1});

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
    Tensor<T> grad_input = grad_output.reshape(original_shape);

    // micro_batch_original_shapes_.erase(it);
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
    // Dense layer
    register_layer(
        "dense", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
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

          return std::make_unique<DenseLayer<T>>(
              input_features, output_features, std::move(activation), use_bias,
              config.name);
        });

    // Conv2D layer
    register_layer(
        "conv2d", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
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

          return std::make_unique<Conv2DLayer<T>>(
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
dense(size_t input_features, size_t output_features,
      const std::string &activation = "none", bool use_bias = true,
      const std::string &name = "dense") {
  std::unique_ptr<ActivationFunction<T>> act = nullptr;
  if (activation != "none") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    act = factory.create(activation);
  }

  return std::make_unique<layers::DenseLayer<T>>(
      input_features, output_features, std::move(act), use_bias, name);
}

template <typename T = double>
std::unique_ptr<layers::Layer<T>>
conv2d(size_t in_channels, size_t out_channels, size_t kernel_h,
       size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
       size_t pad_h = 0, size_t pad_w = 0,
       const std::string &activation = "none", bool use_bias = true,
       const std::string &name = "conv2d") {
  std::unique_ptr<ActivationFunction<T>> act = nullptr;
  if (activation != "none") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    act = factory.create(activation);
  }
  return std::make_unique<layers::Conv2DLayer<T>>(
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