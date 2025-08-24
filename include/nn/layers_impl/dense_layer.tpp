#include "dense_layer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../../utils/ops.hpp"
#include "../parallel_for.hpp"
#include "parameterized_layer.hpp"

namespace tnn {

// Constructor
template <typename T>
DenseLayer<T>::DenseLayer(size_t input_features, size_t output_features,
                          std::unique_ptr<ActivationFunction<T>> activation,
                          bool use_bias, const std::string &name)
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


template <typename T>
Tensor<T> DenseLayer<T>::forward(const Tensor<T> &input, int micro_batch_id) {
  micro_batch_inputs_[micro_batch_id] = input.clone();

  const size_t batch_size = input.batch_size();
  const size_t total_input_features =
      input.channels() * input.height() * input.width();

  if (total_input_features != input_features_) {
    std::cerr << "Input shape: " << total_input_features
              << " features, expected: " << input_features_ << " features"
              << std::endl;
    throw std::invalid_argument("Input feature size mismatch in DenseLayer");
  }

  Tensor<T> output(std::vector<size_t>{batch_size, output_features_, 1, 1});

  gemm_forward(input.data(), weights_.data(), output.data(), batch_size,
               input_features_, output_features_);

  if (use_bias_) {
    add_bias_vector(output.data(), bias_.data(), batch_size,
                    output_features_);
  }

  micro_batch_pre_activations_[micro_batch_id] = output.clone();

  if (activation_) {
    activation_->apply(output);
  }

  return output;
}


template <typename T>
Tensor<T> DenseLayer<T>::backward(const Tensor<T> &grad_output,
                                  int micro_batch_id) {
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  auto it_pre_act = micro_batch_pre_activations_.find(micro_batch_id);

  if (it_input == micro_batch_inputs_.end()) {
    for (const auto &pair : micro_batch_inputs_) {
      std::cout << "Cached micro-batch IDs: " << pair.first << std::endl;
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
    current_grad = activation_grad.clone();
  }

  // Compute weight gradients
  gemm_weight_gradients(last_input.data(), current_grad.data(),
                        weight_gradients_.data(), batch_size, input_features_,
                        output_features_);

  // Compute bias gradients
  if (use_bias_) {
    bias_gradients_.fill(T(0));
#ifdef USE_TBB
    tnn::parallel_for_range<size_t>(0, output_features_, [&](size_t out_f) {
      T grad_sum = T(0);
      for (size_t n = 0; n < batch_size; ++n) {
        grad_sum += current_grad(n, out_f, 0, 0);
      }
      bias_gradients_(out_f, 0, 0, 0) = grad_sum;
    });
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t out_f = 0; out_f < output_features_; ++out_f) {
      T grad_sum = T(0);
      for (size_t n = 0; n < batch_size; ++n) {
        grad_sum += current_grad(n, out_f, 0, 0);
      }
      bias_gradients_(out_f, 0, 0, 0) = grad_sum;
    }
#endif
  }

  // Compute input gradients
  gemm_input_gradients(current_grad.data(), weights_.data(),
                       grad_input.data(), batch_size, input_features_,
                       output_features_);

  return grad_input;
}

// Helper function implementations
template <typename T>
void DenseLayer<T>::gemm_forward(const T *input_data, const T *weight_data,
                                 T *output_data, size_t batch_size,
                                 size_t input_features,
                                 size_t output_features) const {
  gemm_impl(input_data, weight_data, output_data, batch_size, input_features,
            output_features);
}

template <typename T>
void DenseLayer<T>::gemm_weight_gradients(const T *input_data,
                                          const T *grad_output_data,
                                          T *weight_grad_data,
                                          size_t batch_size,
                                          size_t input_features,
                                          size_t output_features) const {
  weight_gradients_impl(input_data, grad_output_data, weight_grad_data,
                        batch_size, input_features, output_features);
}

template <typename T>
void DenseLayer<T>::gemm_input_gradients(const T *grad_output_data,
                                         const T *weight_data,
                                         T *grad_input_data, size_t batch_size,
                                         size_t input_features,
                                         size_t output_features) const {
  input_gradients_impl(grad_output_data, weight_data, grad_input_data,
                       batch_size, input_features, output_features);
}

template <typename T>
void DenseLayer<T>::add_bias_vector(T *output_data, const T *bias_data,
                                    size_t batch_size,
                                    size_t output_features) const {
  add_bias_impl(output_data, bias_data, batch_size, output_features);
}


template <typename T>
void DenseLayer<T>::gemm_impl(const T *input_data, const T *weight_data,
                              T *output_data, const size_t batch_size,
                              const size_t input_features,
                              const size_t output_features) const {
#ifdef USE_TBB
  tnn::parallel_for_2d(
      batch_size, output_features, [&](size_t n, size_t out_f) {
        output_data[n * output_features + out_f] =
            utils::simd_dot_product(
                &weight_data[out_f * input_features],
                &input_data[n * input_features], input_features);
      });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t out_f = 0; out_f < output_features; ++out_f) {
      // Use SIMD-optimized dot product for contiguous memory access
      output_data[n * output_features + out_f] =
          utils::simd_dot_product(
              &weight_data[out_f * input_features],
              &input_data[n * input_features], input_features);
    }
  }
#endif
}

template <typename T>
void DenseLayer<T>::weight_gradients_impl(const T *input_data,
                                          const T *grad_output_data,
                                          T *weight_grad_data,
                                          size_t batch_size,
                                          size_t input_features,
                                          size_t output_features) const {
  auto input_transposed = std::make_unique<T[]>(input_features * batch_size);
  auto grad_output_transposed =
      std::make_unique<T[]>(output_features * batch_size);

  utils::transpose_2d(input_data, input_transposed.get(), batch_size,
                      input_features);
  utils::transpose_2d(grad_output_data, grad_output_transposed.get(),
                      batch_size, output_features);

#ifdef USE_TBB
  tnn::parallel_for_2d(
      output_features, input_features, [&](size_t out_f, size_t in_f) {
        weight_grad_data[out_f * input_features + in_f] =
            utils::simd_dot_product(
                &grad_output_transposed[out_f * batch_size],
                &input_transposed[in_f * batch_size], batch_size);
      });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t out_f = 0; out_f < output_features; ++out_f) {
    for (size_t in_f = 0; in_f < input_features; ++in_f) {
      // Use SIMD-optimized dot product with contiguous memory access
      weight_grad_data[out_f * input_features + in_f] =
          utils::simd_dot_product(
              &grad_output_transposed[out_f * batch_size],
              &input_transposed[in_f * batch_size], batch_size);
    }
  }
#endif
}

template <typename T>
void DenseLayer<T>::input_gradients_impl(const T *grad_output_data,
                                         const T *weight_data,
                                         T *grad_input_data, size_t batch_size,
                                         size_t input_features,
                                         size_t output_features) const {
  auto weights_transposed =
      std::make_unique<T[]>(input_features * output_features);
  utils::transpose_2d(weight_data, weights_transposed.get(), output_features,
                      input_features);

#ifdef USE_TBB
  tnn::parallel_for_2d(
      batch_size, input_features, [&](size_t n, size_t in_f) {
        grad_input_data[n * input_features + in_f] =
            utils::simd_dot_product(
                &grad_output_data[n * output_features],
                &weights_transposed[in_f * output_features], output_features);
      });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t in_f = 0; in_f < input_features; ++in_f) {
      grad_input_data[n * input_features + in_f] =
          utils::simd_dot_product(
              &grad_output_data[n * output_features],
              &weights_transposed[in_f * output_features], output_features);
    }
  }
#endif
}

template <typename T>
void DenseLayer<T>::add_bias_impl(T *output_data, const T *bias_data,
                                  size_t batch_size,
                                  size_t output_features) const {
#ifdef USE_TBB
  tnn::parallel_for_2d(
      batch_size, output_features, [&](size_t n, size_t out_f) {
        output_data[n * output_features + out_f] += bias_data[out_f];
      });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t out_f = 0; out_f < output_features; ++out_f) {
      output_data[n * output_features + out_f] += bias_data[out_f];
    }
  }
#endif
}


template <typename T>
std::string DenseLayer<T>::type() const {
  return "dense";
}

template <typename T>
LayerConfig DenseLayer<T>::get_config() const {
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

template <typename T>
std::unique_ptr<Layer<T>> DenseLayer<T>::clone() const {
  auto activation_clone = activation_ ? activation_->clone() : nullptr;
  return std::make_unique<DenseLayer<T>>(input_features_, output_features_,
                                         std::move(activation_clone),
                                         use_bias_, this->name_);
}

template <typename T>
std::vector<size_t>
DenseLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("DenseLayer expects 4D input");
  }
  return {input_shape[0], output_features_, 1, 1};
}

// Protected method implementations
template <typename T>
void DenseLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.push_back(&weights_);
  if (use_bias_) {
    params.push_back(&bias_);
  }
}

template <typename T>
void DenseLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.push_back(&weight_gradients_);
  if (use_bias_) {
    grads.push_back(&bias_gradients_);
  }
}

template <typename T>
void DenseLayer<T>::update_parameters_impl(Optimizer<T> &optimizer) {
  std::vector<Tensor<T> *> params = this->parameters();
  std::vector<Tensor<T> *> grads = this->gradients();
  if (params.size() != grads.size()) {
    throw std::runtime_error(
        "Parameter and gradient size mismatch in DenseLayer");
  }
  optimizer.update(params, grads);
}

template <typename T>
std::unique_ptr<Layer<T>>
DenseLayer<T>::create_from_config(const LayerConfig &config) {
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

} // namespace tnn
