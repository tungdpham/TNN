/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "dense_layer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "parameterized_layer.hpp"
#include "utils/ops.hpp"
#include "utils/parallel_for.hpp"

namespace tnn {

template <typename T>
DenseLayer<T>::DenseLayer(size_t input_features, size_t output_features,
                          std::unique_ptr<ActivationFunction<T>> activation, bool use_bias,
                          const std::string &name)
    : ParameterizedLayer<T>(name), input_features_(input_features),
      output_features_(output_features), use_bias_(use_bias), activation_(std::move(activation)) {
  weights_ = Tensor<T>(output_features, input_features, 1, 1);
  weight_gradients_ = Tensor<T>(output_features, input_features, 1, 1);

  if (use_bias_) {
    bias_ = Tensor<T>(output_features, 1, 1, 1);
    bias_gradients_ = Tensor<T>(output_features, 1, 1, 1);
  }

  T fan_in = static_cast<T>(input_features);
  T fan_out = static_cast<T>(output_features);
  T std_dev = std::sqrt(T(2.0) / (fan_in + fan_out));
  weights_.fill_random_normal(T(0), std_dev);
}

template <typename T>
Tensor<T> DenseLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  micro_batch_inputs_[micro_batch_id] = input.clone();

  const size_t batch_size = input.batch_size();
  const size_t total_input_features = input.channels() * input.height() * input.width();

  if (total_input_features != input_features_) {
    std::cerr << "Input shape: " << total_input_features
              << " features, expected: " << input_features_ << " features" << std::endl;
    throw std::invalid_argument("Input feature size mismatch in DenseLayer");
  }

  Tensor<T> output(batch_size, output_features_, size_t(1), size_t(1), nullptr);

  compute_dense_forward(input.data(), weights_.data(), output.data(), batch_size, input_features_,
                        output_features_);

  if (use_bias_) {
    add_bias_vector(output.data(), bias_.data(), batch_size, output_features_);
  }

  micro_batch_pre_activations_[micro_batch_id] = output.clone();

  if (activation_) {
    activation_->apply(output);
  }

  return output;
}

template <typename T>
Tensor<T> DenseLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
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
    throw std::runtime_error("No cached pre-activation output found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  if (gradient.shape() != it_pre_act->second.shape()) {
    throw std::invalid_argument("Gradient output shape does not match cached pre-activation shape");
  }

  const Tensor<T> &last_input = it_input->second;
  size_t batch_size = last_input.batch_size();
  Tensor<T> grad_input(last_input.shape(), nullptr);

  Tensor<T> current_grad = gradient.clone();

  if (activation_) {
    activation_->compute_gradient_inplace(it_pre_act->second, current_grad);
  }

  compute_weight_gradients(last_input.data(), current_grad.data(), weight_gradients_.data(),
                           batch_size, input_features_, output_features_);

  if (use_bias_) {
    compute_bias_gradients(current_grad.data(), bias_gradients_.data(), batch_size,
                           output_features_);
  }

  compute_input_gradients(current_grad.data(), weights_.data(), grad_input.data(), batch_size,
                          input_features_, output_features_);

  return grad_input;
}

template <typename T>
void DenseLayer<T>::compute_dense_forward(const T *input_data, const T *weight_data, T *output_data,
                                          const size_t batch_size, const size_t input_features,
                                          const size_t output_features) const {
  utils::parallel_for_2d(batch_size, output_features, [&](size_t n, size_t out_f) {
    output_data[n * output_features + out_f] = utils::simd_dot_product(
        &weight_data[out_f * input_features], &input_data[n * input_features], input_features);
  });
}

template <typename T>
void DenseLayer<T>::compute_weight_gradients(const T *input_data, const T *gradient_data,
                                             T *weight_grad_data, const size_t batch_size,
                                             const size_t input_features,
                                             const size_t output_features) const {
  T *input_transposed = (T *)malloc(sizeof(T) * input_features * batch_size);
  T *gradient_transposed = (T *)malloc(sizeof(T) * output_features * batch_size);

  utils::transpose_2d_inplace(input_data, input_transposed, batch_size, input_features);
  utils::transpose_2d_inplace(gradient_data, gradient_transposed, batch_size, output_features);

  utils::parallel_for_2d(output_features, input_features, [&](size_t out_f, size_t in_f) {
    weight_grad_data[out_f * input_features + in_f] += utils::simd_dot_product(
        &gradient_transposed[out_f * batch_size], &input_transposed[in_f * batch_size], batch_size);
  });

  free(input_transposed);
  free(gradient_transposed);
}

template <typename T>
void DenseLayer<T>::compute_input_gradients(const T *gradient_data, const T *weight_data,
                                            T *grad_input_data, size_t batch_size,
                                            size_t input_features, size_t output_features) const {
  T *weights_transposed = (T *)malloc(sizeof(T) * output_features * input_features);
  utils::transpose_2d_inplace(weight_data, weights_transposed, output_features, input_features);

  utils::parallel_for_2d(batch_size, input_features, [&](size_t n, size_t in_f) {
    grad_input_data[n * input_features + in_f] =
        utils::simd_dot_product(&gradient_data[n * output_features],
                                &weights_transposed[in_f * output_features], output_features);
  });

  free(weights_transposed);
}

template <typename T>
void DenseLayer<T>::compute_bias_gradients(const T *current_grad_data, T *bias_gradient_data,
                                           size_t batch_size, size_t output_features) const {
  utils::parallel_for_range<size_t>(0, output_features, [&](size_t out_f) {
    T grad_sum = T(0);
    for (size_t n = 0; n < batch_size; ++n) {
      grad_sum += current_grad_data[n * output_features + out_f];
    }
    bias_gradient_data[out_f] += grad_sum;
  });
}

template <typename T>
void DenseLayer<T>::add_bias_vector(T *output_data, const T *bias_data, size_t batch_size,
                                    size_t output_features) const {
  utils::parallel_for_2d(batch_size, output_features, [&](size_t n, size_t out_f) {
    output_data[n * output_features + out_f] += bias_data[out_f];
  });
}

template <typename T> std::string DenseLayer<T>::type() const { return "dense"; }

template <typename T> LayerConfig DenseLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["input_features"] = input_features_;
  config.parameters["output_features"] = output_features_;
  config.parameters["use_bias"] = use_bias_;
  config.parameters["activation"] = activation_ ? activation_->name() : std::string("none");
  config.parameters["optimized"] = std::string("native");
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> DenseLayer<T>::clone() const {
  auto activation_clone = activation_ ? activation_->clone() : nullptr;
  return std::make_unique<DenseLayer<T>>(input_features_, output_features_,
                                         std::move(activation_clone), use_bias_, this->name_);
}

template <typename T>
std::vector<size_t>
DenseLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("DenseLayer expects 4D input");
  }
  return {input_shape[0], output_features_, 1, 1};
}

template <typename T> void DenseLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.push_back(&weights_);
  if (use_bias_) {
    params.push_back(&bias_);
  }
}

template <typename T> void DenseLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.push_back(&weight_gradients_);
  if (use_bias_) {
    grads.push_back(&bias_gradients_);
  }
}

template <typename T> void DenseLayer<T>::clear_gradients() {
  weight_gradients_.fill(T(0));
  if (use_bias_) {
    bias_gradients_.fill(T(0));
  }
}

template <typename T>
std::unique_ptr<Layer<T>> DenseLayer<T>::create_from_config(const LayerConfig &config) {
  size_t input_features = config.get<size_t>("input_features");
  size_t output_features = config.get<size_t>("output_features");
  bool use_bias = config.get<bool>("use_bias");
  std::string activation_name = config.get<std::string>("activation");

  std::unique_ptr<ActivationFunction<T>> activation;
  if (activation_name != "none") {

    ActivationFactory<T>::register_defaults();
    activation = ActivationFactory<T>::create(activation_name);
  }

  return std::make_unique<DenseLayer<T>>(input_features, output_features, std::move(activation),
                                         use_bias, config.name);
}

template <typename T> uint32_t DenseLayer<T>::forward_complexity(std::vector<size_t> input_shape) {
  // compute forward: O(batch_size * input_features * output_features)
  size_t batch_size = input_shape[0];
  size_t total_input_features =
      std::accumulate(input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<size_t>());
  size_t output_features = output_features_;
  return batch_size * total_input_features * output_features;
}

template <typename T>
uint32_t DenseLayer<T>::backward_complexity(std::vector<size_t> gradient_shape) {
  // Weight gradients: O(out_features * in_features * batch_size)
  // Bias gradients: O(batch_size * out_features)
  // Input gradients: O(batch_size * in_features * out_features)
  size_t batch_size = gradient_shape[0];
  size_t weight_grad_ops = output_features_ * input_features_ * batch_size;
  size_t bias_grad_ops = batch_size * output_features_;
  size_t input_grad_ops = batch_size * input_features_ * output_features_;
  size_t total_ops = weight_grad_ops + bias_grad_ops + input_grad_ops;
  return total_ops;
}

} // namespace tnn
