/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_dense_layer.hpp"

#include "device/task.hpp"
#include "nn/layers_impl/cpu/dense_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/dense_ops.hpp"
#endif
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "nn/layers_impl/parameterized_layer.hpp"
#include "type/type.hpp"

namespace tnn {

LegacyDenseLayer::LegacyDenseLayer(size_t input_features, size_t output_features, bool use_bias,
                                   const std::string &name)
    : ParameterizedLayer(name),
      input_features_(input_features),
      output_features_(output_features),
      use_bias_(use_bias) {}

void LegacyDenseLayer::init_params() {
  weights_ = make_param_tensor({output_features_, input_features_});
  weight_gradients_ = make_param_tensor({output_features_, input_features_});
  weight_gradients_->fill(0);
  if (use_bias_) {
    bias_ = make_param_tensor({output_features_});
    bias_gradients_ = make_param_tensor({output_features_});
    bias_gradients_->fill(0);
  }
  // PyTorch default Kaiming Uniform: Uniform(-bound, bound) where bound = 1 / sqrt(fan_in)
  double bound = 1.0 / std::sqrt(static_cast<double>(input_features_));

  if (this->use_seed_) {
    weights_->fill_random_uniform(-bound, bound, this->srand_seed_);
  } else {
    weights_->fill_random_uniform(-bound, bound);
  }

  if (use_bias_) {
    if (this->use_seed_) {
      bias_->fill_random_uniform(-bound, bound, this->srand_seed_);
    } else {
      bias_->fill_random_uniform(-bound, bound);
    }
  }
}

void LegacyDenseLayer::forward_impl(const Tensor &input, Tensor &output, size_t mb_id) {
  const std::vector<size_t> &in_shape = input->shape();
  size_t last_dim = in_shape.back();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  if (last_dim != input_features_) {
    std::cerr << "Input last dimension: " << last_dim << " features, expected: " << input_features_
              << " features" << std::endl;
    throw std::invalid_argument("Input feature size mismatch in LegacyDenseLayer");
  }

  if (this->is_training_) {
    Tensor &cached_input = this->get_cached_tensor(mb_id, "input");
    cached_input = input;
  }

  std::vector<size_t> out_shape = in_shape;
  out_shape.back() = output_features_;
  output->ensure(out_shape);

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_dense_forward, input, weights_, output, batch_size,
                                 input_features_, output_features_, "default");

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(add_bias_vector, output, bias_, batch_size, output_features_,
                                   "default");
  }
}

void LegacyDenseLayer::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id) {
  if (gradient->shape().back() != output_features_) {
    throw std::invalid_argument("Gradient feature size mismatch in LegacyDenseLayer");
  }
  Tensor &input = this->get_cached_tensor(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for micro-batch ID: " + std::to_string(mb_id));
  }
  const std::vector<size_t> &in_shape = input->shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  grad_input->ensure(input->shape());

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_weight_gradients, input, gradient, weight_gradients_,
                                 batch_size, input_features_, output_features_, "default");

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(compute_bias_gradients, gradient, bias_gradients_, batch_size,
                                   output_features_, "default");
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_input_gradients, gradient, weights_, grad_input,
                                 batch_size, input_features_, output_features_, "default");
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyDenseLayer::compute_dense_forward(
    const Tensor &input, const Tensor &weights, Tensor &output, size_t batch_size,
    size_t input_features, size_t output_features, const std::string &flow_id) const {
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyDenseLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weights->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LegacyDenseLayer weight tensor dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "LegacyDenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(flow_id, cpu::legacy_dense::compute_dense_forward<Compute_T>,
                           input->data_as<Compute_T>(), weights->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), batch_size, input_features,
                           output_features);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(flow_id,
                            cuda::legacy_dense::compute_dense_forward_ex<IO_T, Param_T, Compute_T>,
                            input->data_as<IO_T>(), weights->data_as<Param_T>(),
                            output->data_as<IO_T>(), batch_size, input_features, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_dense_forward.");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyDenseLayer::compute_weight_gradients(
    const Tensor &input, const Tensor &gradient, Tensor &weight_grad, size_t batch_size,
    size_t input_features, size_t output_features, const std::string &flow_id) const {
  if (input->data_type() != dtype_of<IO_T>() || gradient->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyDenseLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight_grad->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyDenseLayer weight gradient dtype mismatch with dispatch Param_T");
  }
  if (this->device_->device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "LegacyDenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(flow_id, cpu::legacy_dense::compute_weight_gradients<IO_T>,
                           input->data_as<IO_T>(), gradient->data_as<IO_T>(),
                           weight_grad->data_as<IO_T>(), batch_size, input_features,
                           output_features);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(
        flow_id, cuda::legacy_dense::compute_weight_gradients_ex<IO_T, Param_T, Compute_T>,
        input->data_as<IO_T>(), gradient->data_as<IO_T>(), weight_grad->data_as<Param_T>(),
        batch_size, input_features, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_weight_gradients.");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyDenseLayer::compute_input_gradients(
    const Tensor &gradient, const Tensor &weights, Tensor &grad_input, size_t batch_size,
    size_t input_features, size_t output_features, const std::string &flow_id) const {
  if (gradient->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyDenseLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weights->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LegacyDenseLayer weight tensor dtype mismatch with dispatch Param_T");
  }
  if (this->device_->device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "LegacyDenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(flow_id, cpu::legacy_dense::compute_input_gradients<IO_T>,
                           gradient->data_as<IO_T>(), weights->data_as<IO_T>(),
                           grad_input->data_as<IO_T>(), batch_size, input_features,
                           output_features);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(
        flow_id, cuda::legacy_dense::compute_input_gradients_ex<IO_T, Param_T, Compute_T>,
        gradient->data_as<IO_T>(), weights->data_as<Param_T>(), grad_input->data_as<IO_T>(),
        batch_size, input_features, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_input_gradients.");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyDenseLayer::compute_bias_gradients(const Tensor &gradient,
                                                               Tensor &bias_gradient,
                                                               size_t batch_size,
                                                               size_t output_features,
                                                               const std::string &flow_id) const {
  if (gradient->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyDenseLayer gradient dtype mismatch with dispatch IO_T");
  }
  if (bias_gradient->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LegacyDenseLayer bias gradient dtype mismatch with dispatch Param_T");
  }
  if (this->device_->device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "LegacyDenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(flow_id, cpu::legacy_dense::compute_bias_gradients<IO_T>,
                           gradient->data_as<IO_T>(), bias_gradient->data_as<IO_T>(), batch_size,
                           output_features);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(
        flow_id, cuda::legacy_dense::compute_bias_gradients_ex<IO_T, Param_T, Compute_T>,
        gradient->data_as<IO_T>(), bias_gradient->data_as<Param_T>(), batch_size, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_bias_gradients");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyDenseLayer::add_bias_vector(Tensor &output, const Tensor &bias,
                                                        size_t batch_size, size_t output_features,
                                                        const std::string &flow_id) const {
  if (output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyDenseLayer output dtype mismatch with dispatch IO_T");
  }
  if (bias->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LegacyDenseLayer bias dtype mismatch with dispatch Param_T");
  }
  if (this->device_->device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "LegacyDenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(flow_id, cpu::legacy_dense::add_bias_vector<IO_T>,
                           output->data_as<IO_T>(), bias->data_as<IO_T>(), batch_size,
                           output_features);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(
        flow_id, cuda::legacy_dense::add_bias_vector_ex<IO_T, Param_T, Compute_T>,
        output->data_as<IO_T>(), bias->data_as<Param_T>(), batch_size, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for add_bias_vector");
  }
  return nullptr;
}

LayerConfig LegacyDenseLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["input_features"] = input_features_;
  config.parameters["output_features"] = output_features_;
  config.parameters["use_bias"] = use_bias_;
  return config;
}

std::unique_ptr<Layer> LegacyDenseLayer::clone() const {
  return std::make_unique<LegacyDenseLayer>(input_features_, output_features_, use_bias_,
                                            this->name_);
}

std::vector<size_t> LegacyDenseLayer::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::runtime_error("LegacyDenseLayer::compute_output_shape: Input shape is empty.");
  }
  std::vector<size_t> out_shape = input_shape;
  out_shape.back() = output_features_;
  return out_shape;
}

void LegacyDenseLayer::collect_parameters(std::vector<Tensor> &params) {
  params.push_back(weights_);
  if (use_bias_) {
    params.push_back(bias_);
  }
}

void LegacyDenseLayer::collect_gradients(std::vector<Tensor> &grads) {
  grads.push_back(weight_gradients_);
  if (use_bias_) {
    grads.push_back(bias_gradients_);
  }
}

std::unique_ptr<LegacyDenseLayer> LegacyDenseLayer::create_from_config(const LayerConfig &config) {
  size_t input_features = config.get<size_t>("input_features");
  size_t output_features = config.get<size_t>("output_features");
  bool use_bias = config.get<bool>("use_bias");

  return std::make_unique<LegacyDenseLayer>(input_features, output_features, use_bias, config.name);
}

uint64_t LegacyDenseLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  size_t batch_size = input_shape[0];
  uint64_t gemm_flops = 2ULL * batch_size * input_features_ * output_features_;
  uint64_t bias_flops = use_bias_ ? (batch_size * output_features_) : 0;
  return gemm_flops + bias_flops;
}

uint64_t LegacyDenseLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t batch_size = input_shape[0];
  uint64_t weight_grad_flops = 2ULL * input_features_ * batch_size * output_features_;
  uint64_t bias_grad_flops = use_bias_ ? (batch_size * output_features_) : 0;
  uint64_t input_grad_flops = 2ULL * batch_size * output_features_ * input_features_;
  return weight_grad_flops + bias_grad_flops + input_grad_flops;
}

}  // namespace tnn
