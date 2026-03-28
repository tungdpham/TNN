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

void LegacyDenseLayer::init_impl() {
  float bound = static_cast<float>(1.0 / std::sqrt(static_cast<double>(input_features_)));

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

  weight_gradients_->fill(0.0f);
  if (use_bias_) {
    bias_gradients_->fill(0.0f);
  }
}

Tensor LegacyDenseLayer::forward_impl(const ConstTensor &input, size_t mb_id) {
  const Vec<size_t> &in_shape = input->shape();
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
    set_immutable_cache(mb_id, "input", input);
  }

  Vec<size_t> out_shape = in_shape;
  out_shape.back() = output_features_;
  Tensor output = get_output_tensor(out_shape);

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_dense_forward, input, weights_, output, batch_size,
                                 input_features_, output_features_, this->flow_handle_);

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(add_bias, output, bias_, batch_size, output_features_,
                                   this->flow_handle_);
  }

  return output;
}

Tensor LegacyDenseLayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
  if (grad_output->shape().back() != output_features_) {
    throw std::invalid_argument("Gradient feature size mismatch in LegacyDenseLayer");
  }
  ConstTensor &input = this->get_immutable_cache(mb_id, "input");
  const Vec<size_t> &in_shape = input->shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  Tensor grad_input = get_output_tensor(input->shape());

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_weight_gradients, input, grad_output, weight_gradients_,
                                 batch_size, input_features_, output_features_, this->flow_handle_);

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(compute_bias_gradients, grad_output, bias_gradients_, batch_size,
                                   output_features_, this->flow_handle_);
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_input_gradients, grad_output, weights_, grad_input,
                                 batch_size, input_features_, output_features_, this->flow_handle_);

  return grad_input;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyDenseLayer::compute_dense_forward(
    const ConstTensor &input, const ConstTensor &weights, const Tensor &output, size_t batch_size,
    size_t input_features, size_t output_features, flowHandle_t handle) const {
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyDenseLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weights->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LegacyDenseLayer weight tensor dtype mismatch with dispatch Param_T");
  }

  if (get_engine_type() == EngineType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "LegacyDenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(handle, cpu::legacy_dense::run_forward<Compute_T>,
                           input->data_as<Compute_T>(), weights->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), batch_size, input_features,
                           output_features);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(handle, cuda::legacy_dense::run_forward<IO_T, Param_T, Compute_T>,
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
    const ConstTensor &input, const ConstTensor &grad_output, const Tensor &weight_grad,
    size_t batch_size, size_t input_features, size_t output_features, flowHandle_t handle) const {
  if (input->data_type() != dtype_of<IO_T>() || grad_output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyDenseLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight_grad->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyDenseLayer weight grad_output dtype mismatch with dispatch Param_T");
  }
  if (get_engine_type() == EngineType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "LegacyDenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(handle, cpu::legacy_dense::run_weight_gradients<IO_T>,
                           input->data_as<IO_T>(), grad_output->data_as<IO_T>(),
                           weight_grad->data_as<IO_T>(), batch_size, input_features,
                           output_features);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(
        handle, cuda::legacy_dense::run_weight_gradients<IO_T, Param_T, Compute_T>,
        input->data_as<IO_T>(), grad_output->data_as<IO_T>(), weight_grad->data_as<Param_T>(),
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
    const ConstTensor &grad_output, const ConstTensor &weights, const Tensor &grad_input,
    size_t batch_size, size_t input_features, size_t output_features, flowHandle_t handle) const {
  if (grad_output->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyDenseLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weights->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LegacyDenseLayer weight tensor dtype mismatch with dispatch Param_T");
  }
  if (get_engine_type() == EngineType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "LegacyDenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(handle, cpu::legacy_dense::run_input_gradients<IO_T>,
                           grad_output->data_as<IO_T>(), weights->data_as<IO_T>(),
                           grad_input->data_as<IO_T>(), batch_size, input_features,
                           output_features);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(
        handle, cuda::legacy_dense::run_input_gradients<IO_T, Param_T, Compute_T>,
        grad_output->data_as<IO_T>(), weights->data_as<Param_T>(), grad_input->data_as<IO_T>(),
        batch_size, input_features, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_input_gradients.");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyDenseLayer::compute_bias_gradients(const ConstTensor &grad_output,
                                                               const Tensor &bias_gradient,
                                                               size_t batch_size,
                                                               size_t output_features,
                                                               flowHandle_t handle) const {
  if (grad_output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyDenseLayer grad_output dtype mismatch with dispatch IO_T");
  }
  if (bias_gradient->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyDenseLayer bias grad_output dtype mismatch with dispatch Param_T");
  }
  if (get_engine_type() == EngineType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "LegacyDenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(handle, cpu::legacy_dense::run_bias_gradients<IO_T>,
                           grad_output->data_as<IO_T>(), bias_gradient->data_as<IO_T>(), batch_size,
                           output_features);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(handle,
                            cuda::legacy_dense::run_bias_gradients<IO_T, Param_T, Compute_T>,
                            grad_output->data_as<IO_T>(), bias_gradient->data_as<Param_T>(),
                            batch_size, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_bias_gradients");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyDenseLayer::add_bias(const Tensor &output, const ConstTensor &bias,
                                                 size_t batch_size, size_t output_features,
                                                 flowHandle_t handle) const {
  if (output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyDenseLayer output dtype mismatch with dispatch IO_T");
  }
  if (bias->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LegacyDenseLayer bias dtype mismatch with dispatch Param_T");
  }
  if (get_engine_type() == EngineType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "LegacyDenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(handle, cpu::legacy_dense::add_bias<IO_T>, output->data_as<IO_T>(),
                           bias->data_as<IO_T>(), batch_size, output_features);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(handle, cuda::legacy_dense::add_bias<IO_T, Param_T, Compute_T>,
                            output->data_as<IO_T>(), bias->data_as<Param_T>(), batch_size,
                            output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for add_bias");
  }
  return nullptr;
}

LayerConfig LegacyDenseLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("input_features", input_features_);
  config.set("output_features", output_features_);
  config.set("use_bias", use_bias_);
  return config;
}

Vec<size_t> LegacyDenseLayer::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::runtime_error("LegacyDenseLayer::compute_output_shape: Input shape is empty.");
  }
  Vec<size_t> out_shape = input_shape;
  out_shape.back() = output_features_;
  return out_shape;
}

std::unique_ptr<LegacyDenseLayer> LegacyDenseLayer::create_from_config(const LayerConfig &config) {
  size_t input_features = config.get<size_t>("input_features");
  size_t output_features = config.get<size_t>("output_features");
  bool use_bias = config.get<bool>("use_bias");

  return std::make_unique<LegacyDenseLayer>(input_features, output_features, use_bias, config.name);
}

}  // namespace tnn
