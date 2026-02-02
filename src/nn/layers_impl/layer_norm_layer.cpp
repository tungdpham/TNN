/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/layer_norm_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/layer_norm_ops.hpp"
#endif
#include <stdexcept>
#include <type_traits>

#include "device/task.hpp"
#include "nn/layers_impl/layer_norm_layer.hpp"

namespace tnn {

LayerNormLayer::LayerNormLayer(size_t normalized_shape, float epsilon, bool affine,
                               const std::string &name)
    : ParameterizedLayer(name),
      normalized_shape_(normalized_shape),
      epsilon_(epsilon),
      affine_(affine) {}

void LayerNormLayer::init_params() {
  if (this->initialized_) return;

  if (affine_) {
    gamma_ = make_param_tensor({normalized_shape_});
    beta_ = make_param_tensor({normalized_shape_});
    gamma_->fill(1);
    beta_->fill(0);

    gamma_gradients_ = make_param_tensor({normalized_shape_});
    beta_gradients_ = make_param_tensor({normalized_shape_});
    gamma_gradients_->fill(0);
    beta_gradients_->fill(0);
  }

  this->initialized_ = true;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayer::layer_norm_forward(const ConstTensor &input, Tensor &output,
                                                         const ConstTensor &gamma,
                                                         const ConstTensor &beta, size_t batch_size,
                                                         size_t channels,
                                                         const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LayerNormLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma && gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LayerNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task(
        "default", cpu::layer_norm::layer_norm_forward<Compute_T>, input->data_as<Compute_T>(),
        output->data_as<Compute_T>(), gamma ? gamma->data_as<Compute_T>() : nullptr,
        beta ? beta->data_as<Compute_T>() : nullptr, batch_size, channels, epsilon_);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(
        "default", cuda::layer_norm::layer_norm_forward<Compute_T>, input->data_as<Compute_T>(),
        output->data_as<Compute_T>(), gamma ? gamma->data_as<Compute_T>() : nullptr,
        beta ? beta->data_as<Compute_T>() : nullptr, batch_size, channels, epsilon_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for layer_norm_forward");
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayer::layer_norm_backward(
    const ConstTensor &gradient, const ConstTensor &input, const ConstTensor &gamma,
    Tensor &grad_input, Tensor &gamma_gradients, Tensor &beta_gradients, size_t batch_size,
    size_t channels, const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LayerNormLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (gradient->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma && gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LayerNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::layer_norm::layer_norm_backward<Compute_T>,
                           gradient->data_as<Compute_T>(), input->data_as<Compute_T>(),
                           gamma ? gamma->data_as<Compute_T>() : nullptr,
                           grad_input->data_as<Compute_T>(),
                           gamma_gradients ? gamma_gradients->data_as<Compute_T>() : nullptr,
                           beta_gradients ? beta_gradients->data_as<Compute_T>() : nullptr,
                           batch_size, channels, epsilon_);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task("default", cuda::layer_norm::layer_norm_backward<Compute_T>,
                            gradient->data_as<Compute_T>(), input->data_as<Compute_T>(),
                            gamma ? gamma->data_as<Compute_T>() : nullptr,
                            grad_input->data_as<Compute_T>(),
                            gamma_gradients ? gamma_gradients->data_as<Compute_T>() : nullptr,
                            beta_gradients ? beta_gradients->data_as<Compute_T>() : nullptr,
                            batch_size, channels, epsilon_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for layer_norm_backward");
  }
}

void LayerNormLayer::forward_impl(const ConstTensor &input, Tensor &output, size_t mb_id) {
  if (this->is_training_) {
    ConstTensor &cached_input = this->get_cached_tensor(mb_id, "input");
    cached_input = input;
  }
  const auto &shape = input->shape();
  size_t last_dim = shape.back();

  if (last_dim != normalized_shape_) {
    throw std::invalid_argument("Input last dimension (" + std::to_string(last_dim) +
                                ") must match normalized_shape (" +
                                std::to_string(normalized_shape_) + ") in LayerNormLayer");
  }

  output->ensure(shape);

  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(layer_norm_forward, input, output, gamma_, beta_, batch_size,
                                 channels, "default");
}

void LayerNormLayer::backward_impl(const ConstTensor &gradient, Tensor &grad_input, size_t mb_id) {
  ConstTensor &input = this->get_cached_tensor(mb_id, "input");
  if (!input) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const auto &shape = input->shape();
  grad_input->ensure(shape);

  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(layer_norm_backward, gradient, input, gamma_, grad_input,
                                 gamma_gradients_, beta_gradients_, batch_size, channels,
                                 "default");
}

uint64_t LayerNormLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() < 2) return 0;
  size_t elements = 1;
  for (size_t s : input_shape) elements *= s;
  return elements * 8;
}

uint64_t LayerNormLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() < 2) return 0;
  size_t elements = 1;
  for (size_t s : input_shape) elements *= s;
  return elements * 16;
}

LayerConfig LayerNormLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["normalized_shape"] = normalized_shape_;
  config.parameters["epsilon"] = epsilon_;
  config.parameters["affine"] = affine_;
  return config;
}

std::unique_ptr<Layer> LayerNormLayer::clone() const {
  return std::make_unique<LayerNormLayer>(normalized_shape_, epsilon_, affine_, this->name_);
}

void LayerNormLayer::collect_parameters(std::vector<Tensor> &params) {
  if (affine_) {
    params.push_back(gamma_);
    params.push_back(beta_);
  }
}

void LayerNormLayer::collect_gradients(std::vector<Tensor> &grads) {
  if (affine_) {
    grads.push_back(gamma_gradients_);
    grads.push_back(beta_gradients_);
  }
}

std::unique_ptr<LayerNormLayer> LayerNormLayer::create_from_config(const LayerConfig &config) {
  size_t normalized_shape = config.get<size_t>("normalized_shape");
  float epsilon = config.get<float>("epsilon", 1e-5f);
  bool affine = config.get<bool>("affine", true);
  return std::make_unique<LayerNormLayer>(normalized_shape, epsilon, affine, config.name);
}

}  // namespace tnn
