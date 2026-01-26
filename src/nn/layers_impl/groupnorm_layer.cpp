/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/groupnorm_layer.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/cpu/groupnorm_ops.hpp"
#include "nn/layers_impl/cuda/groupnorm_ops.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace tnn {

GroupNormLayer::GroupNormLayer(size_t num_groups, size_t num_channels, float epsilon, bool affine,
                               const std::string &name)
    : ParameterizedLayer(name), num_groups_(num_groups), num_channels_(num_channels),
      epsilon_(epsilon), affine_(affine) {
  if (num_channels_ % num_groups_ != 0) {
    throw std::invalid_argument("num_channels must be divisible by num_groups in GroupNormLayer");
  }
}

void GroupNormLayer::init_params() {
  if (this->initialized_) {
    return;
  }

  gamma_gradients_ = make_param_tensor({num_channels_, 1, 1, 1});
  beta_gradients_ = make_param_tensor({num_channels_, 1, 1, 1});
  gamma_gradients_->fill(0);
  beta_gradients_->fill(0);

  if (affine_) {
    gamma_ = make_param_tensor({num_channels_, 1, 1, 1});
    beta_ = make_param_tensor({num_channels_, 1, 1, 1});
    gamma_->fill(1);
    beta_->fill(0);
  }

  this->initialized_ = true;
}

void GroupNormLayer::forward_impl(const Tensor &input, Tensor &output, size_t mb_id) {
  if (input->shape()[1] != num_channels_) {
    throw std::invalid_argument("Input channels must match num_channels in GroupNormLayer");
  }

  size_t batch_size = input->dimension(0);
  size_t channels = input->dimension(1);
  size_t spatial_size = input->stride(1);

  if (num_channels_ != channels) {
    throw std::invalid_argument("Input channels must match num_channels in GroupNormLayer");
  }

  output->ensure(input->shape());

  Tensor &norm = this->get_cached_tensor(mb_id, "norm");
  if (norm == nullptr) {
    norm = make_io_tensor(input->shape());
  } else {
    norm->ensure(input->shape());
  }

  Tensor &mean = this->get_cached_tensor(mb_id, "mean");
  if (mean == nullptr) {
    mean = make_io_tensor({batch_size * num_groups_});
  } else {
    mean->ensure({batch_size * num_groups_});
  }
  Tensor &inv_std = this->get_cached_tensor(mb_id, "inv_std");
  if (inv_std == nullptr) {
    inv_std = make_io_tensor({batch_size * num_groups_});
  } else {
    inv_std->ensure({batch_size * num_groups_});
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_forward_fused, input, mean, inv_std, gamma_, beta_, output,
                                 norm, batch_size, channels, spatial_size, "default");

  if (this->is_training_) {
    Tensor &cached_input = this->get_cached_tensor(mb_id, "input");
    cached_input = input;
  }
}

void GroupNormLayer::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id) {
  Tensor &normalized = this->get_cached_tensor(mb_id, "norm");
  Tensor &inv_std = this->get_cached_tensor(mb_id, "inv_std");
  const Tensor &input = this->get_cached_tensor(mb_id, "input");
  if (!normalized || !inv_std || !input) {
    throw std::runtime_error("No cached tensors found for micro-batch ID in GroupNormLayer: " +
                             std::to_string(mb_id));
  }

  const size_t batch_size = input->dimension(0);
  const size_t channels = input->dimension(1);
  const size_t spatial_size = input->stride(1);

  grad_input->ensure(input->shape());

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_backward_fused, gradient, normalized, inv_std, gamma_,
                                 gamma_gradients_, beta_gradients_, grad_input, batch_size,
                                 channels, spatial_size, "default");
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task>
GroupNormLayer::run_forward_fused(const Tensor &input, Tensor &group_mean, Tensor &group_inv_std,
                                  const Tensor &gamma, const Tensor &beta, Tensor &output,
                                  Tensor &norm_cache, size_t batch_size, size_t channels,
                                  size_t spatial_size, const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "GroupNormLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("GroupNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("GroupNormLayer gamma dtype mismatch with dispatch Param_T");
  }
#ifdef USE_CUDA
  if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task("default", cuda::groupnorm::run_forward_fused<Compute_T>,
                           input->data_as<Compute_T>(), group_mean->data_as<Compute_T>(),
                           group_inv_std->data_as<Compute_T>(),
                           affine_ ? gamma->data_as<Compute_T>() : nullptr,
                           affine_ ? beta->data_as<Compute_T>() : nullptr,
                           output->data_as<Compute_T>(), norm_cache->data_as<Compute_T>(),
                           batch_size, channels, spatial_size, num_groups_, epsilon_, affine_);
  } else
#endif
  {
    return create_cpu_task("default", cpu::groupnorm::run_forward_fused<Compute_T>,
                           input->data_as<Compute_T>(), group_mean->data_as<Compute_T>(),
                           group_inv_std->data_as<Compute_T>(),
                           affine_ ? gamma->data_as<Compute_T>() : nullptr,
                           affine_ ? beta->data_as<Compute_T>() : nullptr,
                           output->data_as<Compute_T>(), norm_cache->data_as<Compute_T>(),
                           batch_size, channels, spatial_size, num_groups_, epsilon_, affine_);
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> GroupNormLayer::run_backward_fused(
    const Tensor &grad_output, const Tensor &norm_input, const Tensor &inv_std, const Tensor &gamma,
    Tensor &d_gamma, Tensor &d_beta, Tensor &grad_input, size_t batch_size, size_t channels,
    size_t spatial_size, const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "GroupNormLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (grad_output->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("GroupNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("GroupNormLayer gamma dtype mismatch with dispatch Param_T");
  }
#ifdef USE_CUDA
  if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task("default", cuda::groupnorm::run_backward_fused<Compute_T>,
                           grad_output->data_as<Compute_T>(), norm_input->data_as<Compute_T>(),
                           inv_std->data_as<Compute_T>(), gamma->data_as<Compute_T>(),
                           d_gamma->data_as<Compute_T>(), d_beta->data_as<Compute_T>(),
                           grad_input->data_as<Compute_T>(), batch_size, channels, spatial_size,
                           num_groups_, affine_);
  } else
#endif
  {
    return create_cpu_task(
        "default", cpu::groupnorm::run_backward_fused<Compute_T>, grad_output->data_as<Compute_T>(),
        norm_input->data_as<Compute_T>(), inv_std->data_as<Compute_T>(),
        gamma->data_as<Compute_T>(), d_gamma->data_as<Compute_T>(), d_beta->data_as<Compute_T>(),
        grad_input->data_as<Compute_T>(), batch_size, channels, spatial_size, num_groups_, affine_);
  }
}

LayerConfig GroupNormLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["num_groups"] = num_groups_;
  config.parameters["num_channels"] = num_channels_;
  config.parameters["epsilon"] = epsilon_;
  config.parameters["affine"] = affine_;
  return config;
}

std::unique_ptr<Layer> GroupNormLayer::clone() const {
  return std::make_unique<GroupNormLayer>(num_groups_, num_channels_, epsilon_, affine_,
                                          this->name_);
}

std::vector<size_t>
GroupNormLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

void GroupNormLayer::collect_parameters(std::vector<Tensor> &params) {
  if (affine_) {
    params.push_back(gamma_);
    params.push_back(beta_);
  }
}

void GroupNormLayer::collect_gradients(std::vector<Tensor> &grads) {
  if (affine_) {
    grads.push_back(gamma_gradients_);
    grads.push_back(beta_gradients_);
  }
}

std::unique_ptr<GroupNormLayer> GroupNormLayer::create_from_config(const LayerConfig &config) {
  size_t num_groups = config.get<size_t>("num_groups");
  size_t num_channels = config.get<size_t>("num_channels");
  float epsilon = config.get<float>("epsilon", 1e-5f);
  bool affine = config.get<bool>("affine");

  return std::make_unique<GroupNormLayer>(num_groups, num_channels, epsilon, affine, config.name);
}

uint64_t GroupNormLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t channels_per_group = num_channels_ / num_groups_;
  size_t spatial_size = num_elements / (batch_size * num_channels_);

  // Mean computation per group
  uint64_t mean_flops = batch_size * num_groups_ * channels_per_group * spatial_size;

  // Variance computation per group
  uint64_t var_flops = 2 * num_elements + mean_flops;

  // Normalization
  uint64_t norm_flops = 3 * num_elements;

  // Affine transformation
  uint64_t affine_flops = affine_ ? (2 * num_elements) : 0;

  return mean_flops + var_flops + norm_flops + affine_flops;
}

uint64_t GroupNormLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t channels_per_group = num_channels_ / num_groups_;
  size_t spatial_size = num_elements / (batch_size * num_channels_);

  // Parameter gradients
  uint64_t param_grad_flops =
      affine_ ? (2 * batch_size * num_groups_ * channels_per_group * spatial_size) : 0;

  // Input gradients
  uint64_t input_grad_flops = 9 * num_elements;

  return param_grad_flops + input_grad_flops;
}

} // namespace tnn
