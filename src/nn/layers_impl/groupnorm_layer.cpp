/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/groupnorm_layer.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "device/task.hpp"
#include "nn/layers_impl/cpu/groupnorm_ops.hpp"
#include "nn/layers_impl/cuda/groupnorm_ops.hpp"
#include "tensor/tensor.hpp"

namespace synet {

GroupNormLayerImpl::GroupNormLayerImpl(size_t num_groups, size_t num_channels, float epsilon,
                                       bool affine, const std::string &name)
    : SISOLayerImpl(name),
      num_groups_(num_groups),
      num_channels_(num_channels),
      epsilon_(epsilon),
      affine_(affine) {
  if (num_channels_ % num_groups_ != 0) {
    throw std::invalid_argument(
        "num_channels must be divisible by num_groups in GroupNormLayerImpl");
  }
}

void GroupNormLayerImpl::init_impl() {
  gamma_.fill(1.0f);
  beta_.fill(0.0f);

  gamma_gradients_.fill(0.0f);
  beta_gradients_.fill(0.0f);
}

Tensor GroupNormLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (input.shape()[1] != num_channels_) {
    throw std::invalid_argument("Input channels must match num_channels in GroupNormLayerImpl");
  }

  size_t batch_size = input.dimension(0);
  size_t channels = input.dimension(1);
  size_t spatial_size = input.stride(1);

  if (num_channels_ != channels) {
    throw std::invalid_argument("Input channels must match num_channels in GroupNormLayerImpl");
  }

  Tensor output = get_tensor(input.shape(), io_dtype_);

  Tensor norm = this->get_tensor(input.shape(), io_dtype_);
  residuals["norm"] = norm;

  Tensor mean = this->get_tensor({batch_size * num_groups_}, io_dtype_);
  residuals["mean"] = mean;

  Tensor inv_std = this->get_tensor({batch_size * num_groups_}, io_dtype_);
  residuals["inv_std"] = inv_std;

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_forward, input, mean, inv_std, gamma_, beta_, output, norm,
                                 batch_size, channels, spatial_size, this->flow_handle_);

  if (this->is_training_) {
    residuals["input"] = input;
  }

  return output;
}

Tensor GroupNormLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  Tensor &normalized = residuals["norm"];
  Tensor &inv_std = residuals["inv_std"];
  const Tensor &input = residuals["input"];
  if (!normalized || !inv_std || !input) {
    throw std::runtime_error("No cached tensors found for GroupNormLayerImpl backward pass");
  }

  size_t batch_size = input.dimension(0);
  size_t channels = input.dimension(1);
  size_t spatial_size = input.stride(1);

  Tensor grad_input = get_tensor(input.shape(), io_dtype_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_backward, grad_output, normalized, inv_std, gamma_,
                                 gamma_gradients_, beta_gradients_, grad_input, batch_size,
                                 channels, spatial_size, this->flow_handle_);

  return grad_input;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> GroupNormLayerImpl::run_forward(const Tensor &input, Tensor &group_mean,
                                                      Tensor &group_inv_std, const Tensor &gamma,
                                                      const Tensor &beta, Tensor &output,
                                                      Tensor &norm_cache, size_t batch_size,
                                                      size_t channels, size_t spatial_size,
                                                      flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "GroupNormLayerImpl mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input.data_type() != dtype_of<IO_T>() || output.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("GroupNormLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("GroupNormLayerImpl gamma dtype mismatch with dispatch Param_T");
  }
#ifdef USE_CUDA
  if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(this->flow_handle_, cuda::groupnorm::run_forward<Compute_T>,
                            input.data_as<Compute_T>(), group_mean.data_as<Compute_T>(),
                            group_inv_std.data_as<Compute_T>(),
                            affine_ ? gamma.data_as<Compute_T>() : nullptr,
                            affine_ ? beta.data_as<Compute_T>() : nullptr,
                            output.data_as<Compute_T>(), norm_cache.data_as<Compute_T>(),
                            batch_size, channels, spatial_size, num_groups_, epsilon_, affine_);
  } else
#endif
  {
    return create_cpu_task(this->flow_handle_, cpu::groupnorm::run_forward<Compute_T>,
                           input.data_as<Compute_T>(), group_mean.data_as<Compute_T>(),
                           group_inv_std.data_as<Compute_T>(),
                           affine_ ? gamma.data_as<Compute_T>() : nullptr,
                           affine_ ? beta.data_as<Compute_T>() : nullptr,
                           output.data_as<Compute_T>(), norm_cache.data_as<Compute_T>(), batch_size,
                           channels, spatial_size, num_groups_, epsilon_, affine_);
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> GroupNormLayerImpl::run_backward(
    const Tensor &grad_output, const Tensor &norm_input, const Tensor &inv_std, const Tensor &gamma,
    Tensor &d_gamma, Tensor &d_beta, Tensor &grad_input, size_t batch_size, size_t channels,
    size_t spatial_size, flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "GroupNormLayerImpl mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (grad_output.data_type() != dtype_of<IO_T>() || grad_input.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("GroupNormLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("GroupNormLayerImpl gamma dtype mismatch with dispatch Param_T");
  }
#ifdef USE_CUDA
  if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(this->flow_handle_, cuda::groupnorm::run_backward<Compute_T>,
                            grad_output.data_as<Compute_T>(), norm_input.data_as<Compute_T>(),
                            inv_std.data_as<Compute_T>(), gamma.data_as<Compute_T>(),
                            d_gamma.data_as<Compute_T>(), d_beta.data_as<Compute_T>(),
                            grad_input.data_as<Compute_T>(), batch_size, channels, spatial_size,
                            num_groups_, affine_);
  } else
#endif
  {
    return create_cpu_task(this->flow_handle_, cpu::groupnorm::run_backward<Compute_T>,
                           grad_output.data_as<Compute_T>(), norm_input.data_as<Compute_T>(),
                           inv_std.data_as<Compute_T>(), gamma.data_as<Compute_T>(),
                           d_gamma.data_as<Compute_T>(), d_beta.data_as<Compute_T>(),
                           grad_input.data_as<Compute_T>(), batch_size, channels, spatial_size,
                           num_groups_, affine_);
  }
}

LayerConfig GroupNormLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("num_groups", num_groups_);
  config.set("num_channels", num_channels_);
  config.set("epsilon", epsilon_);
  config.set("affine", affine_);
  return config;
}

Vec<size_t> GroupNormLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  return input_shape;
}

std::shared_ptr<GroupNormLayerImpl> GroupNormLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t num_groups = config.get<size_t>("num_groups");
  size_t num_channels = config.get<size_t>("num_channels");
  float epsilon = config.get<float>("epsilon", 1e-5f);
  bool affine = config.get<bool>("affine");

  return std::make_shared<GroupNormLayerImpl>(num_groups, num_channels, epsilon, affine,
                                              config.name);
}

}  // namespace synet
