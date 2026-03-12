/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_batchnorm_layer.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "device/task.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/cpu/batchnorm_nchw_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/batchnorm_nchw_ops.hpp"
#endif

namespace tnn {

LegacyBatchNormLayer::LegacyBatchNormLayer(size_t num_features, float epsilon, float momentum,
                                           bool affine, const std::string &name)
    : ParameterizedLayer(name),
      num_features_(num_features),
      epsilon_(epsilon),
      momentum_(momentum),
      affine_(affine) {}

void LegacyBatchNormLayer::init_impl() {
  gamma_->fill(1.0f);
  beta_->fill(0.0f);

  running_mean_->fill(0.0f);
  running_var_->fill(1.0f);

  gamma_gradients_->fill(0.0f);
  beta_gradients_->fill(0.0f);

  dummy_mean_gradients_->fill(0.0f);
  dummy_var_gradients_->fill(0.0f);
}

void LegacyBatchNormLayer::forward_impl(const ConstTensor &input, const Tensor &output,
                                        size_t mb_id) {
  if (input->dims() < 3) {
    throw std::invalid_argument("BatchNorm: Input tensor must have at least 3 dimensions");
  }
  if (input->dimension(1) != num_features_) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features");
  }

  def_forward(input, output, mb_id);
}

void LegacyBatchNormLayer::backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                                         size_t mb_id) {
  def_backward(grad_output, grad_input, mb_id);
}

void LegacyBatchNormLayer::def_forward(const ConstTensor &input, const Tensor &output,
                                       size_t mb_id) {
  size_t batch_size, channels, spatial_size;
  batch_size = input->dimension(0);
  channels = input->dimension(1);
  spatial_size = input->stride(1);

  if (num_features_ != channels) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features.");
  }

  output->ensure(input->shape());

  Tensor &norm = this->get_mutable_tensor(mb_id, "norm");
  Tensor &batch_inv_std = this->get_mutable_tensor(mb_id, "inv_std");
  Tensor &batch_mean = this->get_mutable_tensor(mb_id, "mean");

  if (!norm)
    norm = make_tensor<float>(input->shape(), this->device());
  else
    norm->ensure(input->shape());

  if (!batch_inv_std)
    batch_inv_std = make_tensor<float>({num_features_}, this->device());
  else
    batch_inv_std->ensure({num_features_});

  if (!batch_mean)
    batch_mean = make_tensor<float>({num_features_}, this->device());
  else
    batch_mean->ensure({num_features_});

  if (this->is_training_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(run_forward_fused, input, batch_mean, batch_inv_std,
                                   running_mean_, running_var_, gamma_, beta_, output, norm,
                                   batch_size, channels, spatial_size, this->flow_handle_);
  } else {
    DISPATCH_ON_3_DTYPES_TO_METHOD(compute_inference_output_impl, input, output, batch_size,
                                   channels, spatial_size, this->flow_handle_);
  }
}

void LegacyBatchNormLayer::def_backward(const ConstTensor &grad_output, const Tensor &grad_input,
                                        size_t mb_id) {
  const Tensor &norm = this->get_mutable_tensor(mb_id, "norm");
  const Tensor &inv_std = this->get_mutable_tensor(mb_id, "inv_std");
  const Tensor &batch_mean = this->get_mutable_tensor(mb_id, "mean");

  if (!norm || !inv_std || !batch_mean) {
    throw std::runtime_error("Missing cached tensors for backward pass in LegacyBatchNormLayer");
  }

  const size_t batch_size = grad_output->dimension(0);
  const size_t channels = grad_output->dimension(1);
  const size_t spatial_size = grad_output->stride(1);

  grad_input->ensure(grad_output->shape());
  DISPATCH_ON_3_DTYPES_TO_METHOD(run_backward_fused, grad_output, norm, inv_std, gamma_,
                                 gamma_gradients_, beta_gradients_, grad_input, batch_size,
                                 channels, spatial_size, this->flow_handle_);
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyBatchNormLayer::compute_inference_output_impl(
    const ConstTensor &input, const Tensor &output, size_t batch_size, size_t channels,
    size_t spatial_size, flowHandle_t handle) {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyBatchNormLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyBatchNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (input->device_type() != output->device_type() ||
      input->device_type() != running_mean_->device_type() ||
      running_mean_->device_type() != running_var_->device_type()) {
    throw std::runtime_error("All tensors must be on the same device for inference output");
  }

  if (affine_ && (input->device_type() != gamma_->device_type() ||
                  gamma_->device_type() != beta_->device_type())) {
    throw std::runtime_error("Gamma and beta must be on the same device as input");
  }

  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::batchnorm_nchw::compute_inference_output<IO_T>,
                           input->data_as<IO_T>(), running_mean_->data_as<float>(),
                           running_var_->data_as<float>(), gamma_->data_as<float>(),
                           affine_ ? beta_->data_as<float>() : nullptr, output->data_as<IO_T>(),
                           batch_size, channels, spatial_size, epsilon_, affine_);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(
        handle, cuda::batchnorm_nchw::compute_inference_output<IO_T>, input->data_as<IO_T>(),
        running_mean_->data_as<float>(), running_var_->data_as<float>(),
        affine_ ? gamma_->data_as<float>() : nullptr, affine_ ? beta_->data_as<float>() : nullptr,
        output->data_as<IO_T>(), batch_size, channels, spatial_size, epsilon_, affine_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_inference_output");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyBatchNormLayer::compute_inference_output(
    const ConstTensor &input, const Tensor &output, size_t batch_size, size_t channels,
    size_t spatial_size, flowHandle_t handle) {
  return compute_inference_output_impl<IO_T, Param_T, Compute_T>(input, output, batch_size,
                                                                 channels, spatial_size, handle);
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyBatchNormLayer::run_forward_fused(
    const ConstTensor &input, const Tensor &batch_mean, const Tensor &batch_inv_std,
    const Tensor &running_mean, const Tensor &running_var, const ConstTensor &gamma,
    const ConstTensor &beta, const Tensor &output, const Tensor &norm, size_t batch_size,
    size_t channels, size_t spatial_size, flowHandle_t handle) {
  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::batchnorm_nchw::run_forward_fused<IO_T>,
                           input->data_as<IO_T>(), batch_mean->data_as<float>(),
                           batch_inv_std->data_as<float>(), running_mean->data_as<float>(),
                           running_var->data_as<float>(), gamma->data_as<float>(),
                           beta->data_as<float>(), output->data_as<IO_T>(), norm->data_as<float>(),
                           batch_size, channels, spatial_size, momentum_, epsilon_, affine_);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::batchnorm_nchw::run_forward_fused<IO_T>,
                            input->data_as<IO_T>(), batch_mean->data_as<float>(),
                            batch_inv_std->data_as<float>(), running_mean->data_as<float>(),
                            running_var->data_as<float>(), gamma->data_as<float>(),
                            beta->data_as<float>(), output->data_as<IO_T>(), norm->data_as<float>(),
                            batch_size, channels, spatial_size, momentum_, epsilon_, affine_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for run_forward_fused");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyBatchNormLayer::run_backward_fused(
    const ConstTensor &grad_output, const ConstTensor &norm_input, const ConstTensor &inv_std,
    const ConstTensor &gamma, const Tensor &d_gamma, const Tensor &d_beta, const Tensor &grad_input,
    size_t batch_size, size_t channels, size_t spatial_size, flowHandle_t handle) {
  if (grad_output->device_type() == DeviceType::CPU) {
    return create_cpu_task(
        handle, cpu::batchnorm_nchw::run_backward_fused<IO_T>, grad_output->data_as<IO_T>(),
        norm_input->data_as<float>(), inv_std->data_as<float>(), gamma->data_as<float>(),
        d_gamma->data_as<float>(), d_beta->data_as<float>(), grad_input->data_as<IO_T>(),
        batch_size, channels, spatial_size, affine_);
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    return create_cuda_task(
        handle, cuda::batchnorm_nchw::run_backward_fused<IO_T>, grad_output->data_as<IO_T>(),
        norm_input->data_as<float>(), inv_std->data_as<float>(), gamma->data_as<float>(),
        d_gamma->data_as<float>(), d_beta->data_as<float>(), grad_input->data_as<IO_T>(),
        batch_size, channels, spatial_size, affine_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for run_backward_fused");
  }
  return nullptr;
}

LayerConfig LegacyBatchNormLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("num_features", num_features_);
  config.set("epsilon", epsilon_);
  config.set("momentum", momentum_);
  config.set("affine", affine_);
  return config;
}

std::vector<size_t> LegacyBatchNormLayer::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  return input_shape;
}

std::unique_ptr<LegacyBatchNormLayer> LegacyBatchNormLayer::create_from_config(
    const LayerConfig &config) {
  size_t num_features = config.get<size_t>("num_features");
  float epsilon = config.get<float>("epsilon");
  float momentum = config.get<float>("momentum");
  bool affine = config.get<bool>("affine");

  return std::make_unique<LegacyBatchNormLayer>(num_features, epsilon, momentum, affine,
                                                config.name);
}

}  // namespace tnn
