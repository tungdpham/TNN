/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_batchnorm_layer.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/base_layer.hpp"
#include "nn/layers_impl/cpu/batchnorm_nchw_ops.hpp"
#include "nn/layers_impl/cuda/batchnorm_nchw_ops.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace tnn {

LegacyBatchNormLayer::LegacyBatchNormLayer(size_t num_features, float epsilon, float momentum,
                                           bool affine, const std::string &name)
    : ParameterizedLayer(name), num_features_(num_features), epsilon_(epsilon), momentum_(momentum),
      affine_(affine) {}

void LegacyBatchNormLayer::init_params() {
  if (this->initialized_) {
    return;
  }

  gamma_gradients_ = make_param_tensor({num_features_});
  beta_gradients_ = make_param_tensor({num_features_});
  gamma_gradients_->fill(0.0f);
  beta_gradients_->fill(0.0f);

  gamma_ = make_param_tensor({num_features_});
  beta_ = make_param_tensor({num_features_});
  gamma_->fill(1.0f);
  beta_->fill(0.0f);

  running_mean_ = make_param_tensor({num_features_});
  running_var_ = make_param_tensor({num_features_});
  running_mean_->fill(0.0f);
  running_var_->fill(1.0f);

  this->initialized_ = true;
}

void LegacyBatchNormLayer::forward_impl(const Tensor &input, Tensor &output,
                                        size_t micro_batch_id) {
  if (input->dims() < 3) {
    throw std::invalid_argument("BatchNorm: Input tensor must have at least 3 dimensions");
  }
  if (input->dimension(1) != num_features_) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features");
  }

  def_forward(input, output, micro_batch_id);
}

void LegacyBatchNormLayer::backward_impl(const Tensor &gradient, Tensor &grad_input,
                                         size_t micro_batch_id) {
  def_backward(gradient, grad_input, micro_batch_id);
}

void LegacyBatchNormLayer::def_forward(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  size_t batch_size, channels, spatial_size;
  batch_size = input->dimension(0);
  channels = input->dimension(1);
  spatial_size = input->stride(1);

  if (num_features_ != channels) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features.");
  }

  output->ensure(input->shape(), this->device_);

  Tensor &norm_cache = micro_batch_normalized_[micro_batch_id];
  Tensor &batch_inv_std = micro_batch_inv_std_[micro_batch_id];
  Tensor &batch_mean_fixed = batch_mean_fixed_[micro_batch_id];

  if (!norm_cache)
    norm_cache = make_tensor<float>({input->size()}, this->device_);
  else
    norm_cache->ensure({input->size()}, this->device_);

  if (!batch_inv_std)
    batch_inv_std = make_tensor<float>({num_features_}, this->device_);
  else
    batch_inv_std->ensure({num_features_}, this->device_);

  if (!batch_mean_fixed)
    batch_mean_fixed = make_tensor<float>({num_features_}, this->device_);
  else
    batch_mean_fixed->ensure({num_features_}, this->device_);

  if (this->is_training_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(run_forward_fused, input, batch_mean_fixed, batch_inv_std,
                                   running_mean_, running_var_, gamma_, beta_, output, norm_cache,
                                   batch_size, channels, spatial_size, "default");
  } else {
    DISPATCH_ON_3_DTYPES_TO_METHOD(compute_inference_output_impl, input, output, batch_size,
                                   channels, spatial_size, "default");
  }
}

void LegacyBatchNormLayer::def_backward(const Tensor &gradient, Tensor &grad_input,
                                        size_t micro_batch_id) {
  auto it_normalized = micro_batch_normalized_.find(micro_batch_id);

  if (it_normalized == micro_batch_normalized_.end()) {
    throw std::runtime_error("No cached data found for micro-batch ID in LegacyBatchNormLayer: " +
                             std::to_string(micro_batch_id));
  }

  auto it_inv_std = micro_batch_inv_std_.find(micro_batch_id);
  if (it_inv_std == micro_batch_inv_std_.end()) {
    throw std::runtime_error("No cached inv_std found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  const size_t batch_size = gradient->dimension(0);
  const size_t channels = gradient->dimension(1);
  const size_t spatial_size = gradient->stride(1);

  grad_input->ensure(gradient->shape(), this->device_);
  DISPATCH_ON_3_DTYPES_TO_METHOD(run_backward_fused, gradient, it_normalized->second,
                                 it_inv_std->second, gamma_, gamma_gradients_, beta_gradients_,
                                 grad_input, batch_size, channels, spatial_size, "default");
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyBatchNormLayer::compute_inference_output_impl(
    const Tensor &input, Tensor &output, size_t batch_size, size_t channels, size_t spatial_size,
    const std::string &flow_id) {
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
    return create_cpu_task(flow_id, cpu::batchnorm_nchw::compute_inference_output<IO_T>,
                           input->data_as<IO_T>(), running_mean_->data_as<float>(),
                           running_var_->data_as<float>(), gamma_->data_as<float>(),
                           affine_ ? beta_->data_as<float>() : nullptr, output->data_as<IO_T>(),
                           batch_size, channels, spatial_size, epsilon_, affine_);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_gpu_task(
        flow_id, cuda::batchnorm_nchw::compute_inference_output<IO_T>, input->data_as<IO_T>(),
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
std::unique_ptr<Task>
LegacyBatchNormLayer::compute_inference_output(const Tensor &input, Tensor &output,
                                               size_t batch_size, size_t channels,
                                               size_t spatial_size, const std::string &flow_id) {
  return compute_inference_output_impl<IO_T, Param_T, Compute_T>(input, output, batch_size,
                                                                 channels, spatial_size, flow_id);
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyBatchNormLayer::run_forward_fused(
    const Tensor &input, Tensor &batch_mean_fixed, Tensor &batch_inv_std, Tensor &running_mean,
    Tensor &running_var, const Tensor &gamma, const Tensor &beta, Tensor &output,
    Tensor &norm_cache, size_t batch_size, size_t channels, size_t spatial_size,
    const std::string &flow_id) {
  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(
        flow_id, cpu::batchnorm_nchw::run_forward_fused<IO_T>, input->data_as<IO_T>(),
        batch_mean_fixed->data_as<float>(), batch_inv_std->data_as<float>(),
        running_mean->data_as<float>(), running_var->data_as<float>(), gamma->data_as<float>(),
        beta->data_as<float>(), output->data_as<IO_T>(), norm_cache->data_as<float>(), batch_size,
        channels, spatial_size, momentum_, epsilon_, affine_);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_gpu_task(
        flow_id, cuda::batchnorm_nchw::run_forward_fused<IO_T>, input->data_as<IO_T>(),
        batch_mean_fixed->data_as<float>(), batch_inv_std->data_as<float>(),
        running_mean->data_as<float>(), running_var->data_as<float>(), gamma->data_as<float>(),
        beta->data_as<float>(), output->data_as<IO_T>(), norm_cache->data_as<float>(), batch_size,
        channels, spatial_size, momentum_, epsilon_, affine_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for run_forward_fused");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyBatchNormLayer::run_backward_fused(
    const Tensor &grad_output, const Tensor &norm_input, const Tensor &inv_std, const Tensor &gamma,
    Tensor &d_gamma, Tensor &d_beta, Tensor &grad_input, size_t batch_size, size_t channels,
    size_t spatial_size, const std::string &flow_id) {
  if (grad_output->device_type() == DeviceType::CPU) {
    return create_cpu_task(
        flow_id, cpu::batchnorm_nchw::run_backward_fused<IO_T>, grad_output->data_as<IO_T>(),
        norm_input->data_as<float>(), inv_std->data_as<float>(), gamma->data_as<float>(),
        d_gamma->data_as<float>(), d_beta->data_as<float>(), grad_input->data_as<IO_T>(),
        batch_size, channels, spatial_size, affine_);
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    return create_gpu_task(
        flow_id, cuda::batchnorm_nchw::run_backward_fused<IO_T>, grad_output->data_as<IO_T>(),
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

std::string LegacyBatchNormLayer::type() const { return "legacy_batchnorm"; }

LayerConfig LegacyBatchNormLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["num_features"] = num_features_;
  config.parameters["epsilon"] = epsilon_;
  config.parameters["momentum"] = momentum_;
  config.parameters["affine"] = affine_;
  return config;
}

std::unique_ptr<Layer> LegacyBatchNormLayer::clone() const {
  return std::make_unique<LegacyBatchNormLayer>(num_features_, epsilon_, momentum_, affine_,
                                                this->name_);
}

std::vector<size_t>
LegacyBatchNormLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

void LegacyBatchNormLayer::save_state(std::ofstream &file) {
  if (affine_) {
    gamma_->save(file);
    beta_->save(file);
  }
  running_mean_->save(file);
  running_var_->save(file);
}

void LegacyBatchNormLayer::load_state(std::ifstream &file) {
  if (this->device_ == nullptr) {
    std::cerr << "ERR: Device not set for LegacyBatchNormLayer when loading state." << std::endl;
    return;
  }
  if (affine_) {
    gamma_ = load(file, this->device_);
    beta_ = load(file, this->device_);
  }
  running_mean_ = load(file, this->device_);
  running_var_ = load(file, this->device_);
  this->initialized_ = true;
}

void LegacyBatchNormLayer::collect_parameters(std::vector<Tensor> &params) {
  if (affine_) {
    params.push_back(gamma_);
    params.push_back(beta_);
  }
}

void LegacyBatchNormLayer::collect_gradients(std::vector<Tensor> &grads) {
  if (affine_) {
    grads.push_back(gamma_gradients_);
    grads.push_back(beta_gradients_);
  }
}

std::unique_ptr<Layer> LegacyBatchNormLayer::create_from_config(const LayerConfig &config) {
  size_t num_features = config.get<size_t>("num_features");
  float epsilon = config.get<float>("epsilon");
  float momentum = config.get<float>("momentum");
  bool affine = config.get<bool>("affine");

  return std::make_unique<LegacyBatchNormLayer>(num_features, epsilon, momentum, affine,
                                                config.name);
}

uint64_t LegacyBatchNormLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t spatial_size = num_elements / (batch_size * num_features_);

  uint64_t mean_flops = batch_size * spatial_size * num_features_;

  uint64_t var_flops = 2 * num_elements + mean_flops;

  uint64_t norm_flops = 3 * num_elements;

  uint64_t affine_flops = affine_ ? (2 * num_elements) : 0;

  return mean_flops + var_flops + norm_flops + affine_flops;
}

uint64_t LegacyBatchNormLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t spatial_size = num_elements / (batch_size * num_features_);

  uint64_t param_grad_flops = affine_ ? (2 * batch_size * spatial_size * num_features_) : 0;

  uint64_t input_grad_flops = 9 * num_elements;

  return param_grad_flops + input_grad_flops;
}

size_t LegacyBatchNormLayer::cached_memory_bytes() const {
  size_t total_bytes = 0;

  size_t normalized_cache_size = 0;
  // Cached normalized outputs per micro-batch
  for (const auto &pair : micro_batch_normalized_) {
    size_t dtype_size = get_dtype_size(pair.second->data_type());
    normalized_cache_size += pair.second->size() * dtype_size;
  }

  size_t inv_std_cache_size = 0;
  // Cached inverse stddev per micro-batch
  for (const auto &pair : micro_batch_inv_std_) {
    size_t dtype_size = get_dtype_size(pair.second->data_type());
    inv_std_cache_size += pair.second->size() * dtype_size;
  }

  size_t mean_cache_size = 0;
  // Cached fixed batch means per micro-batch
  for (const auto &pair : batch_mean_fixed_) {
    size_t dtype_size = get_dtype_size(pair.second->data_type());
    mean_cache_size += pair.second->size() * dtype_size;
  }

  total_bytes += normalized_cache_size + inv_std_cache_size + mean_cache_size;
  total_bytes += Layer::cached_memory_bytes();
  return total_bytes;
}

} // namespace tnn
