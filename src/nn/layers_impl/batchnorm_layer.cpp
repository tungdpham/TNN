/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/batchnorm_layer.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/base_layer.hpp"
#include "nn/layers_impl/common/batchnorm.hpp"
#include "type/type.hpp"
#ifdef USE_CUDNN
#include "device/cuda/cuda_context.hpp"
#include "nn/layers_impl/cuda/cudnn_batchnorm_ops.hpp"
#endif
#include <cmath>
#include <memory>
#include <stdexcept>

namespace tnn {

BatchNormLayer::BatchNormLayer(size_t num_features, float epsilon, float momentum, bool affine,
                               const std::string &name)
    : ParameterizedLayer(name), num_features_(num_features), epsilon_(epsilon), momentum_(momentum),
      affine_(affine) {}

BatchNormLayer::~BatchNormLayer() {
#ifdef USE_CUDNN
  if (fe_handle) {
    cuda::cudnn_batchnorm::destroy_fe_handle(fe_handle);
  }
#endif
}

void BatchNormLayer::init_params() {
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

/**
 * @brief Forward pass for BatchNormLayer
 * @param input Tensor in NHWC format
 * @param output Tensor in NHWC format
 * @param micro_batch_id Micro-batch identifier for caching
 */
void BatchNormLayer::forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  if (input->dims() < 4) {
    throw std::invalid_argument("BatchNorm: Input tensor must have at least 4 dimensions got " +
                                std::to_string(input->dims()) + " dims");
  }
  if (input->dimension(3) != num_features_) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features" +
                                std::to_string(num_features_) + ", but got " +
                                std::to_string(input->dimension(3)));
  }

#ifdef USE_CUDNN
  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_forward(input, output, micro_batch_id);
  } else
#endif
  {
    throw std::runtime_error("BatchNormLayer forward only implemented for GPU with cuDNN");
  }
}

void BatchNormLayer::backward_impl(const Tensor &gradient, Tensor &grad_input,
                                   size_t micro_batch_id) {
#ifdef USE_CUDNN
  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_backward(gradient, grad_input, micro_batch_id);
  } else
#endif
  {
    throw std::runtime_error("BatchNormLayer backward only implemented for GPU with cuDNN");
  }
}

void BatchNormLayer::cudnn_forward(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  const size_t batch_size = input->dimension(0);
  const size_t height = input->dimension(1);
  const size_t width = input->dimension(2);
  const size_t channels = input->dimension(3);
  if (num_features_ != channels) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features." +
                                std::to_string(num_features_) + ", but got " +
                                std::to_string(channels));
  }

  output->ensure(input->shape(), this->device_);

  // check if we need to (re)initialize cuDNN handle
  bool dimensions_changed =
      (batch_size != stats_.batch_size) || (height != stats_.height) || (width != stats_.width);

  if (!fe_handle) {
    init_batchnorm_stats(stats_, batch_size, channels, height, width, epsilon_, momentum_);
    auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
    if (!cuda_context) {
      throw std::runtime_error("BatchNormLayer requires CUDAContext for cuDNN operations");
    }
    cudnnHandle_t shared_handle = cuda_context->getCudnnHandle();
    cudnnDataType_t data_type = cuda::cudnn_batchnorm::get_cudnn_data_type(io_dtype_);

    fe_handle = cuda::cudnn_batchnorm::initialize_fe_handle(shared_handle, data_type, stats_);
  } else if (dimensions_changed) {
    init_batchnorm_stats(stats_, batch_size, channels, height, width, epsilon_, momentum_);
    cuda::cudnn_batchnorm::destroy_fe_handle(fe_handle);

    auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
    cudnnHandle_t shared_handle = cuda_context->getCudnnHandle();
    cudnnDataType_t data_type = cuda::cudnn_batchnorm::get_cudnn_data_type(io_dtype_);

    fe_handle = cuda::cudnn_batchnorm::initialize_fe_handle(shared_handle, data_type, stats_);
  }
  round_workspace_size(stats_);

  if (this->is_training_) {
    // Tensor &cached_input = micro_batch_inputs_cache_[micro_batch_id];
    // if (cached_input == nullptr) {
    //   cached_input = make_io_tensor(input->shape());
    // }
    // cached_input->ensure(input->shape(), this->device_);
    // input->copy_to(cached_input);
    micro_batch_inputs_cache_[micro_batch_id] = input;
  }

  Tensor &batch_invar = micro_batch_invar_[micro_batch_id];
  Tensor &batch_mean = micro_batch_mean_[micro_batch_id];
  if (batch_invar == nullptr) {
    batch_invar = make_io_tensor({num_features_});
  }
  if (batch_mean == nullptr) {
    batch_mean = make_io_tensor({num_features_});
  }

  size_t io_dtype_size = get_dtype_size(io_dtype_);

  if (this->is_training_) {
    size_t workspace_elems = (stats_.fwd_workspace_size + io_dtype_size - 1) / io_dtype_size;
    Tensor workspace = this->get_buffer({workspace_elems}, io_dtype_);
    DISPATCH_ON_3_DTYPES_TO_METHOD(forward_training_task, input, output, gamma_, beta_,
                                   running_mean_, running_var_, running_mean_, running_var_,
                                   batch_mean, batch_invar, workspace, "default");
  } else {
    size_t workspace_elems = (stats_.inf_workspace_size + io_dtype_size - 1) / io_dtype_size;
    Tensor workspace = this->get_buffer({workspace_elems}, io_dtype_);
    DISPATCH_ON_3_DTYPES_TO_METHOD(forward_inference_task, input, output, gamma_, beta_,
                                   running_mean_, running_var_, workspace, "default");
  }
}

void BatchNormLayer::cudnn_backward(const Tensor &gradient, Tensor &grad_input,
                                    size_t micro_batch_id) {
  auto it_input_cache = micro_batch_inputs_cache_.find(micro_batch_id);
  if (it_input_cache == micro_batch_inputs_cache_.end()) {
    throw std::runtime_error("No cached input found for micro-batch ID in BatchNormLayer: " +
                             std::to_string(micro_batch_id));
  }

  auto it_batch_mean = micro_batch_mean_.find(micro_batch_id);
  auto it_batch_invar = micro_batch_invar_.find(micro_batch_id);

  if (it_batch_mean == micro_batch_mean_.end() || it_batch_invar == micro_batch_invar_.end()) {
    throw std::runtime_error(
        "No cached batch statistics found for micro-batch ID in BatchNormLayer: " +
        std::to_string(micro_batch_id));
  }

  size_t io_dtype_size = get_dtype_size(io_dtype_);
  Tensor workspace = this->get_buffer(
      {(stats_.bwd_workspace_size + io_dtype_size - 1) / io_dtype_size}, io_dtype_);

  grad_input->ensure(gradient->shape(), this->device_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(backward_task, gradient, it_input_cache->second, grad_input,
                                 gamma_, gamma_gradients_, beta_gradients_, it_batch_mean->second,
                                 it_batch_invar->second, workspace, "default");
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> BatchNormLayer::forward_training_task(
    const Tensor &input, Tensor &output, const Tensor &gamma, const Tensor &beta,
    Tensor &prev_running_mean, Tensor &prev_running_var, Tensor &next_running_mean,
    Tensor &next_running_var, Tensor &batch_mean, Tensor &batch_invar, Tensor &workspace,
    const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "BatchNormLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("BatchNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("BatchNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
    return create_gpu_task(flow_id, cuda::cudnn_batchnorm::run_forward_training, fe_handle, stats_,
                           input->data(), gamma->data(), beta->data(), output->data(),
                           running_mean_->data(), running_var_->data(), running_mean_->data(),
                           running_var_->data(), batch_mean->data(), batch_invar->data(),
                           workspace->data());
#endif
  } else {
    throw std::runtime_error("BatchNormLayer forward only implemented for GPU with cuDNN");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task>
BatchNormLayer::forward_inference_task(const Tensor &input, Tensor &output, const Tensor &gamma,
                                       const Tensor &beta, const Tensor &saved_mean,
                                       const Tensor &saved_invar, Tensor &workspace,
                                       const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "BatchNormLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("BatchNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("BatchNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
    return create_gpu_task(flow_id, cuda::cudnn_batchnorm::run_forward_inference, fe_handle, stats_,
                           input->data(), gamma->data(), beta->data(), saved_mean->data(),
                           saved_invar->data(), output->data(), workspace->data());
#endif
  } else {
    throw std::runtime_error("BatchNormLayer forward only implemented for GPU with cuDNN");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task>
BatchNormLayer::backward_task(const Tensor &gradient, const Tensor &input, Tensor &grad_input,
                              const Tensor &gamma, Tensor &gamma_gradients, Tensor &beta_gradients,
                              const Tensor &batch_mean, const Tensor &batch_invar,
                              Tensor &workspace, const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "BatchNormLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (gradient->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("BatchNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("BatchNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
    return create_gpu_task(flow_id, cuda::cudnn_batchnorm::run_backward, fe_handle, stats_,
                           input->data(), gradient->data(), gamma->data(), grad_input->data(),
                           gamma_gradients->data(), beta_gradients->data(), batch_mean->data(),
                           batch_invar->data(), workspace->data());
#endif
  } else {
    throw std::runtime_error("BatchNormLayer backward only implemented for GPU with cuDNN");
  }
  return nullptr;
}

std::string BatchNormLayer::type() const { return "batchnorm"; }

LayerConfig BatchNormLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["num_features"] = num_features_;
  config.parameters["epsilon"] = epsilon_;
  config.parameters["momentum"] = momentum_;
  config.parameters["affine"] = affine_;
  return config;
}

std::unique_ptr<Layer> BatchNormLayer::clone() const {
  return std::make_unique<BatchNormLayer>(num_features_, epsilon_, momentum_, affine_, this->name_);
}

std::vector<size_t>
BatchNormLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

void BatchNormLayer::save_state(std::ofstream &file) {
  if (affine_) {
    gamma_->save(file);
    beta_->save(file);
  }
  running_mean_->save(file);
  running_var_->save(file);
}

void BatchNormLayer::load_state(std::ifstream &file) {
  if (this->device_ == nullptr) {
    std::cerr << "ERR: Device not set for BatchNormLayer when loading state." << std::endl;
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

void BatchNormLayer::collect_parameters(std::vector<Tensor> &params) {
  if (affine_) {
    params.push_back(gamma_);
    params.push_back(beta_);
  }
}

void BatchNormLayer::collect_gradients(std::vector<Tensor> &grads) {
  if (affine_) {
    grads.push_back(gamma_gradients_);
    grads.push_back(beta_gradients_);
  }
}

std::unique_ptr<Layer> BatchNormLayer::create_from_config(const LayerConfig &config) {
  size_t num_features = config.get<size_t>("num_features");
  float epsilon = config.get<float>("epsilon");
  float momentum = config.get<float>("momentum");
  bool affine = config.get<bool>("affine");

  return std::make_unique<BatchNormLayer>(num_features, epsilon, momentum, affine, config.name);
}

uint64_t BatchNormLayer::forward_flops(const std::vector<size_t> &input_shape) const {
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

uint64_t BatchNormLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t spatial_size = num_elements / (batch_size * num_features_);
  uint64_t param_grad_flops = affine_ ? (2 * batch_size * spatial_size * num_features_) : 0;
  uint64_t input_grad_flops = 9 * num_elements;
  return param_grad_flops + input_grad_flops;
}

size_t BatchNormLayer::cached_memory_bytes() const {
  size_t total_bytes = 0;
  size_t normalized_cache_size = 0;
  size_t inv_std_cache_size = 0;
  for (const auto &pair : micro_batch_invar_) {
    size_t dtype_size = get_dtype_size(pair.second->data_type());
    inv_std_cache_size += pair.second->size() * dtype_size;
  }
  size_t mean_cache_size = 0;
  for (const auto &pair : micro_batch_mean_) {
    size_t dtype_size = get_dtype_size(pair.second->data_type());
    mean_cache_size += pair.second->size() * dtype_size;
  }
  total_bytes += normalized_cache_size + inv_std_cache_size + mean_cache_size;
  return total_bytes;
}

} // namespace tnn
