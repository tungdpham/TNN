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
                               bool use_relu, const std::string &name)
    : ParameterizedLayer(name), num_features_(num_features), epsilon_(epsilon), momentum_(momentum),
      affine_(affine), use_relu_(use_relu) {}

BatchNormLayer::~BatchNormLayer() {
#ifdef USE_CUDNN
  for (auto &pair : fe_handle_cache) {
    if (pair.second) {
      cuda::cudnn_batchnorm::destroy_fe_handle(pair.second);
    }
  }
  fe_handle_cache.clear();
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

  dummy_mean_gradients_ = make_param_tensor({num_features_});
  dummy_var_gradients_ = make_param_tensor({num_features_});
  dummy_mean_gradients_->fill(0.0f);
  dummy_var_gradients_->fill(0.0f);

  this->initialized_ = true;
}

#ifdef USE_CUDNN
size_t BatchNormLayer::get_shape_hash(size_t n, size_t c, size_t h, size_t w) const {
  size_t seed = 0;
  auto hash_combine = [&](size_t v) {
    seed ^= std::hash<size_t>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  };
  hash_combine(n);
  hash_combine(c);
  hash_combine(h);
  hash_combine(w);
  return seed;
}
#endif

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

#ifdef USE_CUDNN
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

  size_t shape_key = get_shape_hash(batch_size, channels, height, width);

  cuda::cudnn_batchnorm::feHandle_t *fe_handle = nullptr;
  if (fe_handle_cache.find(shape_key) == fe_handle_cache.end()) {
    BatchNormStats new_stats;
    init_batchnorm_stats(new_stats, batch_size, channels, height, width, epsilon_, momentum_,
                         use_relu_);
    auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
    if (!cuda_context) {
      throw std::runtime_error("BatchNormLayer requires CUDAContext for cuDNN operations");
    }
    cudnnHandle_t shared_handle = cuda_context->getCudnnHandle();
    cudnnDataType_t io_data_type = cuda::cudnn_batchnorm::get_cudnn_data_type(io_dtype_);
    cudnnDataType_t compute_data_type = cuda::cudnn_batchnorm::get_cudnn_data_type(compute_dtype_);
    fe_handle_cache[shape_key] = cuda::cudnn_batchnorm::initialize_fe_handle(
        shared_handle, io_data_type, compute_data_type, new_stats);
    stats_cache[shape_key] = new_stats;
  }

  fe_handle = fe_handle_cache.at(shape_key);
  BatchNormStats &current_stats = stats_cache.at(shape_key);
  round_workspace_size(current_stats);

  if (this->is_training_) {
    Tensor &cached_input = this->get_cached_tensor(micro_batch_id, "input");
    cached_input = input;
  }

  Tensor &batch_invar = this->get_cached_tensor(micro_batch_id, "batch_invar");
  Tensor &batch_mean = this->get_cached_tensor(micro_batch_id, "batch_mean");
  if (batch_invar == nullptr) {
    batch_invar = make_io_tensor({num_features_});
  }
  if (batch_mean == nullptr) {
    batch_mean = make_io_tensor({num_features_});
  }

  size_t io_dtype_size = get_dtype_size(io_dtype_);

  if (this->is_training_) {
    size_t workspace_elems = (current_stats.fwd_workspace_size + io_dtype_size - 1) / io_dtype_size;
    Tensor workspace = this->get_buffer({workspace_elems}, io_dtype_);
    DISPATCH_ON_3_DTYPES_TO_METHOD(forward_training_task, fe_handle, current_stats, input, output,
                                   gamma_, beta_, running_mean_, running_var_, running_mean_,
                                   running_var_, batch_mean, batch_invar, workspace, "default");
  } else {
    size_t workspace_elems = (current_stats.inf_workspace_size + io_dtype_size - 1) / io_dtype_size;
    Tensor workspace = this->get_buffer({workspace_elems}, io_dtype_);
    DISPATCH_ON_3_DTYPES_TO_METHOD(forward_inference_task, fe_handle, current_stats, input, output,
                                   gamma_, beta_, running_mean_, running_var_, workspace,
                                   "default");
  }
}

void BatchNormLayer::cudnn_backward(const Tensor &gradient, Tensor &grad_input,
                                    size_t micro_batch_id) {
  Tensor &input = this->get_cached_tensor(micro_batch_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for micro-batch ID in BatchNormLayer: " +
                             std::to_string(micro_batch_id));
  }

  Tensor &batch_mean = this->get_cached_tensor(micro_batch_id, "batch_mean");
  Tensor &batch_invar = this->get_cached_tensor(micro_batch_id, "batch_invar");
  if (!batch_mean || !batch_invar) {
    throw std::runtime_error(
        "No cached batch statistics found for micro-batch ID in BatchNormLayer: " +
        std::to_string(micro_batch_id));
  }

  const auto &input_shape = input->shape();
  const size_t batch_size = input_shape[0];
  const size_t height = input_shape[1];
  const size_t width = input_shape[2];
  const size_t channels = input_shape[3];

  size_t shape_key = get_shape_hash(batch_size, channels, height, width);
  cuda::cudnn_batchnorm::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  BatchNormStats &current_stats = stats_cache.at(shape_key);

  size_t io_dtype_size = get_dtype_size(io_dtype_);
  Tensor workspace = this->get_buffer(
      {(current_stats.bwd_workspace_size + io_dtype_size - 1) / io_dtype_size}, io_dtype_);
  grad_input->ensure(gradient->shape(), this->device_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(backward_task, fe_handle, current_stats, gradient, input,
                                 grad_input, gamma_, gamma_gradients_, beta_gradients_, batch_mean,
                                 batch_invar, workspace, "default");

  input = nullptr;
  batch_mean = nullptr;
  batch_invar = nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> BatchNormLayer::forward_training_task(
    cuda::cudnn_batchnorm::feHandle_t *fe_handle, BatchNormStats &stats, const Tensor &input,
    Tensor &output, const Tensor &gamma, const Tensor &beta, Tensor &prev_running_mean,
    Tensor &prev_running_var, Tensor &next_running_mean, Tensor &next_running_var,
    Tensor &batch_mean, Tensor &batch_invar, Tensor &workspace, const std::string &flow_id) const {
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("BatchNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("BatchNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
    return create_gpu_task(flow_id, cuda::cudnn_batchnorm::run_forward_training, fe_handle, stats,
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
std::unique_ptr<Task> BatchNormLayer::forward_inference_task(
    cuda::cudnn_batchnorm::feHandle_t *fe_handle, BatchNormStats &stats, const Tensor &input,
    Tensor &output, const Tensor &gamma, const Tensor &beta, const Tensor &saved_mean,
    const Tensor &saved_invar, Tensor &workspace, const std::string &flow_id) const {
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("BatchNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("BatchNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
    return create_gpu_task(flow_id, cuda::cudnn_batchnorm::run_forward_inference, fe_handle, stats,
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
BatchNormLayer::backward_task(cuda::cudnn_batchnorm::feHandle_t *fe_handle, BatchNormStats &stats,
                              const Tensor &gradient, const Tensor &input, Tensor &grad_input,
                              const Tensor &gamma, Tensor &gamma_gradients, Tensor &beta_gradients,
                              const Tensor &batch_mean, const Tensor &batch_invar,
                              Tensor &workspace, const std::string &flow_id) const {
  if (gradient->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("BatchNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("BatchNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
    return create_gpu_task(flow_id, cuda::cudnn_batchnorm::run_backward, fe_handle, stats,
                           input->data(), gradient->data(), gamma->data(), grad_input->data(),
                           gamma_gradients->data(), beta_gradients->data(), batch_mean->data(),
                           batch_invar->data(), workspace->data());
#endif
  } else {
    throw std::runtime_error("BatchNormLayer backward only implemented for GPU with cuDNN");
  }
  return nullptr;
}
#endif

LayerConfig BatchNormLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["num_features"] = num_features_;
  config.parameters["epsilon"] = epsilon_;
  config.parameters["momentum"] = momentum_;
  config.parameters["affine"] = affine_;
  config.parameters["use_relu"] = use_relu_;
  return config;
}

std::unique_ptr<Layer> BatchNormLayer::clone() const {
  return std::make_unique<BatchNormLayer>(num_features_, epsilon_, momentum_, affine_, use_relu_,
                                          this->name_);
}

std::vector<size_t>
BatchNormLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

void BatchNormLayer::collect_parameters(std::vector<Tensor> &params) {
  if (affine_) {
    params.push_back(gamma_);
    params.push_back(beta_);
  }
  params.push_back(running_mean_);
  params.push_back(running_var_);
}

void BatchNormLayer::collect_gradients(std::vector<Tensor> &grads) {
  if (affine_) {
    grads.push_back(gamma_gradients_);
    grads.push_back(beta_gradients_);
  }
  grads.push_back(dummy_mean_gradients_);
  grads.push_back(dummy_var_gradients_);
}

std::unique_ptr<BatchNormLayer> BatchNormLayer::create_from_config(const LayerConfig &config) {
  size_t num_features = config.get<size_t>("num_features");
  float epsilon = config.get<float>("epsilon");
  float momentum = config.get<float>("momentum");
  bool affine = config.get<bool>("affine");
  bool use_relu = config.get<bool>("use_relu", false);

  return std::make_unique<BatchNormLayer>(num_features, epsilon, momentum, affine, use_relu,
                                          config.name);
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

} // namespace tnn
