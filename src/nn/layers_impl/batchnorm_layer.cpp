/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/batchnorm_layer.hpp"

#include <cstddef>

#include "device/task.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/common/batchnorm.hpp"
#include "type/type.hpp"
#include "utils/misc.hpp"
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
    : ParameterizedLayer(name),
      num_features_(num_features),
      epsilon_(epsilon),
      momentum_(momentum),
      affine_(affine),
      use_relu_(use_relu) {}

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

void BatchNormLayer::init_impl() {
  gamma_->fill(1.0f);
  beta_->fill(0.0f);

  running_mean_->fill(0.0f);
  running_var_->fill(1.0f);

  gamma_gradients_->fill(0.0f);
  beta_gradients_->fill(0.0f);

  dummy_mean_gradients_->fill(0.0f);
  dummy_var_gradients_->fill(0.0f);
}

/**
 * @brief Forward pass for BatchNormLayer
 * @param input Tensor in NHWC format
 * @param output Tensor in NHWC format
 * @param mb_id Micro-batch identifier for caching
 */
Tensor BatchNormLayer::forward_impl(const ConstTensor &input, size_t mb_id) {
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
  if (this->device().device_type() == DeviceType::GPU) {
    return cudnn_forward(input, mb_id);
  } else
#endif
  {
    throw std::runtime_error("BatchNormLayer forward only implemented for GPU with cuDNN");
  }
}

Tensor BatchNormLayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
#ifdef USE_CUDNN
  if (this->device().device_type() == DeviceType::GPU) {
    return cudnn_backward(grad_output, mb_id);
  } else
#endif
  {
    throw std::runtime_error("BatchNormLayer backward only implemented for GPU with cuDNN");
  }
}

#ifdef USE_CUDNN
void BatchNormLayer::build_graph(const Vec<size_t> &input_shape) const {
  size_t batch_size = input_shape[0];
  size_t height = input_shape[1];
  size_t width = input_shape[2];
  size_t channels = input_shape[3];
  size_t shape_key = get_shape_hash(input_shape);
  if (fe_handle_cache.find(shape_key) == fe_handle_cache.end()) {
    BatchNormStats new_stats;
    init_batchnorm_stats(new_stats, batch_size, height, width, channels, epsilon_, momentum_,
                         use_relu_);
    cudnnHandle_t shared_handle = CUDAContext::getCudnnHandle();
    cudnnDataType_t io_data_type = cuda::cudnn_batchnorm::get_cudnn_data_type(io_dtype_);
    cudnnDataType_t compute_data_type = cuda::cudnn_batchnorm::get_cudnn_data_type(compute_dtype_);
    fe_handle_cache[shape_key] = cuda::cudnn_batchnorm::initialize_fe_handle(
        shared_handle, io_data_type, compute_data_type, new_stats);
    stats_cache[shape_key] = new_stats;
  }
}

Tensor BatchNormLayer::cudnn_forward(const ConstTensor &input, size_t mb_id) {
  const size_t channels = input->dimension(3);
  if (num_features_ != channels) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features." +
                                std::to_string(num_features_) + ", but got " +
                                std::to_string(channels));
  }

  build_graph(input->shape());

  size_t shape_key = get_shape_hash(input->shape());

  cuda::cudnn_batchnorm::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  BatchNormStats &current_stats = stats_cache.at(shape_key);
  round_workspace_size(current_stats);

  if (this->is_training_) {
    set_immutable_cache(mb_id, "input", input);
  }

  if (this->is_training_) {
    Tensor batch_invar = get_cache_tensor({num_features_}, DType_t::FP32);
    Tensor batch_mean = get_cache_tensor({num_features_}, DType_t::FP32);
    set_mutable_cache(mb_id, "batch_invar", batch_invar);
    set_mutable_cache(mb_id, "batch_mean", batch_mean);
    Tensor relu_mask;
    if (use_relu_) {
      relu_mask = get_cache_tensor(input->shape(), DType_t::BOOL);
      set_mutable_cache(mb_id, "relu_mask", relu_mask);
    }
    Tensor output = get_output_tensor(input->shape());

    Tensor workspace = this->get_workspace({current_stats.fwd_workspace_size}, DType_t::BYTE);
    DISPATCH_ON_3_DTYPES_TO_METHOD(forward_training_task, fe_handle, current_stats, input, output,
                                   gamma_, beta_, running_mean_, running_var_, running_mean_,
                                   running_var_, batch_mean, batch_invar, relu_mask, workspace,
                                   this->flow_handle_);
    return output;
  } else {
    Tensor output = get_output_tensor(input->shape());

    Tensor workspace = this->get_workspace({current_stats.inf_workspace_size}, DType_t::BYTE);
    DISPATCH_ON_3_DTYPES_TO_METHOD(forward_inference_task, fe_handle, current_stats, input, output,
                                   gamma_, beta_, running_mean_, running_var_, workspace,
                                   this->flow_handle_);
    return output;
  }
}

Tensor BatchNormLayer::cudnn_backward(const ConstTensor &grad_output, size_t mb_id) {
  ConstTensor &input = this->get_immutable_cache(mb_id, "input");

  Tensor &batch_mean = this->get_mutable_cache(mb_id, "batch_mean");
  Tensor &batch_invar = this->get_mutable_cache(mb_id, "batch_invar");
  Tensor &relu_mask = this->get_mutable_cache(mb_id, "relu_mask");

  const auto &input_shape = input->shape();

  size_t shape_key = get_shape_hash(input_shape);
  cuda::cudnn_batchnorm::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  BatchNormStats &current_stats = stats_cache.at(shape_key);

  Tensor grad_input = get_output_tensor(grad_output->shape());

  Tensor workspace = this->get_workspace({current_stats.bwd_workspace_size}, DType_t::BYTE);

  DISPATCH_ON_3_DTYPES_TO_METHOD(backward_task, fe_handle, current_stats, grad_output, relu_mask,
                                 input, grad_input, gamma_, gamma_gradients_, beta_gradients_,
                                 batch_mean, batch_invar, workspace, this->flow_handle_);

  batch_mean = nullptr;
  batch_invar = nullptr;
  relu_mask = nullptr;

  return grad_input;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> BatchNormLayer::forward_training_task(
    cuda::cudnn_batchnorm::feHandle_t *fe_handle, BatchNormStats &stats, const ConstTensor &input,
    const Tensor &output, const ConstTensor &gamma, const ConstTensor &beta,
    const Tensor &prev_running_mean, const Tensor &prev_running_var,
    const Tensor &next_running_mean, const Tensor &next_running_var, const Tensor &batch_mean,
    const Tensor &batch_invar, const Tensor &relu_mask, const Tensor &workspace,
    flowHandle_t handle) {
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("BatchNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("BatchNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device().device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
    return create_cuda_task(handle, cuda::cudnn_batchnorm::run_forward_training, fe_handle, stats,
                            input->data(), gamma->data(), beta->data(), output->data(),
                            running_mean_->data(), running_var_->data(), running_mean_->data(),
                            running_var_->data(), batch_mean->data(), batch_invar->data(),
                            use_relu_ ? relu_mask->data() : nullptr, workspace->data());
#endif
  } else {
    throw std::runtime_error("BatchNormLayer forward only implemented for GPU with cuDNN");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> BatchNormLayer::forward_inference_task(
    cuda::cudnn_batchnorm::feHandle_t *fe_handle, BatchNormStats &stats, const ConstTensor &input,
    const Tensor &output, const ConstTensor &gamma, const ConstTensor &beta,
    const ConstTensor &saved_mean, const ConstTensor &saved_var, const Tensor &workspace,
    flowHandle_t handle) {
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("BatchNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("BatchNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device().device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
    return create_cuda_task(handle, cuda::cudnn_batchnorm::run_forward_inference, fe_handle, stats,
                            input->data(), gamma->data(), beta->data(), saved_mean->data(),
                            saved_var->data(), output->data(), workspace->data());
#endif
  } else {
    throw std::runtime_error("BatchNormLayer forward only implemented for GPU with cuDNN");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> BatchNormLayer::backward_task(
    cuda::cudnn_batchnorm::feHandle_t *fe_handle, BatchNormStats &stats,
    const ConstTensor &grad_output, const ConstTensor &relu_mask, const ConstTensor &input,
    const Tensor &grad_input, const ConstTensor &gamma, const Tensor &gamma_gradients,
    const Tensor &beta_gradients, const ConstTensor &batch_mean, const ConstTensor &batch_invar,
    const Tensor &workspace, flowHandle_t handle) {
  if (grad_output->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("BatchNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("BatchNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device().device_type() == DeviceType::GPU) {
#ifdef USE_CUDNN
    return create_cuda_task(handle, cuda::cudnn_batchnorm::run_backward, fe_handle, stats,
                            input->data(), grad_output->data(),
                            use_relu_ ? relu_mask->data() : nullptr, gamma->data(),
                            grad_input->data(), gamma_gradients->data(), beta_gradients->data(),
                            batch_mean->data(), batch_invar->data(), workspace->data());
#endif
  } else {
    throw std::runtime_error("BatchNormLayer backward only implemented for GPU with cuDNN");
  }
  return nullptr;
}
#endif

size_t BatchNormLayer::fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const {
  size_t total_cache = 0;
  total_cache += get_shapes_bytes(input_shapes, io_dtype_);       // cache input for backward
  total_cache += affine_ ? 2 * num_features_ * sizeof(fp32) : 0;  // cache batch mean and invar
  if (use_relu_) {
    total_cache += get_shapes_bytes(input_shapes, DType_t::BOOL);  // cache ReLU mask
  }
  return total_cache;
}

size_t BatchNormLayer::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 4) return 0;
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  size_t shape_key = get_shape_hash(shape);
  build_graph(shape);
  BatchNormStats stats = stats_cache.at(shape_key);
  round_workspace_size(stats);
  auto output_shapes = this->output_shapes(input_shapes);
  size_t total_workspace = stats.fwd_workspace_size + get_shapes_bytes(output_shapes, io_dtype_);
  return total_workspace;
#else
  return 0;
#endif
}

size_t BatchNormLayer::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 4) return 0;
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  size_t shape_key = get_shape_hash(shape);
  build_graph(shape);
  BatchNormStats stats = stats_cache.at(shape_key);
  round_workspace_size(stats);
  auto output_shapes = this->output_shapes(input_shapes);
  return get_shapes_bytes(output_shapes, io_dtype_) + stats.inf_workspace_size;
#else
  return 0;
#endif  // USE_CUDNN
}

size_t BatchNormLayer::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 4) return 0;
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  size_t shape_key = get_shape_hash(shape);
  build_graph(shape);
  BatchNormStats stats = stats_cache.at(shape_key);
  round_workspace_size(stats);
  auto output_shapes = this->output_shapes(input_shapes);
  return get_shapes_bytes(output_shapes, io_dtype_) + stats.bwd_workspace_size;

#else
  return 0;
#endif
}

LayerConfig BatchNormLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("num_features", num_features_);
  config.set("epsilon", epsilon_);
  config.set("momentum", momentum_);
  config.set("affine", affine_);
  config.set("use_relu", use_relu_);
  return config;
}

Vec<size_t> BatchNormLayer::compute_output_shape(const Vec<size_t> &input_shape) const {
  return input_shape;
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

}  // namespace tnn
