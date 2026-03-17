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
void BatchNormLayer::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
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
    cudnn_forward(input, output, mb_id);
  } else
#endif
  {
    throw std::runtime_error("BatchNormLayer forward only implemented for GPU with cuDNN");
  }
}

void BatchNormLayer::backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                                   size_t mb_id) {
#ifdef USE_CUDNN
  if (this->device().device_type() == DeviceType::GPU) {
    cudnn_backward(grad_output, grad_input, mb_id);
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

void BatchNormLayer::cudnn_forward(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  const size_t channels = input->dimension(3);
  if (num_features_ != channels) {
    throw std::invalid_argument("BatchNorm: Input channels must match num_features." +
                                std::to_string(num_features_) + ", but got " +
                                std::to_string(channels));
  }

  output->ensure(input->shape());

  build_graph(input->shape());

  size_t shape_key = get_shape_hash(input->shape());

  cuda::cudnn_batchnorm::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  BatchNormStats &current_stats = stats_cache.at(shape_key);
  round_workspace_size(current_stats);

  if (this->is_training_) {
    ConstTensor &cached_input = this->get_cached_tensor(mb_id, "input");
    cached_input = input;
  }

  if (this->is_training_) {
    Tensor &batch_invar = this->get_mutable_tensor(mb_id, "batch_invar");
    Tensor &batch_mean = this->get_mutable_tensor(mb_id, "batch_mean");
    Tensor &relu_mask = this->get_mutable_tensor(mb_id, "relu_mask");
    if (batch_invar == nullptr) {
      batch_invar = this->make_io_tensor({num_features_});
    }
    if (batch_mean == nullptr) {
      batch_mean = this->make_io_tensor({num_features_});
    }
    if (use_relu_ && relu_mask == nullptr) {
      relu_mask = this->get_tensor(input->shape(), DType_t::UINT8_T);
    }
    Tensor workspace = this->get_workspace({current_stats.fwd_workspace_size}, DType_t::BYTE);
    DISPATCH_ON_3_DTYPES_TO_METHOD(forward_training_task, fe_handle, current_stats, input, output,
                                   gamma_, beta_, running_mean_, running_var_, running_mean_,
                                   running_var_, batch_mean, batch_invar, relu_mask, workspace,
                                   this->flow_handle_);
  } else {
    Tensor workspace = this->get_workspace({current_stats.inf_workspace_size}, DType_t::BYTE);
    DISPATCH_ON_3_DTYPES_TO_METHOD(forward_inference_task, fe_handle, current_stats, input, output,
                                   gamma_, beta_, running_mean_, running_var_, workspace,
                                   this->flow_handle_);
  }
}

void BatchNormLayer::cudnn_backward(const ConstTensor &grad_output, const Tensor &grad_input,
                                    size_t mb_id) {
  ConstTensor &input = this->get_cached_tensor(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for micro-batch ID in BatchNormLayer: " +
                             std::to_string(mb_id));
  }

  const Tensor &batch_mean = this->get_mutable_tensor(mb_id, "batch_mean");
  const Tensor &batch_invar = this->get_mutable_tensor(mb_id, "batch_invar");
  const Tensor &relu_mask = this->get_mutable_tensor(mb_id, "relu_mask");
  if (!batch_mean || !batch_invar || (use_relu_ && !relu_mask)) {
    throw std::runtime_error(
        "No cached batch statistics found for micro-batch ID in BatchNormLayer: " +
        std::to_string(mb_id));
  }

  const auto &input_shape = input->shape();

  size_t shape_key = get_shape_hash(input_shape);
  cuda::cudnn_batchnorm::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  BatchNormStats &current_stats = stats_cache.at(shape_key);

  Tensor workspace = this->get_workspace({current_stats.bwd_workspace_size}, DType_t::BYTE);
  grad_input->ensure(grad_output->shape());

  DISPATCH_ON_3_DTYPES_TO_METHOD(backward_task, fe_handle, current_stats, grad_output, relu_mask,
                                 input, grad_input, gamma_, gamma_gradients_, beta_gradients_,
                                 batch_mean, batch_invar, workspace, this->flow_handle_);
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

size_t BatchNormLayer::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 4) return 0;
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  size_t shape_key = get_shape_hash(shape);
  build_graph(shape);
  BatchNormStats stats = stats_cache.at(shape_key);
  round_workspace_size(stats);
  return stats.fwd_workspace_size;
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
  return stats.inf_workspace_size;
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
  return stats.bwd_workspace_size;
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

std::vector<size_t> BatchNormLayer::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
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
