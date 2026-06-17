/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/layer_norm_layer.hpp"

#include <stdexcept>
#include <type_traits>

#include "device/task.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/common/layer_norm.hpp"
#include "nn/layers_impl/cpu/layer_norm_ops.hpp"
#include "utils/misc.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/layer_norm_ops.hpp"
#endif
#ifdef USE_CUDNN
#include "cuda/cudnn/common.hpp"
#include "device/cuda/cuda_context.hpp"
#include "nn/layers_impl/cuda/cudnn_layer_norm_ops.hpp"
#include "ops/cuda/kernels.hpp"
#endif

namespace synet {

LayerNormLayerImpl::LayerNormLayerImpl(size_t normalized_shape, float epsilon, bool affine,
                                       const std::string &name)
    : SISOLayerImpl(name),
      normalized_shape_(normalized_shape),
      epsilon_(epsilon),
      affine_(affine) {}

LayerNormLayerImpl::~LayerNormLayerImpl() {
#ifdef USE_CUDNN
  for (auto &pair : fe_handle_cache) {
    if (pair.second) {
      cuda::cudnn_layer_norm::destroy_fe_handle(pair.second);
    }
  }
  fe_handle_cache.clear();
  stats_cache.clear();
#endif
}

void LayerNormLayerImpl::init_impl() {
  gamma_.fill(1.0f);
  beta_.fill(0.0f);

  gamma_gradients_.fill(0.0f);
  beta_gradients_.fill(0.0f);
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayerImpl::run_forward(const Tensor &input, Tensor &output,
                                                      const Tensor &gamma, const Tensor &beta,
                                                      size_t batch_size, size_t channels,
                                                      flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LayerNormLayerImpl mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (input.dtype() != dtype_of<IO_T>() || output.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma && gamma.dtype() != dtype_of<Param_T>()) {
    throw std::runtime_error("LayerNormLayerImpl gamma dtype mismatch with dispatch Param_T");
  }

  if (get_engine_type() == EngineType::CPU) {
    return create_cpu_task(
        this->flow_handle_, cpu::layer_norm::run_forward<Compute_T>, input.data_as<Compute_T>(),
        output.data_as<Compute_T>(), gamma ? gamma.data_as<Compute_T>() : nullptr,
        beta ? beta.data_as<Compute_T>() : nullptr, batch_size, channels, epsilon_);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(
        this->flow_handle_, cuda::layer_norm::run_forward<Compute_T>, input.data_as<Compute_T>(),
        output.data_as<Compute_T>(), gamma ? gamma.data_as<Compute_T>() : nullptr,
        beta ? beta.data_as<Compute_T>() : nullptr, batch_size, channels, epsilon_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for run_forward");
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayerImpl::run_backward(const Tensor &grad_output,
                                                       const Tensor &input, const Tensor &gamma,
                                                       Tensor &grad_input, Tensor &gamma_gradients,
                                                       Tensor &beta_gradients, size_t batch_size,
                                                       size_t channels, flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LayerNormLayerImpl mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (grad_output.dtype() != dtype_of<IO_T>() || grad_input.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma && gamma.dtype() != dtype_of<Param_T>()) {
    throw std::runtime_error("LayerNormLayerImpl gamma dtype mismatch with dispatch Param_T");
  }

  if (get_engine_type() == EngineType::CPU) {
    return create_cpu_task(this->flow_handle_, cpu::layer_norm::run_backward<Compute_T>,
                           grad_output.data_as<Compute_T>(), input.data_as<Compute_T>(),
                           gamma ? gamma.data_as<Compute_T>() : nullptr,
                           grad_input.data_as<Compute_T>(),
                           gamma_gradients ? gamma_gradients.data_as<Compute_T>() : nullptr,
                           beta_gradients ? beta_gradients.data_as<Compute_T>() : nullptr,
                           batch_size, channels, epsilon_);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(this->flow_handle_, cuda::layer_norm::run_backward<Compute_T>,
                            grad_output.data_as<Compute_T>(), input.data_as<Compute_T>(),
                            gamma ? gamma.data_as<Compute_T>() : nullptr,
                            grad_input.data_as<Compute_T>(),
                            gamma_gradients ? gamma_gradients.data_as<Compute_T>() : nullptr,
                            beta_gradients ? beta_gradients.data_as<Compute_T>() : nullptr,
                            batch_size, channels, epsilon_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for run_backward");
  }
}

#ifdef USE_CUDNN
void LayerNormLayerImpl::build_graph(const Vec<size_t> &input_shape) const {
  size_t batch_size = 1;
  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
    batch_size *= input_shape[i];
  }

  size_t channels = input_shape.back();
  size_t shape_key = get_shape_hash({batch_size, channels});

  if (fe_handle_cache.find(shape_key) == fe_handle_cache.end()) {
    LayerNormStats new_stats;
    init_layer_norm_stats(new_stats, batch_size, channels, affine_, epsilon_);

    cudnnHandle_t shared_handle = CUDAContext::getCudnnHandle();
    auto io_dtype = cuda::cudnn::to_cudnn_datatype(io_dtype_);
    auto compute_type = cuda::cudnn::to_cudnn_datatype(compute_dtype_);
    fe_handle_cache[shape_key] = cuda::cudnn_layer_norm::initialize_fe_handle(
        shared_handle, io_dtype, compute_type, new_stats);
    stats_cache[shape_key] = new_stats;
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayerImpl::cudnn_run_forward(
    cuda::cudnn_layer_norm::feHandle_t *fe_handle, LayerNormStats &stats, const Tensor &input,
    Tensor &output, const Tensor &gamma, const Tensor &beta, Tensor &mean, Tensor &inv_variance,
    Tensor &workspace, size_t batch_size, size_t channels, flowHandle_t handle) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("LayerNormLayerImpl IO_T and Param_T must be the same type");
  }
  if (input.dtype() != dtype_of<IO_T>() || output.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(handle, cuda::cudnn_layer_norm::run_forward, fe_handle, stats,
                          input.data_as<void>(), gamma ? gamma.data_as<void>() : nullptr,
                          beta ? beta.data_as<void>() : nullptr, output.data_as<void>(),
                          mean.data_as<void>(), inv_variance.data_as<void>(),
                          workspace.data_as<void>());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayerImpl::cudnn_run_backward(
    cuda::cudnn_layer_norm::feHandle_t *fe_handle, LayerNormStats &stats, const Tensor &grad_output,
    const Tensor &input, const Tensor &gamma, Tensor &grad_input, Tensor &gamma_gradients,
    Tensor &beta_gradients, const Tensor &mean, const Tensor &inv_variance, Tensor &workspace,
    size_t batch_size, size_t channels, flowHandle_t handle) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("LayerNormLayerImpl IO_T and Param_T must be the same type");
  }
  if (grad_output.dtype() != dtype_of<IO_T>() || grad_input.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(
      handle, cuda::cudnn_layer_norm::run_backward, fe_handle, stats, grad_output.data_as<void>(),
      input.data_as<void>(), gamma ? gamma.data_as<void>() : nullptr, mean.data_as<void>(),
      inv_variance.data_as<void>(), grad_input.data_as<void>(),
      gamma_gradients ? gamma_gradients.data_as<void>() : nullptr,
      beta_gradients ? beta_gradients.data_as<void>() : nullptr, workspace.data_as<void>());
}

Tensor LayerNormLayerImpl::cudnn_forward(const Tensor &input, Residuals &residuals) {
  if (this->is_training_) {
    residuals["input"] = input;
  }

  const auto &shape = input.shape();
  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  build_graph(shape);

  Tensor batch_mean = this->get_tensor({batch_size}, compute_dtype_);
  Tensor batch_invar = this->get_tensor({batch_size}, compute_dtype_);
  residuals["batch_mean"] = batch_mean;
  residuals["batch_invar"] = batch_invar;

  Tensor output = get_tensor(shape, io_dtype_);

  size_t shape_key = get_shape_hash({batch_size, channels});

  cuda::cudnn_layer_norm::feHandle_t *fe_handle = nullptr;

  fe_handle = fe_handle_cache.at(shape_key);
  LayerNormStats &current_stats = stats_cache.at(shape_key);

  size_t workspace_size = current_stats.fwd_workspace_size;
  Tensor cudnn_workspace = this->get_tensor({workspace_size}, DType_t::BYTE);

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_run_forward, fe_handle, current_stats, input, output, gamma_,
                                 beta_, batch_mean, batch_invar, cudnn_workspace, batch_size,
                                 channels, this->flow_handle_);

  return output;
}

Tensor LayerNormLayerImpl::cudnn_backward(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];
  if (!input) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const auto &shape = input.shape();
  Tensor grad_input = get_tensor(shape, io_dtype_);

  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  size_t shape_key = get_shape_hash({batch_size, channels});
  cuda::cudnn_layer_norm::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  LayerNormStats &current_stats = stats_cache.at(shape_key);

  size_t workspace_size = current_stats.bwd_workspace_size;
  Tensor cudnn_workspace = this->get_tensor({workspace_size}, DType_t::BYTE);

  // Retrieve cached mean and inv_variance from forward pass (like batch norm)
  const Tensor &batch_mean = residuals["batch_mean"];
  const Tensor &batch_invar = residuals["batch_invar"];
  if (!batch_mean || !batch_invar) {
    throw std::runtime_error("No cached batch statistics found in LayerNormLayerImpl");
  }

  Tensor dscale_scratch_ = this->get_tensor({normalized_shape_}, compute_dtype_);
  Tensor dbias_scratch_ = this->get_tensor({normalized_shape_}, compute_dtype_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_run_backward, fe_handle, current_stats, grad_output, input,
                                 gamma_, grad_input, dscale_scratch_, dbias_scratch_, batch_mean,
                                 batch_invar, cudnn_workspace, batch_size, channels,
                                 this->flow_handle_);

  DISPATCH_DTYPE(param_dtype_, T, {
    create_cuda_task(this->flow_handle_, ops::cuda::cuda_axpy<T>, T{1},
                     dscale_scratch_.data_as<T>(), gamma_gradients_.data_as<T>(),
                     normalized_shape_);
    create_cuda_task(this->flow_handle_, ops::cuda::cuda_axpy<T>, T{1}, dbias_scratch_.data_as<T>(),
                     beta_gradients_.data_as<T>(), normalized_shape_);
  });

  return grad_input;
}
#endif

Tensor LayerNormLayerImpl::def_forward(const Tensor &input, Residuals &residuals) {
  const auto &shape = input.shape();
  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  Tensor output = get_tensor(shape, io_dtype_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_forward, input, output, gamma_, beta_, batch_size, channels,
                                 this->flow_handle_);

  return output;
}

Tensor LayerNormLayerImpl::def_backward(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];
  if (!input) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const auto &shape = input.shape();
  Tensor grad_input = get_tensor(shape, io_dtype_);

  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_backward, grad_output, input, gamma_, grad_input,
                                 gamma_gradients_, beta_gradients_, batch_size, channels,
                                 this->flow_handle_);

  return grad_input;
}

Tensor LayerNormLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (this->is_training_) {
    residuals["input"] = input;
  }

#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return cudnn_forward(input, residuals);
  } else
#endif
  {
    return def_forward(input, residuals);
  }
}

Tensor LayerNormLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return cudnn_backward(grad_output, residuals);
  } else
#endif
  {
    return def_backward(grad_output, residuals);
  }
}

LayerConfig LayerNormLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("normalized_shape", normalized_shape_);
  config.set("epsilon", epsilon_);
  config.set("affine", affine_);
  return config;
}

std::shared_ptr<LayerNormLayerImpl> LayerNormLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t normalized_shape = config.get<size_t>("normalized_shape");
  float epsilon = config.get<float>("epsilon", 1e-5f);
  bool affine = config.get<bool>("affine", true);
  return std::make_shared<LayerNormLayerImpl>(normalized_shape, epsilon, affine, config.name);
}

}  // namespace synet
