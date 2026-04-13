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
#endif

namespace tnn {

LayerNormLayer::LayerNormLayer(size_t normalized_shape, float epsilon, bool affine,
                               const std::string &name)
    : ParameterizedLayer(name),
      normalized_shape_(normalized_shape),
      epsilon_(epsilon),
      affine_(affine) {}

LayerNormLayer::~LayerNormLayer() {
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

void LayerNormLayer::init_impl() {
  gamma_->fill(1.0f);
  beta_->fill(0.0f);

  gamma_gradients_->fill(0.0f);
  beta_gradients_->fill(0.0f);
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayer::run_forward(const ConstTensor &input, const Tensor &output,
                                                  const ConstTensor &gamma, const ConstTensor &beta,
                                                  size_t batch_size, size_t channels,
                                                  flowHandle_t handle) const {
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

  if (get_engine_type() == EngineType::CPU) {
    return create_cpu_task(
        this->flow_handle_, cpu::layer_norm::run_forward<Compute_T>, input->data_as<Compute_T>(),
        output->data_as<Compute_T>(), gamma ? gamma->data_as<Compute_T>() : nullptr,
        beta ? beta->data_as<Compute_T>() : nullptr, batch_size, channels, epsilon_);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(
        this->flow_handle_, cuda::layer_norm::run_forward<Compute_T>, input->data_as<Compute_T>(),
        output->data_as<Compute_T>(), gamma ? gamma->data_as<Compute_T>() : nullptr,
        beta ? beta->data_as<Compute_T>() : nullptr, batch_size, channels, epsilon_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for run_forward");
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayer::run_backward(
    const ConstTensor &grad_output, const ConstTensor &input, const ConstTensor &gamma,
    const Tensor &grad_input, const Tensor &gamma_gradients, const Tensor &beta_gradients,
    size_t batch_size, size_t channels, flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LayerNormLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (grad_output->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma && gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LayerNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (get_engine_type() == EngineType::CPU) {
    return create_cpu_task(this->flow_handle_, cpu::layer_norm::run_backward<Compute_T>,
                           grad_output->data_as<Compute_T>(), input->data_as<Compute_T>(),
                           gamma ? gamma->data_as<Compute_T>() : nullptr,
                           grad_input->data_as<Compute_T>(),
                           gamma_gradients ? gamma_gradients->data_as<Compute_T>() : nullptr,
                           beta_gradients ? beta_gradients->data_as<Compute_T>() : nullptr,
                           batch_size, channels, epsilon_);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(this->flow_handle_, cuda::layer_norm::run_backward<Compute_T>,
                            grad_output->data_as<Compute_T>(), input->data_as<Compute_T>(),
                            gamma ? gamma->data_as<Compute_T>() : nullptr,
                            grad_input->data_as<Compute_T>(),
                            gamma_gradients ? gamma_gradients->data_as<Compute_T>() : nullptr,
                            beta_gradients ? beta_gradients->data_as<Compute_T>() : nullptr,
                            batch_size, channels, epsilon_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for run_backward");
  }
}

#ifdef USE_CUDNN
void LayerNormLayer::build_graph(const Vec<size_t> &input_shape) const {
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
    auto io_data_type = cuda::cudnn::to_cudnn_datatype(io_dtype_);
    auto compute_type = cuda::cudnn::to_cudnn_datatype(compute_dtype_);
    fe_handle_cache[shape_key] = cuda::cudnn_layer_norm::initialize_fe_handle(
        shared_handle, io_data_type, compute_type, new_stats);
    stats_cache[shape_key] = new_stats;
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayer::cudnn_run_forward(
    cuda::cudnn_layer_norm::feHandle_t *fe_handle, LayerNormStats &stats, const ConstTensor &input,
    const Tensor &output, const ConstTensor &gamma, const ConstTensor &beta, const Tensor &mean,
    const Tensor &inv_variance, const Tensor &workspace, size_t batch_size, size_t channels,
    flowHandle_t handle) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("LayerNormLayer IO_T and Param_T must be the same type");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(handle, cuda::cudnn_layer_norm::run_forward, fe_handle, stats,
                          input->data(), gamma ? gamma->data() : nullptr,
                          beta ? beta->data() : nullptr, output->data(), mean->data(),
                          inv_variance->data(), workspace->data());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayer::cudnn_run_backward(
    cuda::cudnn_layer_norm::feHandle_t *fe_handle, LayerNormStats &stats,
    const ConstTensor &grad_output, const ConstTensor &input, const ConstTensor &gamma,
    const Tensor &grad_input, const Tensor &gamma_gradients, const Tensor &beta_gradients,
    const ConstTensor &mean, const ConstTensor &inv_variance, const Tensor &workspace,
    size_t batch_size, size_t channels, flowHandle_t handle) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("LayerNormLayer IO_T and Param_T must be the same type");
  }
  if (grad_output->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(handle, cuda::cudnn_layer_norm::run_backward, fe_handle, stats,
                          grad_output->data(), input->data(), gamma ? gamma->data() : nullptr,
                          mean->data(), inv_variance->data(), grad_input->data(),
                          gamma_gradients ? gamma_gradients->data() : nullptr,
                          beta_gradients ? beta_gradients->data() : nullptr, workspace->data());
}

Tensor LayerNormLayer::cudnn_forward(const ConstTensor &input, size_t mb_id) {
  const auto &shape = input->shape();
  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  build_graph(shape);

  Tensor output = get_output_tensor(shape);

  size_t shape_key = get_shape_hash({batch_size, channels});

  cuda::cudnn_layer_norm::feHandle_t *fe_handle = nullptr;

  fe_handle = fe_handle_cache.at(shape_key);
  LayerNormStats &current_stats = stats_cache.at(shape_key);

  size_t workspace_size = current_stats.fwd_workspace_size;
  Tensor cudnn_workspace = this->get_workspace({workspace_size}, DType_t::BYTE);

  Tensor batch_mean = this->get_cache_tensor({batch_size}, compute_dtype_);
  Tensor batch_invar = this->get_cache_tensor({batch_size}, compute_dtype_);
  set_mutable_cache(mb_id, "batch_mean", batch_mean);
  set_mutable_cache(mb_id, "batch_invar", batch_invar);

  if (this->is_training_) {
    set_immutable_cache(mb_id, "input", input);
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_run_forward, fe_handle, current_stats, input, output, gamma_,
                                 beta_, batch_mean, batch_invar, cudnn_workspace, batch_size,
                                 channels, this->flow_handle_);

  return output;
}

Tensor LayerNormLayer::cudnn_backward(const ConstTensor &grad_output, size_t mb_id) {
  ConstTensor &input = this->get_immutable_cache(mb_id, "input");
  if (!input) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const auto &shape = input->shape();
  Tensor grad_input = get_output_tensor(shape);

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
  Tensor cudnn_workspace = this->get_workspace({workspace_size}, DType_t::BYTE);

  // Retrieve cached mean and inv_variance from forward pass (like batch norm)
  const Tensor &batch_mean = this->get_mutable_cache(mb_id, "batch_mean");
  const Tensor &batch_invar = this->get_mutable_cache(mb_id, "batch_invar");
  if (!batch_mean || !batch_invar) {
    throw std::runtime_error(
        "No cached batch statistics found for micro-batch ID in LayerNormLayer: " +
        std::to_string(mb_id));
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_run_backward, fe_handle, current_stats, grad_output, input,
                                 gamma_, grad_input, gamma_gradients_, beta_gradients_, batch_mean,
                                 batch_invar, cudnn_workspace, batch_size, channels,
                                 this->flow_handle_);

  return grad_input;
}
#endif

Tensor LayerNormLayer::def_forward(const ConstTensor &input, size_t mb_id) {
  const auto &shape = input->shape();
  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  Tensor output = get_output_tensor(shape);

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_forward, input, output, gamma_, beta_, batch_size, channels,
                                 this->flow_handle_);

  return output;
}

Tensor LayerNormLayer::def_backward(const ConstTensor &grad_output, size_t mb_id) {
  ConstTensor &input = this->get_immutable_cache(mb_id, "input");
  if (!input) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const auto &shape = input->shape();
  Tensor grad_input = get_output_tensor(shape);

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

Tensor LayerNormLayer::forward_impl(const ConstTensor &input, size_t mb_id) {
  if (this->is_training_) {
    ConstTensor &cached_input = this->get_immutable_cache(mb_id, "input");
    cached_input = input;
  }

#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return cudnn_forward(input, mb_id);
  } else
#endif
  {
    return def_forward(input, mb_id);
  }
}

Tensor LayerNormLayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return cudnn_backward(grad_output, mb_id);
  } else
#endif
  {
    return def_backward(grad_output, mb_id);
  }
}

size_t LayerNormLayer::fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 2) return 0;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }
  return batch_size * sizeof(float) + batch_size * sizeof(float);  // mean + inv_variance
}

size_t LayerNormLayer::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 2) return 0;
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  build_graph(shape);
  size_t channels = shape.back();
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) batch_size *= shape[i];
  size_t shape_key = get_shape_hash({batch_size, channels});
  const LayerNormStats &stats = stats_cache.at(shape_key);
  auto output_shapes = this->output_shapes(input_shapes);
  return stats.fwd_workspace_size + get_shapes_bytes(output_shapes, io_dtype_);
#else
  return 0;
#endif
}

size_t LayerNormLayer::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  return fwd_workspace(input_shapes);
}

size_t LayerNormLayer::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 2) return 0;
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  build_graph(shape);
  size_t channels = shape.back();
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) batch_size *= shape[i];
  size_t shape_key = get_shape_hash({batch_size, channels});
  const LayerNormStats &stats = stats_cache.at(shape_key);
  return stats.bwd_workspace_size + get_shapes_bytes(input_shapes, io_dtype_);
#else
  return 0;
#endif
}

LayerConfig LayerNormLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("normalized_shape", normalized_shape_);
  config.set("epsilon", epsilon_);
  config.set("affine", affine_);
  return config;
}

std::unique_ptr<LayerNormLayer> LayerNormLayer::create_from_config(const LayerConfig &config) {
  size_t normalized_shape = config.get<size_t>("normalized_shape");
  float epsilon = config.get<float>("epsilon", 1e-5f);
  bool affine = config.get<bool>("affine", true);
  return std::make_unique<LayerNormLayer>(normalized_shape, epsilon, affine, config.name);
}

}  // namespace tnn
