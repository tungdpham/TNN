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
#include "nn/layers_impl/common/layer_norm.hpp"
#include "nn/layers_impl/cpu/layer_norm_ops.hpp"
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

void LayerNormLayer::init_params() {
  if (this->initialized_) return;

  if (affine_) {
    gamma_ = make_param_tensor({normalized_shape_});
    beta_ = make_param_tensor({normalized_shape_});
    gamma_->fill(1);
    beta_->fill(0);

    gamma_gradients_ = make_param_tensor({normalized_shape_});
    beta_gradients_ = make_param_tensor({normalized_shape_});
    gamma_gradients_->fill(0);
    beta_gradients_->fill(0);
  }

  this->initialized_ = true;
}

size_t LayerNormLayer::get_shape_hash(size_t n, size_t c) const {
  size_t seed = 0;
  auto hash_combine = [&](size_t v) {
    seed ^= std::hash<size_t>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  };
  hash_combine(n);
  hash_combine(c);
  return seed;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayer::layer_norm_forward(
    const ConstTensor &input, const Tensor &output, const ConstTensor &gamma,
    const ConstTensor &beta, size_t batch_size, size_t channels, const std::string &flow_id) const {
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

  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task(
        "default", cpu::layer_norm::layer_norm_forward<Compute_T>, input->data_as<Compute_T>(),
        output->data_as<Compute_T>(), gamma ? gamma->data_as<Compute_T>() : nullptr,
        beta ? beta->data_as<Compute_T>() : nullptr, batch_size, channels, epsilon_);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task(
        "default", cuda::layer_norm::layer_norm_forward<Compute_T>, input->data_as<Compute_T>(),
        output->data_as<Compute_T>(), gamma ? gamma->data_as<Compute_T>() : nullptr,
        beta ? beta->data_as<Compute_T>() : nullptr, batch_size, channels, epsilon_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for layer_norm_forward");
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayer::layer_norm_backward(
    const ConstTensor &gradient, const ConstTensor &input, const ConstTensor &gamma,
    const Tensor &grad_input, const Tensor &gamma_gradients, const Tensor &beta_gradients,
    size_t batch_size, size_t channels, const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LayerNormLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (gradient->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (gamma && gamma->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LayerNormLayer gamma dtype mismatch with dispatch Param_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task("default", cpu::layer_norm::layer_norm_backward<Compute_T>,
                           gradient->data_as<Compute_T>(), input->data_as<Compute_T>(),
                           gamma ? gamma->data_as<Compute_T>() : nullptr,
                           grad_input->data_as<Compute_T>(),
                           gamma_gradients ? gamma_gradients->data_as<Compute_T>() : nullptr,
                           beta_gradients ? beta_gradients->data_as<Compute_T>() : nullptr,
                           batch_size, channels, epsilon_);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_cuda_task("default", cuda::layer_norm::layer_norm_backward<Compute_T>,
                            gradient->data_as<Compute_T>(), input->data_as<Compute_T>(),
                            gamma ? gamma->data_as<Compute_T>() : nullptr,
                            grad_input->data_as<Compute_T>(),
                            gamma_gradients ? gamma_gradients->data_as<Compute_T>() : nullptr,
                            beta_gradients ? beta_gradients->data_as<Compute_T>() : nullptr,
                            batch_size, channels, epsilon_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for layer_norm_backward");
  }
}

#ifdef USE_CUDNN
template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayer::cudnn_layer_norm_forward(
    cuda::cudnn_layer_norm::feHandle_t *fe_handle, LayerNormStats &stats, const ConstTensor &input,
    const Tensor &output, const ConstTensor &gamma, const ConstTensor &beta, const Tensor &mean,
    const Tensor &inv_variance, const Tensor &workspace, size_t batch_size, size_t channels,
    const std::string &flow_id) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("LayerNormLayer IO_T and Param_T must be the same type");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(flow_id, cuda::cudnn_layer_norm::run_forward, fe_handle, stats,
                          input->data(), gamma ? gamma->data() : nullptr,
                          beta ? beta->data() : nullptr, output->data(), mean->data(),
                          inv_variance->data(), workspace->data());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LayerNormLayer::cudnn_layer_norm_backward(
    cuda::cudnn_layer_norm::feHandle_t *fe_handle, LayerNormStats &stats,
    const ConstTensor &gradient, const ConstTensor &input, const ConstTensor &gamma,
    const Tensor &grad_input, const Tensor &gamma_gradients, const Tensor &beta_gradients,
    const ConstTensor &mean, const ConstTensor &inv_variance, const Tensor &workspace,
    size_t batch_size, size_t channels, const std::string &flow_id) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("LayerNormLayer IO_T and Param_T must be the same type");
  }
  if (gradient->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LayerNormLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(flow_id, cuda::cudnn_layer_norm::run_backward, fe_handle, stats,
                          gradient->data(), input->data(), gamma ? gamma->data() : nullptr,
                          mean->data(), inv_variance->data(), grad_input->data(),
                          gamma_gradients ? gamma_gradients->data() : nullptr,
                          beta_gradients ? beta_gradients->data() : nullptr, workspace->data());
}

void LayerNormLayer::cudnn_forward(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  const auto &shape = input->shape();
  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  output->ensure(shape);

  size_t shape_key = get_shape_hash(batch_size, channels);

  cuda::cudnn_layer_norm::feHandle_t *fe_handle = nullptr;
  size_t io_dtype_size = get_dtype_size(io_dtype_);

  if (fe_handle_cache.find(shape_key) == fe_handle_cache.end()) {
    LayerNormStats new_stats;
    init_layer_norm_stats(new_stats, batch_size, channels, affine_, epsilon_);

    auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
    if (!cuda_context) {
      throw std::runtime_error("LayerNormLayer requires CUDAContext for cuDNN operations");
    }
    cudnnHandle_t shared_handle = cuda_context->getCudnnHandle();
    auto io_data_type = cuda::cudnn::to_cudnn_datatype(io_dtype_);
    auto compute_type = cuda::cudnn::to_cudnn_datatype(compute_dtype_);
    fe_handle_cache[shape_key] = cuda::cudnn_layer_norm::initialize_fe_handle(
        shared_handle, io_data_type, compute_type, new_stats);
    stats_cache[shape_key] = new_stats;
  }

  fe_handle = fe_handle_cache.at(shape_key);
  LayerNormStats &current_stats = stats_cache.at(shape_key);

  size_t max_workspace_size =
      std::max(current_stats.fwd_workspace_size, current_stats.bwd_workspace_size);
  size_t workspace_elements = (max_workspace_size + io_dtype_size - 1) / io_dtype_size;
  Tensor cudnn_workspace = this->get_buffer({workspace_elements});

  // Cache mean and inv_variance for backward pass (like batch norm)
  Tensor &batch_mean = this->get_mutable_tensor(mb_id, "batch_mean");
  Tensor &batch_invar = this->get_mutable_tensor(mb_id, "batch_invar");
  if (batch_mean == nullptr) {
    batch_mean = this->get_buffer({batch_size}, compute_dtype_);
  }
  if (batch_invar == nullptr) {
    batch_invar = this->get_buffer({batch_size}, compute_dtype_);
  }

  if (this->is_training_) {
    ConstTensor &cached_input = this->get_cached_tensor(mb_id, "input");
    cached_input = input;
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_layer_norm_forward, fe_handle, current_stats, input, output,
                                 gamma_, beta_, batch_mean, batch_invar, cudnn_workspace,
                                 batch_size, channels, "default");
}

void LayerNormLayer::cudnn_backward(const ConstTensor &gradient, const Tensor &grad_input,
                                    size_t mb_id) {
  ConstTensor &input = this->get_cached_tensor(mb_id, "input");
  if (!input) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const auto &shape = input->shape();
  grad_input->ensure(shape);

  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  size_t shape_key = get_shape_hash(batch_size, channels);
  cuda::cudnn_layer_norm::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  LayerNormStats &current_stats = stats_cache.at(shape_key);

  size_t io_dtype_size = get_dtype_size(io_dtype_);
  size_t max_workspace_size =
      std::max(current_stats.fwd_workspace_size, current_stats.bwd_workspace_size);
  size_t workspace_elements = (max_workspace_size + io_dtype_size - 1) / io_dtype_size;
  Tensor cudnn_workspace = this->get_buffer({workspace_elements});

  // Retrieve cached mean and inv_variance from forward pass (like batch norm)
  const Tensor &batch_mean = this->get_mutable_tensor(mb_id, "batch_mean");
  const Tensor &batch_invar = this->get_mutable_tensor(mb_id, "batch_invar");
  if (!batch_mean || !batch_invar) {
    throw std::runtime_error(
        "No cached batch statistics found for micro-batch ID in LayerNormLayer: " +
        std::to_string(mb_id));
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_layer_norm_backward, fe_handle, current_stats, gradient,
                                 input, gamma_, grad_input, gamma_gradients_, beta_gradients_,
                                 batch_mean, batch_invar, cudnn_workspace, batch_size, channels,
                                 "default");
}
#endif

void LayerNormLayer::def_forward(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  const auto &shape = input->shape();
  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  output->ensure(shape);

  DISPATCH_ON_3_DTYPES_TO_METHOD(layer_norm_forward, input, output, gamma_, beta_, batch_size,
                                 channels, "default");
}

void LayerNormLayer::def_backward(const ConstTensor &gradient, const Tensor &grad_input,
                                  size_t mb_id) {
  ConstTensor &input = this->get_cached_tensor(mb_id, "input");
  if (!input) {
    throw std::runtime_error("LayerNorm backward called without forward for this micro-batch");
  }

  const auto &shape = input->shape();
  grad_input->ensure(shape);

  size_t last_dim = shape.back();
  size_t channels = last_dim;
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(layer_norm_backward, gradient, input, gamma_, grad_input,
                                 gamma_gradients_, beta_gradients_, batch_size, channels,
                                 "default");
}

void LayerNormLayer::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  if (this->is_training_) {
    ConstTensor &cached_input = this->get_cached_tensor(mb_id, "input");
    cached_input = input;
  }

#ifdef USE_CUDNN
  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_forward(input, output, mb_id);
  } else
#endif
  {
    def_forward(input, output, mb_id);
  }
}

void LayerNormLayer::backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                                   size_t mb_id) {
#ifdef USE_CUDNN
  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_backward(gradient, grad_input, mb_id);
  } else
#endif
  {
    def_backward(gradient, grad_input, mb_id);
  }
}

uint64_t LayerNormLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() < 2) return 0;
  size_t elements = 1;
  for (size_t s : input_shape) elements *= s;
  return elements * 8;
}

uint64_t LayerNormLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() < 2) return 0;
  size_t elements = 1;
  for (size_t s : input_shape) elements *= s;
  return elements * 16;
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

std::unique_ptr<Layer> LayerNormLayer::clone() const {
  return std::make_unique<LayerNormLayer>(normalized_shape_, epsilon_, affine_, this->name_);
}

void LayerNormLayer::collect_parameters(std::vector<Tensor> &params) {
  if (affine_) {
    params.push_back(gamma_);
    params.push_back(beta_);
  }
}

void LayerNormLayer::collect_gradients(std::vector<Tensor> &grads) {
  if (affine_) {
    grads.push_back(gamma_gradients_);
    grads.push_back(beta_gradients_);
  }
}

std::unique_ptr<LayerNormLayer> LayerNormLayer::create_from_config(const LayerConfig &config) {
  size_t normalized_shape = config.get<size_t>("normalized_shape");
  float epsilon = config.get<float>("epsilon", 1e-5f);
  bool affine = config.get<bool>("affine", true);
  return std::make_unique<LayerNormLayer>(normalized_shape, epsilon, affine, config.name);
}

}  // namespace tnn
