/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/dense_layer.hpp"

#include "device/task.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/cpu/dense_ops.hpp"
#include "utils/misc.hpp"
#ifdef USE_CUDNN
#include "cuda/cudnn/common.hpp"
#include "device/cuda/cuda_context.hpp"
#include "math/cuda/cudnn_gemm.hpp"
#endif
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/dense_ops.hpp"
#endif
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "nn/layers_impl/parameterized_layer.hpp"
#include "type/type.hpp"

namespace tnn {

DenseLayer::DenseLayer(size_t input_features, size_t output_features, bool use_bias,
                       const std::string &name)
    : ParameterizedLayer(name),
      input_features_(input_features),
      output_features_(output_features),
      use_bias_(use_bias) {}

DenseLayer::~DenseLayer() {
#ifdef USE_CUDNN
  for (auto &pair : fe_handle_cache) {
    cuda::cudnn_gemm::destroy_fe_handle(pair.second);
  }
  fe_handle_cache.clear();
#endif
}

void DenseLayer::init_impl() {
  float bound = static_cast<float>(1.0 / std::sqrt(static_cast<double>(input_features_)));

  if (this->use_seed_) {
    weights_->fill_random_uniform(-bound, bound, this->srand_seed_);
  } else {
    weights_->fill_random_uniform(-bound, bound);
  }

  if (use_bias_) {
    if (this->use_seed_) {
      bias_->fill_random_uniform(-bound, bound, this->srand_seed_);
    } else {
      bias_->fill_random_uniform(-bound, bound);
    }
  }

  weight_gradients_->fill(0.0f);
  if (use_bias_) {
    bias_gradients_->fill(0.0f);
  }
}

Tensor DenseLayer::forward_impl(const ConstTensor &input, size_t mb_id) {
  const Vec<size_t> &in_shape = input->shape();
  size_t last_dim = in_shape.back();

  if (last_dim != input_features_) {
    std::cerr << "Input last dimension: " << last_dim << " features, expected: " << input_features_
              << " features" << std::endl;
    throw std::invalid_argument("Input feature size mismatch in DenseLayer");
  }

  if (this->is_training_) {
    set_immutable_cache(mb_id, "input", input);
  }

#ifdef USE_CUDNN
  if (this->device().device_type() == DeviceType::GPU) {
    return cudnn_forward(input, mb_id);
  } else
#endif
  {
    return def_forward(input, mb_id);
  }
}

Tensor DenseLayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
  if (grad_output->shape().back() != output_features_) {
    throw std::invalid_argument("Gradient feature size mismatch in DenseLayer. Expected " +
                                std::to_string(output_features_) + " features in grad_output" +
                                " but got " + std::to_string(grad_output->shape().back()) +
                                " features in grad_output" + ".");
  }

  ConstTensor &input = this->get_immutable_cache(mb_id, "input");
  Tensor grad_input = get_output_tensor(input->shape());

#ifdef USE_CUDNN
  if (this->device().device_type() == DeviceType::GPU) {
    return cudnn_backward(grad_output, mb_id);
  } else
#endif
  {
    return def_backward(grad_output, mb_id);
  }
}

Tensor DenseLayer::def_forward(const ConstTensor &input, size_t mb_id) {
  Vec<size_t> input_shape = input->shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < input->shape().size() - 1; ++i) {
    batch_size *= input->shape()[i];
  }

  Tensor output = get_output_tensor({batch_size, output_features_});
  if (this->device().device_type() == DeviceType::CPU) {
    DISPATCH_DTYPE(io_dtype_, T, {
      create_cpu_task(this->flow_handle_, cpu::legacy_dense::run_forward<T>, input->data_as<T>(),
                      weights_->data_as<T>(), output->data_as<T>(), batch_size, input_features_,
                      output_features_);
      if (use_bias_) {
        create_cpu_task(this->flow_handle_, cpu::legacy_dense::add_bias<T>, output->data_as<T>(),
                        bias_->data_as<T>(), batch_size, output_features_);
      }
    });
  } else {
    throw std::runtime_error("DenseLayer only supports CPU device in def_forward");
  }
  return output;
}

Tensor DenseLayer::def_backward(const ConstTensor &grad_output, size_t mb_id) {
  if (grad_output->shape().back() != output_features_) {
    throw std::invalid_argument("Gradient feature size mismatch in DenseLayer. Expected " +
                                std::to_string(output_features_) + " features in grad_output" +
                                " but got " + std::to_string(grad_output->shape().back()) +
                                " features in grad_output" + ".");
  }

  ConstTensor &input = this->get_immutable_cache(mb_id, "input");

  Vec<size_t> input_shape = input->shape();

  size_t batch_size = 1;

  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
    batch_size *= input_shape[i];
  }

  Tensor grad_input = get_output_tensor(input_shape);

  if (this->device().device_type() == DeviceType::CPU) {
    DISPATCH_DTYPE(io_dtype_, T, {
      create_cpu_task(this->flow_handle_, cpu::legacy_dense::run_weight_gradients<T>,
                      input->data_as<T>(), grad_output->data_as<T>(),
                      weight_gradients_->data_as<T>(), batch_size, input_features_,
                      output_features_);
      create_cpu_task(this->flow_handle_, cpu::legacy_dense::run_input_gradients<T>,
                      grad_output->data_as<T>(), weights_->data_as<T>(), grad_input->data_as<T>(),
                      batch_size, input_features_, output_features_);
      if (use_bias_) {
        create_cpu_task(this->flow_handle_, cpu::legacy_dense::run_bias_gradients<T>,
                        grad_output->data_as<T>(), bias_gradients_->data_as<T>(), batch_size,
                        output_features_);
      }
    });
  } else {
    throw std::runtime_error("DenseLayer only supports CPU device in def_backward");
  }
  return grad_input;
}

#ifdef USE_CUDNN
void DenseLayer::build_graph(const Vec<size_t> &input_shape) const {
  size_t batch_size = 1;
  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
    batch_size *= input_shape[i];
  }

  size_t shape_key = get_shape_hash({batch_size});

  if (fe_handle_cache.find(shape_key) == fe_handle_cache.end()) {
    cudnnDataType_t io_dtype = cuda::cudnn::to_cudnn_datatype(io_dtype_);
    cudnnDataType_t param_dtype = cuda::cudnn::to_cudnn_datatype(param_dtype_);
    cudnnDataType_t compute_dtype = cuda::cudnn::to_cudnn_datatype(compute_dtype_);
    cudnnHandle_t cudnn_handle = CUDAContext::getCudnnHandle();
    GemmStats stats;

    init_gemm_stats(stats, batch_size, output_features_, input_features_);

    cuda::cudnn_gemm::feHandle_t *handle = cuda::cudnn_gemm::initialize_fe_handle(
        cudnn_handle, io_dtype, param_dtype, compute_dtype, stats);
    fe_handle_cache[shape_key] = handle;
    stats_cache[shape_key] = stats;
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> DenseLayer::compute_bias_gradients(const ConstTensor &grad_output,
                                                         const Tensor &bias_gradient,
                                                         size_t batch_size, size_t output_features,
                                                         flowHandle_t handle) const {
  if (grad_output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("DenseLayer grad_output dtype mismatch with dispatch IO_T");
  }
  if (bias_gradient->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("DenseLayer bias grad_output dtype mismatch with dispatch Param_T");
  }
  if (this->device().device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "DenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(handle, cpu::legacy_dense::run_bias_gradients<IO_T>,
                           grad_output->data_as<IO_T>(), bias_gradient->data_as<IO_T>(), batch_size,
                           output_features);
  }
#ifdef USE_CUDA
  else if (this->device().device_type() == DeviceType::GPU) {
    return create_cuda_task(handle,
                            cuda::legacy_dense::run_bias_gradients<IO_T, Param_T, Compute_T>,
                            grad_output->data_as<IO_T>(), bias_gradient->data_as<Param_T>(),
                            batch_size, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_bias_gradients");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> DenseLayer::add_bias(const Tensor &output, const ConstTensor &bias,
                                           size_t batch_size, size_t output_features,
                                           flowHandle_t handle) const {
  if (output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("DenseLayer output dtype mismatch with dispatch IO_T");
  }
  if (bias->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("DenseLayer bias dtype mismatch with dispatch Param_T");
  }
  if (this->device().device_type() == DeviceType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "DenseLayer mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(handle, cpu::legacy_dense::add_bias<IO_T>, output->data_as<IO_T>(),
                           bias->data_as<IO_T>(), batch_size, output_features);
  }
#ifdef USE_CUDA
  else if (this->device().device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::legacy_dense::add_bias<IO_T, Param_T, Compute_T>,
                            output->data_as<IO_T>(), bias->data_as<Param_T>(), batch_size,
                            output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for add_bias");
  }
  return nullptr;
}

Tensor DenseLayer::cudnn_forward(const ConstTensor &input, size_t mb_id) {
  const Vec<size_t> &in_shape = input->shape();

  build_graph(in_shape);

  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }
  size_t shape_key = get_shape_hash({batch_size});

  cuda::cudnn_gemm::feHandle_t *handle = fe_handle_cache[shape_key];
  GemmStats &stats = stats_cache[shape_key];

  Tensor output = get_output_tensor({batch_size, output_features_});

  Tensor cudnn_workspace = this->get_workspace({stats.fwd_workspace_size}, DType_t::BYTE);

  create_cuda_task(this->flow_handle_, cuda::cudnn_gemm::run_forward, handle, stats, input->data(),
                   weights_->data(), output->data(), cudnn_workspace->data());

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(add_bias, output, bias_, batch_size, output_features_,
                                   this->flow_handle_);
  }

  return output;
}

Tensor DenseLayer::cudnn_backward(const ConstTensor &grad_output, size_t mb_id) {
  ConstTensor &input = this->get_immutable_cache(mb_id, "input");

  const Vec<size_t> &in_shape = input->shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  size_t shape_key = get_shape_hash({batch_size});
  cuda::cudnn_gemm::feHandle_t *handle = fe_handle_cache.at(shape_key);

  GemmStats &stats = stats_cache.at(shape_key);

  Tensor grad_input = get_output_tensor(input->shape());

  Tensor cudnn_workspace = this->get_workspace(
      {std::max(stats.dgrad_workspace_size, stats.wgrad_workspace_size)}, DType_t::BYTE);

  // Compute weight gradients
  create_cuda_task(this->flow_handle_, cuda::cudnn_gemm::run_wgrad, handle, stats, input->data(),
                   grad_output->data(), weight_gradients_->data(), cudnn_workspace->data());

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(compute_bias_gradients, grad_output, bias_gradients_, batch_size,
                                   output_features_, this->flow_handle_);
  }

  // Compute input gradients
  create_cuda_task(this->flow_handle_, cuda::cudnn_gemm::run_dgrad, handle, stats,
                   grad_output->data(), weights_->data(), grad_input->data(),
                   cudnn_workspace->data());

  return grad_input;
}
#endif

size_t DenseLayer::fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const {
  // Cache the input for backward pass
  auto &shape = input_shapes[0];
  size_t input_bytes = std::accumulate(shape.begin(), shape.end(), get_dtype_size(io_dtype_),
                                       std::multiplies<size_t>());
  return input_bytes;
}

size_t DenseLayer::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  build_graph(shape);
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }
  size_t shape_key = get_shape_hash({batch_size});
  const GemmStats &stats = stats_cache.at(shape_key);
  auto output_shapes = this->output_shapes(input_shapes);
  return stats.fwd_workspace_size + get_shapes_bytes(output_shapes, io_dtype_);
#else
  return 0;
#endif
}

size_t DenseLayer::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  return fwd_workspace(input_shapes);
}

size_t DenseLayer::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  build_graph(shape);
  size_t batch_size = 1;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    batch_size *= shape[i];
  }
  size_t shape_key = get_shape_hash({batch_size});
  const GemmStats &stats = stats_cache.at(shape_key);
  size_t max_workspace = std::max(stats.dgrad_workspace_size, stats.wgrad_workspace_size);
  auto output_shapes = this->output_shapes(input_shapes);
  return max_workspace + get_shapes_bytes(output_shapes, io_dtype_);
#else
  return 0;
#endif
}

LayerConfig DenseLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("input_features", input_features_);
  config.set("output_features", output_features_);
  config.set("use_bias", use_bias_);
  return config;
}

Vec<size_t> DenseLayer::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::runtime_error("DenseLayer::compute_output_shape: Input shape is empty.");
  }
  Vec<size_t> out_shape = input_shape;
  out_shape.back() = output_features_;
  return out_shape;
}

std::unique_ptr<DenseLayer> DenseLayer::create_from_config(const LayerConfig &config) {
  size_t input_features = config.get<size_t>("input_features");
  size_t output_features = config.get<size_t>("output_features");
  bool use_bias = config.get<bool>("use_bias");

  return std::make_unique<DenseLayer>(input_features, output_features, use_bias, config.name);
}

}  // namespace tnn
