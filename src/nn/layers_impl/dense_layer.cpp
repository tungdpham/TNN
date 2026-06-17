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

#include "type/type.hpp"

namespace synet {

DenseLayerImpl::DenseLayerImpl(size_t input_features, size_t output_features, bool use_bias,
                               const std::string &name)
    : SISOLayerImpl(name),
      input_features_(input_features),
      output_features_(output_features),
      use_bias_(use_bias) {}

DenseLayerImpl::~DenseLayerImpl() {
#ifdef USE_CUDNN
  for (auto &pair : fe_handle_cache) {
    cuda::cudnn_gemm::destroy_fe_handle(pair.second);
  }
  fe_handle_cache.clear();
#endif
}

void DenseLayerImpl::init_impl() {
  float stddev = static_cast<float>(1.0 / std::sqrt(static_cast<double>(input_features_)));

  if (this->use_seed_) {
    weights_.fill_random_normal(0, stddev, this->srand_seed_);
  } else {
    weights_.fill_random_normal(0, stddev);
  }

  if (use_bias_) {
    if (this->use_seed_) {
      bias_.fill_random_normal(0, stddev, this->srand_seed_);
    } else {
      bias_.fill_random_normal(0, stddev);
    }
  }

  weight_gradients_.fill(0.0f);
  if (use_bias_) {
    bias_gradients_.fill(0.0f);
  }
}

Tensor DenseLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  const Vec<size_t> &in_shape = input.shape();
  size_t last_dim = in_shape.back();

  if (last_dim != input_features_) {
    std::cerr << "Input last dimension: " << last_dim << " features, expected: " << input_features_
              << " features" << std::endl;
    throw std::invalid_argument("Input feature size mismatch in DenseLayerImpl");
  }

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

Tensor DenseLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  if (grad_output.shape().back() != output_features_) {
    throw std::invalid_argument("Gradient feature size mismatch in DenseLayerImpl. Expected " +
                                std::to_string(output_features_) + " features in grad_output" +
                                " but got " + std::to_string(grad_output.shape().back()) +
                                " features in grad_output" + ".");
  }

#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return cudnn_backward(grad_output, residuals);
  } else
#endif
  {
    return def_backward(grad_output, residuals);
  }
}

Tensor DenseLayerImpl::def_forward(const Tensor &input, Residuals &residuals) {
  Vec<size_t> input_shape = input.shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < input.shape().size() - 1; ++i) {
    batch_size *= input.shape()[i];
  }

  Tensor output = get_tensor({batch_size, output_features_}, input.dtype());
  if (get_engine_type() == EngineType::CPU) {
    DISPATCH_DTYPE(io_dtype_, T, {
      create_cpu_task(this->flow_handle_, cpu::legacy_dense::run_forward<T>, input.data_as<T>(),
                      weights_.data_as<T>(), output.data_as<T>(), batch_size, input_features_,
                      output_features_);
      if (use_bias_) {
        create_cpu_task(this->flow_handle_, cpu::legacy_dense::add_bias<T>, output.data_as<T>(),
                        bias_.data_as<T>(), batch_size, output_features_);
      }
    });
  } else {
    throw std::runtime_error("DenseLayerImpl only supports CPU device in def_forward");
  }
  return output;
}

Tensor DenseLayerImpl::def_backward(const Tensor &grad_output, Residuals &residuals) {
  if (grad_output.shape().back() != output_features_) {
    throw std::invalid_argument("Gradient feature size mismatch in DenseLayerImpl. Expected " +
                                std::to_string(output_features_) + " features in grad_output" +
                                " but got " + std::to_string(grad_output.shape().back()) +
                                " features in grad_output" + ".");
  }

  const Tensor &input = residuals["input"];

  Vec<size_t> input_shape = input.shape();

  size_t batch_size = 1;

  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
    batch_size *= input_shape[i];
  }

  Tensor grad_input = get_tensor(input_shape, grad_output.dtype());

  if (get_engine_type() == EngineType::CPU) {
    DISPATCH_DTYPE(io_dtype_, T, {
      create_cpu_task(this->flow_handle_, cpu::legacy_dense::run_wgrad<T>, input.data_as<T>(),
                      grad_output.data_as<T>(), weight_gradients_.data_as<T>(), batch_size,
                      input_features_, output_features_);
      create_cpu_task(this->flow_handle_, cpu::legacy_dense::run_dgrad<T>, grad_output.data_as<T>(),
                      weights_.data_as<T>(), grad_input.data_as<T>(), batch_size, input_features_,
                      output_features_);
      if (use_bias_) {
        create_cpu_task(this->flow_handle_, cpu::legacy_dense::run_bgrad<T>,
                        grad_output.data_as<T>(), bias_gradients_.data_as<T>(), batch_size,
                        output_features_);
      }
    });
  } else {
    throw std::runtime_error("DenseLayerImpl only supports CPU device in def_backward");
  }
  return grad_input;
}

#ifdef USE_CUDNN
void DenseLayerImpl::build_cudnn_graph(const Vec<size_t> &input_shape) const {
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
std::unique_ptr<Task> DenseLayerImpl::run_bgrad(const Tensor &grad_output, Tensor &bias_gradient,
                                                size_t batch_size, size_t output_features,
                                                flowHandle_t handle) const {
  if (grad_output.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("DenseLayerImpl grad_output dtype mismatch with dispatch IO_T");
  }
  if (bias_gradient.dtype() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "DenseLayerImpl bias grad_output dtype mismatch with dispatch Param_T");
  }
  if (get_engine_type() == EngineType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "DenseLayerImpl mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(handle, cpu::legacy_dense::run_bgrad<IO_T>, grad_output.data_as<IO_T>(),
                           bias_gradient.data_as<IO_T>(), batch_size, output_features);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(handle, cuda::legacy_dense::run_bgrad<IO_T, Param_T, Compute_T>,
                            grad_output.data_as<IO_T>(), bias_gradient.data_as<Param_T>(),
                            batch_size, output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for run_bgrad");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> DenseLayerImpl::add_bias(Tensor &output, const Tensor &bias,
                                               size_t batch_size, size_t output_features,
                                               flowHandle_t handle) const {
  if (output.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("DenseLayerImpl output dtype mismatch with dispatch IO_T");
  }
  if (bias.dtype() != dtype_of<Param_T>()) {
    throw std::runtime_error("DenseLayerImpl bias dtype mismatch with dispatch Param_T");
  }
  if (get_engine_type() == EngineType::CPU) {
    if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
      throw std::runtime_error(
          "DenseLayerImpl mixed dtype dispatch not implemented for CPU "
          "(io/param/compute must match).");
    }
    return create_cpu_task(handle, cpu::legacy_dense::add_bias<IO_T>, output.data_as<IO_T>(),
                           bias.data_as<IO_T>(), batch_size, output_features);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(handle, cuda::legacy_dense::add_bias<IO_T, Param_T, Compute_T>,
                            output.data_as<IO_T>(), bias.data_as<Param_T>(), batch_size,
                            output_features);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for add_bias");
  }
  return nullptr;
}

Tensor DenseLayerImpl::cudnn_forward(const Tensor &input, Residuals &residuals) {
  const Vec<size_t> &in_shape = input.shape();

  build_cudnn_graph(in_shape);

  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }
  size_t shape_key = get_shape_hash({batch_size});

  cuda::cudnn_gemm::feHandle_t *handle = fe_handle_cache[shape_key];
  GemmStats &stats = stats_cache[shape_key];

  Vec<size_t> out_shape = input.shape();
  out_shape.back() = output_features_;

  Tensor output = get_tensor(out_shape, io_dtype_);

  Tensor cudnn_workspace = this->get_tensor({stats.fwd_workspace_size}, DType_t::BYTE);

  create_cuda_task(this->flow_handle_, cuda::cudnn_gemm::run_forward, handle, stats,
                   input.data_as<void>(), weights_.data_as<void>(), output.data_as<void>(),
                   cudnn_workspace.data_as<void>());

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(add_bias, output, bias_, batch_size, output_features_,
                                   this->flow_handle_);
  }

  return output;
}

Tensor DenseLayerImpl::cudnn_backward(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];

  const Vec<size_t> &in_shape = input.shape();
  size_t batch_size = 1;
  for (size_t i = 0; i < in_shape.size() - 1; ++i) {
    batch_size *= in_shape[i];
  }

  size_t shape_key = get_shape_hash({batch_size});
  cuda::cudnn_gemm::feHandle_t *handle = fe_handle_cache.at(shape_key);

  GemmStats &stats = stats_cache.at(shape_key);

  Tensor grad_input = get_tensor(input.shape(), io_dtype_);

  Tensor cudnn_workspace = this->get_tensor(
      {std::max(stats.dgrad_workspace_size, stats.wgrad_workspace_size)}, DType_t::BYTE);

  // Compute weight gradients
  create_cuda_task(this->flow_handle_, cuda::cudnn_gemm::run_wgrad, handle, stats,
                   input.data_as<void>(), grad_output.data_as<void>(),
                   weight_gradients_.data_as<void>(), cudnn_workspace.data_as<void>());

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(run_bgrad, grad_output, bias_gradients_, batch_size,
                                   output_features_, this->flow_handle_);
  }

  // Compute input gradients
  create_cuda_task(this->flow_handle_, cuda::cudnn_gemm::run_dgrad, handle, stats,
                   grad_output.data_as<void>(), weights_.data_as<void>(),
                   grad_input.data_as<void>(), cudnn_workspace.data_as<void>());

  return grad_input;
}
#endif

LayerConfig DenseLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("input_features", input_features_);
  config.set("output_features", output_features_);
  config.set("use_bias", use_bias_);
  return config;
}

Vec<size_t> DenseLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::runtime_error("DenseLayerImpl::compute_output_shape: Input shape is empty.");
  }
  Vec<size_t> out_shape = input_shape;
  out_shape.back() = output_features_;
  return out_shape;
}

std::shared_ptr<DenseLayerImpl> DenseLayerImpl::create_from_config(const LayerConfig &config) {
  size_t input_features = config.get<size_t>("input_features");
  size_t output_features = config.get<size_t>("output_features");
  bool use_bias = config.get<bool>("use_bias");

  return std::make_shared<DenseLayerImpl>(input_features, output_features, use_bias, config.name);
}

}  // namespace synet
