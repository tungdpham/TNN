/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/conv2d_layer.hpp"

#include "device/task.hpp"
#include "nn/layers_impl/cpu/conv2d_nhwc_ops.hpp"
#include "ops/cuda/kernels.hpp"
#include "tensor/tensor.hpp"
#include "utils/misc.hpp"
#ifdef USE_DNNL
#include "nn/layers_impl/common/conv2d.hpp"
#include "nn/layers_impl/cpu/dnnl_conv2d_ops.hpp"
#endif
#ifdef USE_CUDNN
#include <type_traits>

#include "cuda/cudnn/common.hpp"
#include "device/cuda/cuda_context.hpp"
#include "nn/layers_impl/common/conv2d.hpp"
#include "nn/layers_impl/cuda/cudnn_conv2d_ops.hpp"
#endif
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "type/type.hpp"

namespace synet {

Conv2DLayerImpl::Conv2DLayerImpl(size_t in_channels, size_t out_channels, size_t kernel_h,
                                 size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                 size_t pad_w, bool use_bias, const std::string &name)
    : SISOLayerImpl(name),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_h_(stride_h),
      stride_w_(stride_w),
      pad_h_(pad_h),
      pad_w_(pad_w),
      use_bias_(use_bias) {}

Conv2DLayerImpl::~Conv2DLayerImpl() {
#ifdef USE_CUDNN
  for (auto &pair : fe_handle_cache) {
    if (pair.second) {
      cuda::cudnn_conv2d::destroy_fe_handle(pair.second);
    }
  }
  fe_handle_cache.clear();
  stats_cache.clear();
#endif
#ifdef USE_DNNL
  for (auto &pair : dnnl_handle_cache) {
    if (pair.second) {
      cpu::dnnl_conv2d::destroy_dnnl_handle(pair.second);
    }
  }
  dnnl_handle_cache.clear();
  dnnl_stats_cache.clear();
#endif
}

void Conv2DLayerImpl::init_impl() {
  float stddev = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(in_channels_ * kernel_h_ * kernel_w_)));

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

/**
 * @brief Perform convolution 2d forward on input and save it to output.
 * ! Only support GPU device with cuDNN backend. CPU implementation is to be added.
 * @tparam T
 * @param input input tensor in NHWC format
 * @param output output tensor in NHWC format
 * @param residuals micro batch id for caching input
 */

Tensor Conv2DLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (input.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NHWC)");
  }

  size_t channels = input.dimension(3);

  if (channels != in_channels_) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << in_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in Conv2DLayerImpl");
  }

  if (this->is_training_) {
    residuals["input"] = input;
  }

#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return cudnn_forward(input, residuals);
  }
#endif
#ifdef USE_DNNL
  if (get_engine_type() == EngineType::CPU) {
    return dnnl_forward(input, residuals);
  }
#endif
  return def_forward(input, residuals);
}

/**
 * @brief Perform convolution 2d backward on grad_output and save it to grad_input.
 * ! Only support GPU device with cuDNN backend. CPU implementation is to be added.
 * @tparam T
 * @param grad_output upstream grad_output tensor in NHWC format
 * @param grad_input output grad_output tensor in NHWC format
 */

Tensor Conv2DLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  if (grad_output.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NHWC)");
  }

  size_t channels = grad_output.dimension(3);

  if (channels != out_channels_) {
    std::cerr << "Gradient shape: " << channels << " channels, expected: " << out_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Gradient channel size mismatch in Conv2DLayerImpl");
  }

  const Tensor &input = residuals["input"];
  const auto &input_shape = input.shape();
  Tensor grad_input = get_tensor(input_shape, io_dtype_);

#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return cudnn_backward(grad_output, residuals);
  }
#endif
#ifdef USE_DNNL
  if (get_engine_type() == EngineType::CPU) {
    return dnnl_backward(grad_output, residuals);
  }
#endif

  return def_backward(grad_output, residuals);
}

Tensor Conv2DLayerImpl::def_forward(const Tensor &input, Residuals &residuals) {
  size_t batch_size = input.dimension(0);
  size_t input_h = input.dimension(1);
  size_t input_w = input.dimension(2);

  size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  Tensor output = get_tensor({batch_size, output_h, output_w, out_channels_}, input.dtype());

  if (get_engine_type() == EngineType::CPU) {
    DISPATCH_DTYPE(io_dtype_, T, {
      create_cpu_task(this->flow_handle_, cpu::conv2d_nhwc::run_forward<T>, input.data_as<T>(),
                      weights_.data_as<T>(), use_bias_ ? bias_.data_as<T>() : nullptr,
                      output.data_as<T>(), input.dimension(0), input.dimension(1),
                      input.dimension(2), in_channels_, out_channels_, kernel_h_, kernel_w_,
                      stride_h_, stride_w_, pad_h_, pad_w_, output.dimension(1),
                      output.dimension(2), use_bias_);
    });
  } else {
    throw std::runtime_error("Conv2DLayerImpl only supports CPU device in def_forward");
  }

  return output;
}

Tensor Conv2DLayerImpl::def_backward(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];

  Tensor grad_input = get_tensor(input.shape(), io_dtype_);

  size_t batch_size = grad_output.dimension(0);
  size_t input_h = input.dimension(1);
  size_t input_w = input.dimension(2);
  size_t output_h = grad_output.dimension(1);
  size_t output_w = grad_output.dimension(2);

  if (get_engine_type() == EngineType::CPU) {
    DISPATCH_DTYPE(io_dtype_, T, {
      create_cpu_task(this->flow_handle_, cpu::conv2d_nhwc::run_dgrad<T>, grad_output.data_as<T>(),
                      weights_.data_as<T>(), grad_input.data_as<T>(), grad_output.dimension(0),
                      input_h, input_w, in_channels_, out_channels_, kernel_h_, kernel_w_,
                      stride_h_, stride_w_, pad_h_, pad_w_, output_h, output_w);

      create_cpu_task(this->flow_handle_, cpu::conv2d_nhwc::run_wgrad<T>, input.data_as<T>(),
                      grad_output.data_as<T>(), weight_gradients_.data_as<T>(), batch_size, input_h,
                      input_w, in_channels_, out_channels_, kernel_h_, kernel_w_, stride_h_,
                      stride_w_, pad_h_, pad_w_, output_h, output_w);

      if (use_bias_) {
        create_cpu_task(this->flow_handle_, cpu::conv2d_nhwc::run_bgrad<T>,
                        grad_output.data_as<T>(), bias_gradients_.data_as<T>(), batch_size,
                        output_h, output_w, out_channels_);
      }
    });
  } else {
    throw std::runtime_error("Conv2DLayerImpl only supports CPU device in def_backward");
  }
  return grad_input;
}

#ifdef USE_CUDNN
template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> Conv2DLayerImpl::conv2d_forward_task(
    cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats, const Tensor &input,
    Tensor &output, const Tensor &weights, const Tensor &bias, Tensor &workspace, size_t batch_size,
    size_t input_h, size_t input_w, size_t output_h, size_t output_w, flowHandle_t handle) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("Conv2DLayerImpl IO_T and Param_T must be the same type");
  }
  if (input.dtype() != dtype_of<IO_T>() || output.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("Conv2DLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(handle, cuda::cudnn_conv2d::run_forward, fe_handle, stats,
                          input.data_as<void>(), weights.data_as<void>(),
                          bias_ ? bias_.data_as<void>() : nullptr, output.data_as<void>(),
                          workspace.data_as<void>());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> Conv2DLayerImpl::conv2d_backward_data_task(
    cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats, const Tensor &grad_output,
    const Tensor &weights, Tensor &grad_input, Tensor &workspace, size_t batch_size, size_t input_h,
    size_t input_w, size_t output_h, size_t output_w, flowHandle_t handle) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("Conv2DLayerImpl IO_T and Param_T must be the same type");
  }
  if (grad_output.dtype() != dtype_of<IO_T>() || grad_input.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("Conv2DLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(handle, cuda::cudnn_conv2d::run_dgrad, fe_handle, stats,
                          grad_output.data_as<void>(), weights.data_as<void>(),
                          grad_input.data_as<void>(), workspace.data_as<void>());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> Conv2DLayerImpl::conv2d_backward_weights_and_bias_task(
    cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats, const Tensor &input,
    const Tensor &grad_output, Tensor &weight_gradients, Tensor &bias_gradients, Tensor &workspace,
    size_t batch_size, size_t input_h, size_t input_w, size_t output_h, size_t output_w,
    flowHandle_t handle) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("Conv2DLayerImpl IO_T and Param_T must be the same type");
  }
  if (input.dtype() != dtype_of<IO_T>() || grad_output.dtype() != dtype_of<IO_T>()) {
    throw std::runtime_error("Conv2DLayerImpl input/grad_output dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(
      handle, cuda::cudnn_conv2d::run_wgrad_and_bgrad, fe_handle, stats, input.data_as<void>(),
      grad_output.data_as<void>(), weight_gradients.data_as<void>(),
      use_bias_ ? bias_gradients.data_as<void>() : nullptr, workspace.data_as<void>());
}

void Conv2DLayerImpl::build_graph(const Vec<size_t> &input_shape) const {
  // NHWC format: [N, H, W, C]
  size_t shape_key = get_shape_hash(input_shape);
  if (fe_handle_cache.find(shape_key) == fe_handle_cache.end()) {
    ConvolutionStats new_stats;
    size_t batch_size = input_shape[0];
    size_t input_h = input_shape[1];
    size_t input_w = input_shape[2];
    size_t in_channels = input_shape[3];
    init_convolution_stats(new_stats, batch_size, input_h, input_w, in_channels, out_channels_,
                           kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, use_bias_);
    cudnnHandle_t shared_handle = CUDAContext::getCudnnHandle();
    auto io_dtype = cuda::cudnn::to_cudnn_datatype(io_dtype_);
    auto compute_type = cuda::cudnn::to_cudnn_datatype(compute_dtype_);
    fe_handle_cache[shape_key] =
        cuda::cudnn_conv2d::initialize_fe_handle(shared_handle, io_dtype, compute_type, new_stats);
    stats_cache[shape_key] = new_stats;
  }
}

Tensor Conv2DLayerImpl::cudnn_forward(const Tensor &input, Residuals &residuals) {
  size_t batch_size = input.dimension(0);
  size_t input_h = input.dimension(1);
  size_t input_w = input.dimension(2);

  size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  Tensor output = get_tensor({batch_size, output_h, output_w, out_channels_}, input.dtype());

  size_t shape_key = get_shape_hash(input.shape());

  build_graph(input.shape());

  cuda::cudnn_conv2d::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  ConvolutionStats &current_stats = stats_cache.at(shape_key);

  size_t ws_bytes = current_stats.fwd_workspace_size;
  Tensor cudnn_workspace = this->get_tensor({ws_bytes}, DType_t::BYTE);

  DISPATCH_ON_3_DTYPES_TO_METHOD(conv2d_forward_task, fe_handle, current_stats, input, output,
                                 weights_, bias_, cudnn_workspace, batch_size, input_h, input_w,
                                 output_h, output_w, this->flow_handle_);
  return output;
}

Tensor Conv2DLayerImpl::cudnn_backward(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];
  if (!input) {
    throw std::runtime_error("No cached input found for Conv2DLayerImpl backward pass");
  }

  const auto &input_shape = input.shape();
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[1];
  size_t input_w = input_shape[2];
  const auto &grad_shape = grad_output.shape();
  size_t output_h = grad_shape[1];
  size_t output_w = grad_shape[2];

  Tensor grad_input = get_tensor(input_shape, io_dtype_);

  size_t shape_key = get_shape_hash(input_shape);

  cuda::cudnn_conv2d::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  ConvolutionStats &current_stats = stats_cache.at(shape_key);

  size_t max_workspace_size =
      std::max({current_stats.wgrad_workspace_size, current_stats.dgrad_workspace_size,
                current_stats.bgrad_workspace_size});

  Tensor cudnn_workspace = this->get_tensor({max_workspace_size}, DType_t::BYTE);
  Tensor temp_weight_gradients = this->get_tensor(weights_.shape(), io_dtype_);
  Tensor temp_bias_gradients = use_bias_ ? this->get_tensor(bias_.shape(), io_dtype_) : Tensor();

  DISPATCH_ON_3_DTYPES_TO_METHOD(conv2d_backward_weights_and_bias_task, fe_handle, current_stats,
                                 input, grad_output, temp_weight_gradients, temp_bias_gradients,
                                 cudnn_workspace, batch_size, input_h, input_w, output_h, output_w,
                                 this->flow_handle_);

  DISPATCH_DTYPE(param_dtype_, T, {
    create_cuda_task(this->flow_handle_, ops::cuda::cuda_axpy<T>, 1.0f,
                     temp_weight_gradients.data_as<T>(), weight_gradients_.data_as<T>(),
                     weights_.size());
    if (use_bias_) {
      create_cuda_task(this->flow_handle_, ops::cuda::cuda_axpy<T>, 1.0f,
                       temp_bias_gradients.data_as<T>(), bias_gradients_.data_as<T>(),
                       bias_.size());
    }
  });

  DISPATCH_ON_3_DTYPES_TO_METHOD(conv2d_backward_data_task, fe_handle, current_stats, grad_output,
                                 weights_, grad_input, cudnn_workspace, batch_size, input_h,
                                 input_w, output_h, output_w, this->flow_handle_);
  return grad_input;
}
#endif

#ifdef USE_DNNL
void Conv2DLayerImpl::build_dnnl_handle(const Vec<size_t> &input_shape) const {
  size_t shape_key = get_shape_hash(input_shape);
  if (dnnl_handle_cache.find(shape_key) == dnnl_handle_cache.end()) {
    ConvolutionStats new_stats;
    init_convolution_stats(new_stats, input_shape[0], input_shape[1], input_shape[2],
                           input_shape[3], out_channels_, kernel_h_, kernel_w_, stride_h_,
                           stride_w_, pad_h_, pad_w_, use_bias_);
    dnnl_handle_cache[shape_key] = cpu::dnnl_conv2d::initialize_dnnl_handle(new_stats, io_dtype_);
    dnnl_stats_cache[shape_key] = new_stats;
  }
}

Tensor Conv2DLayerImpl::dnnl_forward(const Tensor &input, Residuals & /*residuals*/) {
  size_t batch_size = input.dimension(0);
  size_t input_h = input.dimension(1);
  size_t input_w = input.dimension(2);
  size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  Tensor output = get_tensor({batch_size, output_h, output_w, out_channels_}, input.dtype());

  build_dnnl_handle(input.shape());
  size_t shape_key = get_shape_hash(input.shape());
  cpu::dnnl_conv2d::dnnlHandle_t *dnnl_handle = dnnl_handle_cache.at(shape_key);
  const ConvolutionStats &current_stats = dnnl_stats_cache.at(shape_key);

  Tensor workspace = get_tensor({current_stats.fwd_workspace_size}, DType_t::BYTE);

  create_cpu_task(this->flow_handle_, cpu::dnnl_conv2d::run_forward, dnnl_handle, current_stats,
                  input.data_as<void>(), weights_.data_as<void>(),
                  use_bias_ ? bias_.data_as<void>() : nullptr, output.data_as<void>(),
                  current_stats.fwd_workspace_size > 0 ? workspace.data_as<void>() : nullptr);

  return output;
}

Tensor Conv2DLayerImpl::dnnl_backward(const Tensor &grad_output, Residuals &residuals) {
  const Tensor &input = residuals["input"];
  if (!input) {
    throw std::runtime_error("dnnl_backward: no cached input for Conv2DLayerImpl backward pass");
  }

  const auto &input_shape = input.shape();
  Tensor grad_input = get_tensor(input_shape, io_dtype_);

  build_dnnl_handle(input_shape);
  size_t shape_key = get_shape_hash(input_shape);
  cpu::dnnl_conv2d::dnnlHandle_t *dnnl_handle = dnnl_handle_cache.at(shape_key);
  const ConvolutionStats &current_stats = dnnl_stats_cache.at(shape_key);

  size_t max_ws = std::max(current_stats.wgrad_workspace_size, current_stats.dgrad_workspace_size);
  Tensor workspace = get_tensor({max_ws}, DType_t::BYTE);

  create_cpu_task(this->flow_handle_, cpu::dnnl_conv2d::run_dgrad, dnnl_handle, current_stats,
                  grad_output.data_as<void>(), weights_.data_as<void>(), grad_input.data_as<void>(),
                  workspace.data_as<void>());

  create_cpu_task(this->flow_handle_, cpu::dnnl_conv2d::run_wgrad_and_bgrad, dnnl_handle,
                  current_stats, input.data_as<void>(), grad_output.data_as<void>(),
                  weight_gradients_.data_as<void>(),
                  use_bias_ ? bias_gradients_.data_as<void>() : nullptr, workspace.data_as<void>());

  return grad_input;
}
#endif  // USE_DNNL

LayerConfig Conv2DLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("in_channels", in_channels_);
  config.set("out_channels", out_channels_);
  config.set("kernel_h", kernel_h_);
  config.set("kernel_w", kernel_w_);
  config.set("stride_h", stride_h_);
  config.set("stride_w", stride_w_);
  config.set("pad_h", pad_h_);
  config.set("pad_w", pad_w_);
  config.set("use_bias", use_bias_);
  return config;
}

Vec<size_t> Conv2DLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("Conv2DLayerImpl expects 4D input including batch size");
  }

  size_t batch_size = input_shape[0];
  size_t output_h = (input_shape[1] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[2] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {batch_size, output_h, output_w, out_channels_};
}

std::shared_ptr<Conv2DLayerImpl> Conv2DLayerImpl::create_from_config(const LayerConfig &config) {
  size_t in_channels = config.get<size_t>("in_channels");
  size_t out_channels = config.get<size_t>("out_channels");
  size_t kernel_h = config.get<size_t>("kernel_h");
  size_t kernel_w = config.get<size_t>("kernel_w");
  size_t stride_h = config.get<size_t>("stride_h", 1);
  size_t stride_w = config.get<size_t>("stride_w", 1);
  size_t pad_h = config.get<size_t>("pad_h", 0);
  size_t pad_w = config.get<size_t>("pad_w", 0);
  bool use_bias = config.get<bool>("use_bias", true);
  return std::make_shared<Conv2DLayerImpl>(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                           stride_w, pad_h, pad_w, use_bias, config.name);
}

}  // namespace synet
