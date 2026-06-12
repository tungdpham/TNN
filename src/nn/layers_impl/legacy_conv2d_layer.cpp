/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/legacy_conv2d_layer.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "device/device_type.hpp"
#include "device/task.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/common/conv2d.hpp"
#include "nn/layers_impl/cpu/conv2d_nchw_ops.hpp"
#include "nn/layers_impl/cuda/cudnn_conv2d_nchw_ops.hpp"
#include "utils/misc.hpp"
#ifdef USE_CUDA
#include "device/cuda/cuda_context.hpp"
#include "nn/layers_impl/cuda/conv2d_nchw_ops.hpp"
#endif
#include "tensor/tensor_ops.hpp"
#include "type/type.hpp"

namespace synet {

LegacyConv2DLayerImpl::LegacyConv2DLayerImpl(size_t in_channels, size_t out_channels,
                                             size_t kernel_h, size_t kernel_w, size_t stride_h,
                                             size_t stride_w, size_t pad_h, size_t pad_w,
                                             bool use_bias, const std::string &name)
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

LegacyConv2DLayerImpl::~LegacyConv2DLayerImpl() {
#ifdef USE_CUDNN
  for (auto &pair : convolution_handle_cache) {
    if (pair.second) {
      cuda::cudnn_conv2d::destroy_convolution_handle(pair.second);
    }
  }
  convolution_handle_cache.clear();
  stats_cache.clear();
#endif
}

void LegacyConv2DLayerImpl::init_impl() {
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
 * ! Support both CPU and GPU devices but only NCHW data format.
 * @tparam T I/O data type
 * @param input input tensor in NCHW format
 * @param output input tensor in NCHW format
 * @param mb_id micro batch id for caching input
 */

Tensor LegacyConv2DLayerImpl::forward_impl(const Tensor &input, size_t mb_id) {
  if (input.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }

  size_t channels = input.dimension(1);

  if (channels != in_channels_) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << in_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in LegacyConv2DLayerImpl");
  }

#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return cudnn_forward(input, mb_id);
  }
#endif

  return def_forward(input, mb_id);
}

Tensor LegacyConv2DLayerImpl::backward_impl(const Tensor &grad_output, size_t mb_id) {
  if (grad_output.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }

  size_t channels = grad_output.dimension(1);

  if (channels != out_channels_) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << out_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Gradient channel size mismatch in LegacyConv2DLayerImpl");
  }

#ifdef USE_CUDNN
  if (get_engine_type() == EngineType::CUDA) {
    return cudnn_backward(grad_output, mb_id);
  }
#endif

  return def_backward(grad_output, mb_id);
}

Tensor LegacyConv2DLayerImpl::def_forward(const Tensor &input, size_t mb_id) {
  if (input.dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }
  const size_t batch_size = input.dimension(0);
  const size_t input_h = input.dimension(2);
  const size_t input_w = input.dimension(3);

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  micro_batch_input_shapes_[mb_id] = input.shape();
  Tensor output = get_tensor({batch_size, out_channels_, output_h, output_w}, input.data_type());

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;
  size_t col_matrix_size = kernel_size * output_size;

  Tensor col_buffer = get_tensor({col_matrix_size}, io_dtype_);
  set_mutable_cache(mb_id, "col_buffer", col_buffer);

  size_t output_buffer_size = out_channels_ * output_size;
  Tensor temp_output_buffer = get_tensor({output_buffer_size}, io_dtype_);

  ops::im2col(input, col_buffer, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
              this->flow_handle_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_forward, col_buffer, weights_, temp_output_buffer, output_size,
                                 kernel_size, out_channels_, this->flow_handle_);

  ops::cnhw_to_nchw(temp_output_buffer, output, batch_size, out_channels_, output_h, output_w,
                    this->flow_handle_);

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(add_bias, output, bias_, batch_size, output_h, output_w,
                                   out_channels_, this->flow_handle_);
  }

  return output;
}

Tensor LegacyConv2DLayerImpl::def_backward(const Tensor &grad_output, size_t mb_id) {
  auto it_input_shape = micro_batch_input_shapes_.find(mb_id);

  if (it_input_shape == micro_batch_input_shapes_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(mb_id));
  }

  const auto &input_shape = it_input_shape->second;
  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[2];
  const size_t input_w = input_shape[3];
  const size_t output_h = grad_output.dimension(2);
  const size_t output_w = grad_output.dimension(3);

  Tensor grad_input = get_tensor(input_shape, io_dtype_);
  grad_input.fill(0);  // col2im accumulates, so we need to zero first

  Tensor col_buffer = this->get_mutable_cache(mb_id, "col_buffer");

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;
  size_t col_grad_matrix_size = kernel_size * output_size;

  size_t gradient_buffer_size = out_channels_ * output_size;
  Tensor temp_gradient_buffer = get_tensor({gradient_buffer_size}, io_dtype_);
  Tensor temp_col_grad_matrix_buffer = get_tensor({col_grad_matrix_size}, io_dtype_);

  ops::nchw_to_cnhw(grad_output, temp_gradient_buffer, batch_size, out_channels_, output_h,
                    output_w, this->flow_handle_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_wgrad, col_buffer, temp_gradient_buffer, weight_gradients_,
                                 output_size, kernel_size, out_channels_, this->flow_handle_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_dgrad, temp_gradient_buffer, weights_,
                                 temp_col_grad_matrix_buffer, output_size, kernel_size,
                                 out_channels_, this->flow_handle_);

  ops::col2im(temp_col_grad_matrix_buffer, grad_input, batch_size, in_channels_, input_h, input_w,
              kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, this->flow_handle_);

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(run_bgrad, grad_output, bias_gradients_, batch_size, output_h,
                                   output_w, out_channels_, this->flow_handle_);
  }

  return grad_input;
}

#ifdef USE_CUDNN
void LegacyConv2DLayerImpl::build_graph(const Vec<size_t> &input_shape) const {
  size_t shape_key = get_shape_hash(input_shape);
  if (stats_cache.find(shape_key) != stats_cache.end()) {
    return;  // Graph already built for this input shape
  }

  if (input_shape.size() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }

  size_t batch_size = input_shape[0];
  size_t in_channels_ = input_shape[1];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];

  if (convolution_handle_cache.find(shape_key) == convolution_handle_cache.end()) {
    ConvolutionStats new_stats;
    init_convolution_stats(new_stats, batch_size, input_h, input_w, in_channels_, out_channels_,
                           kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, use_bias_);
    cudnnHandle_t shared_handle = CUDAContext::getCudnnHandle();

    convolution_handle_cache[shape_key] =
        cuda::cudnn_conv2d::initialize_convolution_handle(shared_handle, new_stats, 0);
    stats_cache[shape_key] = new_stats;
  }
}

Tensor LegacyConv2DLayerImpl::cudnn_forward(const Tensor &input, size_t mb_id) {
  const auto &shape = input.shape();
  const size_t batch_size = shape[0];
  const size_t input_h = shape[2];
  const size_t input_w = shape[3];

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  Tensor output = get_tensor({batch_size, out_channels_, output_h, output_w}, this->io_dtype_);

  build_graph(input.shape());

  size_t shape_key = get_shape_hash(input.shape());
  const ConvolutionStats &stats = stats_cache.at(shape_key);

  Tensor cudnn_workspace = this->get_tensor({stats.fwd_workspace_size}, DType_t::BYTE);

  if (this->is_training_) {
    this->set_immutable_cache(mb_id, "input", input);
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_run_forward, input, weights_, bias_, output, batch_size,
                                 input_h, input_w, output_h, output_w, cudnn_workspace,
                                 this->flow_handle_);

  return output;
}

Tensor LegacyConv2DLayerImpl::cudnn_backward(const Tensor &grad_output, size_t mb_id) {
  const Tensor &input = this->get_immutable_cache(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for micro-batch ID: " + std::to_string(mb_id));
  }

  const auto &input_shape = input.shape();

  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[2];
  const size_t input_w = input_shape[3];
  const auto &grad_shape = grad_output.shape();
  const size_t output_h = grad_shape[2];
  const size_t output_w = grad_shape[3];

  Tensor grad_input = get_tensor(input_shape, io_dtype_);

  size_t shape_key = get_shape_hash(input_shape);
  const ConvolutionStats &stats = stats_cache.at(shape_key);
  size_t bwd_workspace = std::max(
      {stats.wgrad_workspace_size, stats.dgrad_workspace_size, stats.bgrad_workspace_size});

  Tensor cudnn_workspace = this->get_tensor({bwd_workspace}, DType_t::BYTE);

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_run_wgrad, input, grad_output, weight_gradients_, batch_size,
                                 input_h, input_w, output_h, output_w, cudnn_workspace,
                                 this->flow_handle_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_run_dgrad, grad_output, weights_, grad_input, batch_size,
                                 input_h, input_w, output_h, output_w, cudnn_workspace,
                                 this->flow_handle_);

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_run_bgrad, grad_output, bias_gradients_, batch_size,
                                   output_h, output_w, out_channels_, this->flow_handle_);
  }

  return grad_input;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayerImpl::cudnn_run_forward(
    const Tensor &input, const Tensor &weight, const Tensor bias, Tensor &output, size_t batch_size,
    size_t input_h, size_t input_w, size_t output_h, size_t output_w, Tensor &cudnn_workspace,
    flowHandle_t handle) {
  if (input.device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN forward requires GPU device");
  }

  size_t shape_key = get_shape_hash(input.shape());

  cuda::cudnn_conv2d::ConvolutionHandle *convolution_handle =
      convolution_handle_cache.at(shape_key);

  return create_cuda_task(handle, cuda::cudnn_conv2d::forward_with_bias<IO_T>, convolution_handle,
                          input.data(), weight.data(), bias ? bias.data() : nullptr, output.data(),
                          batch_size, in_channels_, input_h, input_w, out_channels_, output_h,
                          output_w, cudnn_workspace.data(), cudnn_workspace.capacity());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayerImpl::cudnn_run_dgrad(
    const Tensor &grad_output, const Tensor &weight, Tensor &input_grad, size_t batch_size,
    size_t input_h, size_t input_w, size_t output_h, size_t output_w, Tensor &cudnn_workspace,
    flowHandle_t handle) {
  if (grad_output.device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN backward data requires GPU device");
  }

  size_t shape_key = get_shape_hash(input_grad.shape());

  cuda::cudnn_conv2d::ConvolutionHandle *convolution_handle =
      convolution_handle_cache.at(shape_key);

  return create_cuda_task(handle, cuda::cudnn_conv2d::run_dgrad<IO_T>, convolution_handle,
                          grad_output.data(), weight.data(), input_grad.data(), batch_size,
                          in_channels_, input_h, input_w, out_channels_, output_h, output_w,
                          cudnn_workspace.data(), cudnn_workspace.capacity());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayerImpl::cudnn_run_wgrad(
    const Tensor &input, const Tensor &grad_output, Tensor &weight_grad, size_t batch_size,
    size_t input_h, size_t input_w, size_t output_h, size_t output_w, Tensor &cudnn_workspace,
    flowHandle_t handle) {
  if (grad_output.device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN backward filter requires GPU device");
  }

  size_t shape_key = get_shape_hash(input.shape());

  cuda::cudnn_conv2d::ConvolutionHandle *convolution_handle =
      convolution_handle_cache.at(shape_key);

  return create_cuda_task(handle, cuda::cudnn_conv2d::backward_filter<IO_T>, convolution_handle,
                          input.data(), grad_output.data(), weight_grad.data(), batch_size,
                          in_channels_, input_h, input_w, out_channels_, output_h, output_w,
                          cudnn_workspace.data(), cudnn_workspace.capacity());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayerImpl::cudnn_run_bgrad(const Tensor &grad_output,
                                                             Tensor &bias_grad, size_t batch_size,
                                                             size_t output_h, size_t output_w,
                                                             size_t out_channels,
                                                             flowHandle_t handle) {
  if (grad_output.device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN backward bias requires GPU device");
  }

  // compute input shape
  size_t input_h = (output_h - 1) * stride_h_ + kernel_h_ - 2 * pad_h_;
  size_t input_w = (output_w - 1) * stride_w_ + kernel_w_ - 2 * pad_w_;
  size_t shape_key = get_shape_hash({batch_size, in_channels_, input_h, input_w});

  cuda::cudnn_conv2d::ConvolutionHandle *convolution_handle =
      convolution_handle_cache.at(shape_key);

  return create_cuda_task(handle, cuda::cudnn_conv2d::run_bgrad<IO_T>, convolution_handle,
                          grad_output.data(), bias_grad.data(), batch_size, out_channels, output_h,
                          output_w);
}
#endif

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayerImpl::run_forward(
    const Tensor &col_data, const Tensor &weight_data, Tensor &output_data,
    const size_t output_size, const size_t kernel_size, const size_t out_channels,
    flowHandle_t handle) {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyConv2DLayerImpl mixed dtype dispatch not implemented (io/param/compute must "
        "match).");
  }
  if (col_data.data_type() != dtype_of<IO_T>() || output_data.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyConv2DLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight_data.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyConv2DLayerImpl weight tensor dtype mismatch with dispatch Param_T");
  }
  if (col_data.device_type() != weight_data.device_type() ||
      weight_data.device_type() != output_data.device_type()) {
    throw std::runtime_error("All tensors must be on the same device for conv forward");
  }

  if (col_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::conv2d_nchw::run_forward<Compute_T>,
                           col_data.data_as<Compute_T>(), weight_data.data_as<Compute_T>(),
                           output_data.data_as<Compute_T>(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (col_data.device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::conv2d_nchw::run_forward<Compute_T>,
                            col_data.data_as<Compute_T>(), weight_data.data_as<Compute_T>(),
                            output_data.data_as<Compute_T>(), output_size, kernel_size,
                            out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv forward");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayerImpl::run_wgrad(
    const Tensor &col_data, const Tensor &gradient_data, Tensor &weight_grad_data,
    const size_t output_size, const size_t kernel_size, const size_t out_channels,
    flowHandle_t handle) {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyConv2DLayerImpl mixed dtype dispatch not implemented (io/param/compute must "
        "match).");
  }
  if (col_data.data_type() != dtype_of<IO_T>() || gradient_data.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyConv2DLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight_grad_data.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyConv2DLayerImpl weight grad_output dtype mismatch with dispatch Param_T");
  }
  if (col_data.device_type() != gradient_data.device_type() ||
      gradient_data.device_type() != weight_grad_data.device_type()) {
    throw std::runtime_error("All tensors must be on the same device for conv weight gradients");
  }

  if (col_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::conv2d_nchw::run_wgrad<Compute_T>,
                           col_data.data_as<Compute_T>(), gradient_data.data_as<Compute_T>(),
                           weight_grad_data.data_as<Compute_T>(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (col_data.device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::conv2d_nchw::run_wgrad<Compute_T>,
                            col_data.data_as<Compute_T>(), gradient_data.data_as<Compute_T>(),
                            weight_grad_data.data_as<Compute_T>(), output_size, kernel_size,
                            out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv weight gradients");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayerImpl::run_dgrad(
    const Tensor &gradient_data, const Tensor &weight_data, Tensor &col_grad_data,
    const size_t output_size, const size_t kernel_size, const size_t out_channels,
    flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyConv2DLayerImpl mixed dtype dispatch not implemented (io/param/compute must "
        "match).");
  }
  if (gradient_data.data_type() != dtype_of<IO_T>() ||
      col_grad_data.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyConv2DLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight_data.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyConv2DLayerImpl weight tensor dtype mismatch with dispatch Param_T");
  }
  if (gradient_data.device_type() != weight_data.device_type() ||
      weight_data.device_type() != col_grad_data.device_type()) {
    throw std::runtime_error("All tensors must be on the same device for conv input gradients");
  }

  if (gradient_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::conv2d_nchw::run_dgrad<Compute_T>,
                           gradient_data.data_as<Compute_T>(), weight_data.data_as<Compute_T>(),
                           col_grad_data.data_as<Compute_T>(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (gradient_data.device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::conv2d_nchw::run_dgrad<Compute_T>,
                            gradient_data.data_as<Compute_T>(), weight_data.data_as<Compute_T>(),
                            col_grad_data.data_as<Compute_T>(), output_size, kernel_size,
                            out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv input gradients");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayerImpl::run_bgrad(
    const Tensor &gradient_data, Tensor &bias_grad_data, const size_t batch_size,
    const size_t output_h, const size_t output_w, const size_t out_channels, flowHandle_t handle) {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyConv2DLayerImpl mixed dtype dispatch not implemented (io/param/compute must "
        "match).");
  }
  if (gradient_data.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyConv2DLayerImpl grad_output dtype mismatch with dispatch IO_T");
  }
  if (bias_grad_data.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyConv2DLayerImpl bias grad_output dtype mismatch with dispatch Param_T");
  }
  if (gradient_data.device_type() != bias_grad_data.device_type()) {
    throw std::runtime_error("Gradient and bias grad_output tensors must be on the same device");
  }

  if (gradient_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::conv2d_nchw::run_bgrad<Compute_T>,
                           gradient_data.data_as<Compute_T>(), bias_grad_data.data_as<Compute_T>(),
                           batch_size, output_h, output_w, out_channels);
  }
#ifdef USE_CUDA
  else if (gradient_data.device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::conv2d_nchw::run_bgrad<Compute_T>,
                            gradient_data.data_as<Compute_T>(), bias_grad_data.data_as<Compute_T>(),
                            batch_size, output_h, output_w, out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv bias gradients");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayerImpl::add_bias(Tensor &output_data, const Tensor &bias_data,
                                                      const size_t batch_size,
                                                      const size_t output_h, const size_t output_w,
                                                      const size_t out_channels,
                                                      flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyConv2DLayerImpl mixed dtype dispatch not implemented (io/param/compute must "
        "match).");
  }
  if (output_data.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyConv2DLayerImpl output dtype mismatch with dispatch IO_T");
  }
  if (bias_data.data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LegacyConv2DLayerImpl bias dtype mismatch with dispatch Param_T");
  }
  if (output_data.device_type() != bias_data.device_type()) {
    throw std::runtime_error("Output and bias tensors must be on the same device");
  }

  if (output_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::conv2d_nchw::add_bias<Compute_T>,
                           output_data.data_as<Compute_T>(), bias_data.data_as<Compute_T>(),
                           batch_size, output_h, output_w, out_channels);
  }
#ifdef USE_CUDA
  else if (output_data.device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::conv2d_nchw::add_bias<Compute_T>,
                            output_data.data_as<Compute_T>(), bias_data.data_as<Compute_T>(),
                            batch_size, output_h, output_w, out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv add bias to output");
  }
  return nullptr;
}

LayerConfig LegacyConv2DLayerImpl::get_config() const {
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

Vec<size_t> LegacyConv2DLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("LegacyConv2DLayerImpl expects 4D input including batch size");
  }

  size_t batch_size = input_shape[0];

  size_t output_h = (input_shape[2] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[3] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {batch_size, out_channels_, output_h, output_w};
}

std::shared_ptr<LegacyConv2DLayerImpl> LegacyConv2DLayerImpl::create_from_config(
    const LayerConfig &config) {
  size_t in_channels = config.get<size_t>("in_channels");
  size_t out_channels = config.get<size_t>("out_channels");
  size_t kernel_h = config.get<size_t>("kernel_h");
  size_t kernel_w = config.get<size_t>("kernel_w");
  size_t stride_h = config.get<size_t>("stride_h", 1);
  size_t stride_w = config.get<size_t>("stride_w", 1);
  size_t pad_h = config.get<size_t>("pad_h", 0);
  size_t pad_w = config.get<size_t>("pad_w", 0);
  bool use_bias = config.get<bool>("use_bias", true);
  return std::make_shared<LegacyConv2DLayerImpl>(in_channels, out_channels, kernel_h, kernel_w,
                                                 stride_h, stride_w, pad_h, pad_w, use_bias,
                                                 config.name);
}

}  // namespace synet
