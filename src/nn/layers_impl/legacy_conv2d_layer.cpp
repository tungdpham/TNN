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

namespace tnn {

LegacyConv2DLayer::LegacyConv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h,
                                     size_t kernel_w, size_t stride_h, size_t stride_w,
                                     size_t pad_h, size_t pad_w, bool use_bias,
                                     const std::string &name)
    : ParameterizedLayer(name),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_h_(stride_h),
      stride_w_(stride_w),
      pad_h_(pad_h),
      pad_w_(pad_w),
      use_bias_(use_bias) {}

LegacyConv2DLayer::~LegacyConv2DLayer() {
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

void LegacyConv2DLayer::init_impl() {
  float bound = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(in_channels_ * kernel_h_ * kernel_w_)));

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

/**
 * @brief Perform convolution 2d forward on input and save it to output->
 * ! Support both CPU and GPU devices but only NCHW data format.
 * @tparam T I/O data type
 * @param input input tensor in NCHW format
 * @param output input tensor in NCHW format
 * @param mb_id micro batch id for caching input
 */

void LegacyConv2DLayer::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  if (input->dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }

  size_t channels = input->dimension(1);

  if (channels != in_channels_) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << in_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in LegacyConv2DLayer");
  }

#ifdef USE_CUDNN
  if (this->device().device_type() == DeviceType::GPU) {
    cudnn_forward(input, output, mb_id);
    return;
  }
#endif

  def_forward(input, output, mb_id);
}

void LegacyConv2DLayer::backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                                      size_t mb_id) {
  if (grad_output->dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }

  size_t channels = grad_output->dimension(1);

  if (channels != out_channels_) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << out_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Gradient channel size mismatch in LegacyConv2DLayer");
  }

#ifdef USE_CUDNN
  if (this->device().device_type() == DeviceType::GPU) {
    cudnn_backward(grad_output, grad_input, mb_id);
    return;
  }
#endif

  def_backward(grad_output, grad_input, mb_id);
}

void LegacyConv2DLayer::def_forward(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  if (input->dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NCHW)");
  }
  const size_t batch_size = input->dimension(0);
  const size_t input_h = input->dimension(2);
  const size_t input_w = input->dimension(3);

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  micro_batch_input_shapes_[mb_id] = input->shape();
  output->ensure({batch_size, out_channels_, output_h, output_w});

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;
  size_t col_matrix_size = kernel_size * output_size;

  // Ensure per-microbatch col buffer is allocated
  Tensor &col_buffer = micro_batch_col_buffers_[mb_id];
  if (col_buffer == nullptr) {
    col_buffer = make_io_tensor({col_matrix_size});
  } else {
    col_buffer->ensure({col_matrix_size});
  }

  size_t output_buffer_size = out_channels_ * output_size;
  Tensor temp_output_buffer = make_io_tensor({output_buffer_size});

  DISPATCH_IO_DTYPE(ops::im2col, input, col_buffer, kernel_h_, kernel_w_, stride_h_, stride_w_,
                    pad_h_, pad_w_, this->flow_handle_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_conv_forward_impl, col_buffer, weights_,
                                 temp_output_buffer, output_size, kernel_size, out_channels_,
                                 this->flow_handle_);

  DISPATCH_IO_DTYPE(ops::cnhw_to_nchw, temp_output_buffer, output, batch_size, out_channels_,
                    output_h, output_w, this->flow_handle_);

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(add_bias_to_output_impl, output, bias_, batch_size, output_h,
                                   output_w, out_channels_, this->flow_handle_);
  }
}

void LegacyConv2DLayer::def_backward(const ConstTensor &grad_output, const Tensor &grad_input,
                                     size_t mb_id) {
  auto it_input_shape = micro_batch_input_shapes_.find(mb_id);

  if (it_input_shape == micro_batch_input_shapes_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(mb_id));
  }

  const auto &input_shape = it_input_shape->second;
  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[2];
  const size_t input_w = input_shape[3];
  const size_t output_h = grad_output->dimension(2);
  const size_t output_w = grad_output->dimension(3);

  grad_input->ensure(input_shape);
  grad_input->fill(0);  // col2im accumulates, so we need to zero first

  auto it_col_buffer = micro_batch_col_buffers_.find(mb_id);
  if (it_col_buffer == micro_batch_col_buffers_.end()) {
    throw std::runtime_error("No cached col buffer found for micro-batch ID: " +
                             std::to_string(mb_id));
  }

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;
  size_t col_grad_matrix_size = kernel_size * output_size;

  size_t gradient_buffer_size = out_channels_ * output_size;
  Tensor temp_gradient_buffer = make_io_tensor({gradient_buffer_size});
  Tensor temp_col_grad_matrix_buffer = make_io_tensor({col_grad_matrix_size});

  DISPATCH_IO_DTYPE(ops::nchw_to_cnhw, grad_output, temp_gradient_buffer, batch_size, out_channels_,
                    output_h, output_w, this->flow_handle_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_weight_gradients_impl, it_col_buffer->second,
                                 temp_gradient_buffer, weight_gradients_, output_size, kernel_size,
                                 out_channels_, this->flow_handle_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_input_gradients_impl, temp_gradient_buffer, weights_,
                                 temp_col_grad_matrix_buffer, output_size, kernel_size,
                                 out_channels_, this->flow_handle_);

  DISPATCH_IO_DTYPE(ops::col2im, temp_col_grad_matrix_buffer, grad_input, batch_size, in_channels_,
                    input_h, input_w, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                    this->flow_handle_);

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(compute_bias_gradients_impl, grad_output, bias_gradients_,
                                   batch_size, output_h, output_w, out_channels_,
                                   this->flow_handle_);
  }
}

#ifdef USE_CUDNN
void LegacyConv2DLayer::build_graph(const Vec<size_t> &input_shape) const {
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
  }
}

void LegacyConv2DLayer::cudnn_forward(const ConstTensor &input, const Tensor &output,
                                      size_t mb_id) {
  const auto &shape = input->shape();
  const size_t batch_size = shape[0];
  const size_t input_h = shape[2];
  const size_t input_w = shape[3];

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  output->ensure({batch_size, out_channels_, output_h, output_w});

  build_graph(input->shape());

  size_t shape_key = get_shape_hash(input->shape());
  const ConvolutionStats &stats = stats_cache.at(shape_key);

  Tensor cudnn_workspace = this->get_workspace({stats.fwd_workspace_size}, DType_t::BYTE);

  if (this->is_training_) {
    ConstTensor &cached_input = this->get_immutable_cache(mb_id, "input");
    cached_input = input;
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_compute_fwd, input, weights_, bias_, output, batch_size,
                                 input_h, input_w, output_h, output_w, cudnn_workspace,
                                 this->flow_handle_);
}

void LegacyConv2DLayer::cudnn_backward(const ConstTensor &grad_output, const Tensor &grad_input,
                                       size_t mb_id) {
  ConstTensor &input = this->get_immutable_cache(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for micro-batch ID: " + std::to_string(mb_id));
  }

  const auto &input_shape = input->shape();

  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[2];
  const size_t input_w = input_shape[3];
  const auto &grad_shape = grad_output->shape();
  const size_t output_h = grad_shape[2];
  const size_t output_w = grad_shape[3];

  grad_input->ensure(input_shape);

  size_t shape_key = get_shape_hash(input_shape);
  const ConvolutionStats &stats = stats_cache.at(shape_key);
  size_t bwd_workspace = std::max(
      {stats.wgrad_workspace_size, stats.dgrad_workspace_size, stats.bgrad_workspace_size});

  Tensor cudnn_workspace = this->get_workspace({bwd_workspace}, DType_t::BYTE);

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_backward_filter, input, grad_output, weight_gradients_,
                                 batch_size, input_h, input_w, output_h, output_w, cudnn_workspace,
                                 this->flow_handle_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_backward_data, grad_output, weights_, grad_input, batch_size,
                                 input_h, input_w, output_h, output_w, cudnn_workspace,
                                 this->flow_handle_);

  if (use_bias_) {
    DISPATCH_ON_3_DTYPES_TO_METHOD(cudnn_backward_bias, grad_output, bias_gradients_, batch_size,
                                   output_h, output_w, out_channels_, this->flow_handle_);
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayer::cudnn_compute_fwd(
    const ConstTensor &input, const ConstTensor &weight, const ConstTensor bias,
    const Tensor &output, size_t batch_size, size_t input_h, size_t input_w, size_t output_h,
    size_t output_w, const Tensor &cudnn_workspace, flowHandle_t handle) {
  if (input->device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN forward requires GPU device");
  }

  size_t shape_key = get_shape_hash(input->shape());

  cuda::cudnn_conv2d::ConvolutionHandle *convolution_handle =
      convolution_handle_cache.at(shape_key);

  return create_cuda_task(handle, cuda::cudnn_conv2d::forward_with_bias<IO_T>, convolution_handle,
                          input->data(), weight->data(), bias ? bias->data() : nullptr,
                          output->data(), batch_size, in_channels_, input_h, input_w, out_channels_,
                          output_h, output_w, cudnn_workspace->data(), cudnn_workspace->capacity());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayer::cudnn_backward_data(
    const ConstTensor &grad_output, const ConstTensor &weight, const Tensor &input_grad,
    size_t batch_size, size_t input_h, size_t input_w, size_t output_h, size_t output_w,
    const Tensor &cudnn_workspace, flowHandle_t handle) {
  if (grad_output->device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN backward data requires GPU device");
  }

  size_t shape_key = get_shape_hash(input_grad->shape());

  cuda::cudnn_conv2d::ConvolutionHandle *convolution_handle =
      convolution_handle_cache.at(shape_key);

  return create_cuda_task(handle, cuda::cudnn_conv2d::backward_data<IO_T>, convolution_handle,
                          grad_output->data(), weight->data(), input_grad->data(), batch_size,
                          in_channels_, input_h, input_w, out_channels_, output_h, output_w,
                          cudnn_workspace->data(), cudnn_workspace->capacity());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayer::cudnn_backward_filter(
    const ConstTensor &input, const ConstTensor &grad_output, const Tensor &weight_grad,
    size_t batch_size, size_t input_h, size_t input_w, size_t output_h, size_t output_w,
    const Tensor &cudnn_workspace, flowHandle_t handle) {
  if (grad_output->device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN backward filter requires GPU device");
  }

  size_t shape_key = get_shape_hash(input->shape());

  cuda::cudnn_conv2d::ConvolutionHandle *convolution_handle =
      convolution_handle_cache.at(shape_key);

  return create_cuda_task(handle, cuda::cudnn_conv2d::backward_filter<IO_T>, convolution_handle,
                          input->data(), grad_output->data(), weight_grad->data(), batch_size,
                          in_channels_, input_h, input_w, out_channels_, output_h, output_w,
                          cudnn_workspace->data(), cudnn_workspace->capacity());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayer::cudnn_backward_bias(const ConstTensor &grad_output,
                                                             const Tensor &bias_grad,
                                                             size_t batch_size, size_t output_h,
                                                             size_t output_w, size_t out_channels,
                                                             flowHandle_t handle) {
  if (grad_output->device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN backward bias requires GPU device");
  }

  // compute input shape
  size_t input_h = (output_h - 1) * stride_h_ + kernel_h_ - 2 * pad_h_;
  size_t input_w = (output_w - 1) * stride_w_ + kernel_w_ - 2 * pad_w_;
  size_t shape_key = get_shape_hash({batch_size, in_channels_, input_h, input_w});

  cuda::cudnn_conv2d::ConvolutionHandle *convolution_handle =
      convolution_handle_cache.at(shape_key);

  return create_cuda_task(handle, cuda::cudnn_conv2d::backward_bias<IO_T>, convolution_handle,
                          grad_output->data(), bias_grad->data(), batch_size, out_channels,
                          output_h, output_w);
}
#endif

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayer::compute_conv_forward_impl(
    const ConstTensor &col_data, const ConstTensor &weight_data, const Tensor &output_data,
    const size_t output_size, const size_t kernel_size, const size_t out_channels,
    flowHandle_t handle) {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyConv2DLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (col_data->data_type() != dtype_of<IO_T>() || output_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyConv2DLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight_data->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyConv2DLayer weight tensor dtype mismatch with dispatch Param_T");
  }
  if (col_data->device_type() != weight_data->device_type() ||
      weight_data->device_type() != output_data->device_type()) {
    throw std::runtime_error("All tensors must be on the same device for conv forward");
  }

  if (col_data->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::conv2d_nchw::compute_conv_forward<Compute_T>,
                           col_data->data_as<Compute_T>(), weight_data->data_as<Compute_T>(),
                           output_data->data_as<Compute_T>(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (col_data->device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::conv2d_nchw::compute_conv_forward<Compute_T>,
                            col_data->data_as<Compute_T>(), weight_data->data_as<Compute_T>(),
                            output_data->data_as<Compute_T>(), output_size, kernel_size,
                            out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv forward");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayer::compute_weight_gradients_impl(
    const ConstTensor &col_data, const ConstTensor &gradient_data, const Tensor &weight_grad_data,
    const size_t output_size, const size_t kernel_size, const size_t out_channels,
    flowHandle_t handle) {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyConv2DLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (col_data->data_type() != dtype_of<IO_T>() || gradient_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyConv2DLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight_grad_data->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyConv2DLayer weight grad_output dtype mismatch with dispatch Param_T");
  }
  if (col_data->device_type() != gradient_data->device_type() ||
      gradient_data->device_type() != weight_grad_data->device_type()) {
    throw std::runtime_error("All tensors must be on the same device for conv weight gradients");
  }

  if (col_data->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::conv2d_nchw::compute_weight_gradients<Compute_T>,
                           col_data->data_as<Compute_T>(), gradient_data->data_as<Compute_T>(),
                           weight_grad_data->data_as<Compute_T>(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (col_data->device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::conv2d_nchw::compute_weight_gradients<Compute_T>,
                            col_data->data_as<Compute_T>(), gradient_data->data_as<Compute_T>(),
                            weight_grad_data->data_as<Compute_T>(), output_size, kernel_size,
                            out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv weight gradients");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayer::compute_input_gradients_impl(
    const ConstTensor &gradient_data, const ConstTensor &weight_data, const Tensor &col_grad_data,
    const size_t output_size, const size_t kernel_size, const size_t out_channels,
    flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyConv2DLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (gradient_data->data_type() != dtype_of<IO_T>() ||
      col_grad_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyConv2DLayer IO tensor dtype mismatch with dispatch IO_T");
  }
  if (weight_data->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyConv2DLayer weight tensor dtype mismatch with dispatch Param_T");
  }
  if (gradient_data->device_type() != weight_data->device_type() ||
      weight_data->device_type() != col_grad_data->device_type()) {
    throw std::runtime_error("All tensors must be on the same device for conv input gradients");
  }

  if (gradient_data->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::conv2d_nchw::compute_input_gradients<Compute_T>,
                           gradient_data->data_as<Compute_T>(), weight_data->data_as<Compute_T>(),
                           col_grad_data->data_as<Compute_T>(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (gradient_data->device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::conv2d_nchw::compute_input_gradients<Compute_T>,
                            gradient_data->data_as<Compute_T>(), weight_data->data_as<Compute_T>(),
                            col_grad_data->data_as<Compute_T>(), output_size, kernel_size,
                            out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv input gradients");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayer::compute_bias_gradients_impl(
    const ConstTensor &gradient_data, const Tensor &bias_grad_data, const size_t batch_size,
    const size_t output_h, const size_t output_w, const size_t out_channels, flowHandle_t handle) {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyConv2DLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (gradient_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyConv2DLayer grad_output dtype mismatch with dispatch IO_T");
  }
  if (bias_grad_data->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error(
        "LegacyConv2DLayer bias grad_output dtype mismatch with dispatch Param_T");
  }
  if (gradient_data->device_type() != bias_grad_data->device_type()) {
    throw std::runtime_error("Gradient and bias grad_output tensors must be on the same device");
  }

  if (gradient_data->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::conv2d_nchw::compute_bias_gradients<Compute_T>,
                           gradient_data->data_as<Compute_T>(),
                           bias_grad_data->data_as<Compute_T>(), batch_size, output_h, output_w,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (gradient_data->device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::conv2d_nchw::compute_bias_gradients<Compute_T>,
                            gradient_data->data_as<Compute_T>(),
                            bias_grad_data->data_as<Compute_T>(), batch_size, output_h, output_w,
                            out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv bias gradients");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> LegacyConv2DLayer::add_bias_to_output_impl(
    const Tensor &output_data, const ConstTensor &bias_data, const size_t batch_size,
    const size_t output_h, const size_t output_w, const size_t out_channels,
    flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T> || !std::is_same_v<Param_T, Compute_T>) {
    throw std::runtime_error(
        "LegacyConv2DLayer mixed dtype dispatch not implemented (io/param/compute must match).");
  }
  if (output_data->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("LegacyConv2DLayer output dtype mismatch with dispatch IO_T");
  }
  if (bias_data->data_type() != dtype_of<Param_T>()) {
    throw std::runtime_error("LegacyConv2DLayer bias dtype mismatch with dispatch Param_T");
  }
  if (output_data->device_type() != bias_data->device_type()) {
    throw std::runtime_error("Output and bias tensors must be on the same device");
  }

  if (output_data->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::conv2d_nchw::add_bias_to_output<Compute_T>,
                           output_data->data_as<Compute_T>(), bias_data->data_as<Compute_T>(),
                           batch_size, output_h, output_w, out_channels);
  }
#ifdef USE_CUDA
  else if (output_data->device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::conv2d_nchw::add_bias_to_output<Compute_T>,
                            output_data->data_as<Compute_T>(), bias_data->data_as<Compute_T>(),
                            batch_size, output_h, output_w, out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv add bias to output");
  }
  return nullptr;
}

LayerConfig LegacyConv2DLayer::get_config() const {
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

std::vector<size_t> LegacyConv2DLayer::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("LegacyConv2DLayer expects 4D input including batch size");
  }

  size_t batch_size = input_shape[0];

  size_t output_h = (input_shape[2] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[3] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {batch_size, out_channels_, output_h, output_w};
}

std::unique_ptr<LegacyConv2DLayer> LegacyConv2DLayer::create_from_config(
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
  return std::make_unique<LegacyConv2DLayer>(in_channels, out_channels, kernel_h, kernel_w,
                                             stride_h, stride_w, pad_h, pad_w, use_bias,
                                             config.name);
}

size_t LegacyConv2DLayer::fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 4) return 0;
  size_t batch_size = shape[0];
  size_t input_h = shape[2];
  size_t input_w = shape[3];
  size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  return batch_size * in_channels_ * kernel_h_ * kernel_w_ * sizeof(float) +
         batch_size * out_channels_ * output_h * output_w * sizeof(float);
}

size_t LegacyConv2DLayer::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 4) return 0;
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  build_graph(shape);
  size_t shape_key = get_shape_hash(shape);

  ConvolutionStats stats = stats_cache.at(shape_key);
  return stats.fwd_workspace_size;
#else
  return 0;
#endif
}

size_t LegacyConv2DLayer::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 4) return 0;
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  build_graph(shape);
  size_t shape_key = get_shape_hash(shape);

  ConvolutionStats stats = stats_cache.at(shape_key);
  return stats.fwd_workspace_size;
#else
  return 0;
#endif
}

size_t LegacyConv2DLayer::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
  if (shape.empty() || shape.size() < 4) return 0;
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) return 0;
  build_graph(shape);
  size_t shape_key = get_shape_hash(shape);
  ConvolutionStats stats = stats_cache.at(shape_key);
  return stats.bgrad_workspace_size + stats.wgrad_workspace_size + stats.dgrad_workspace_size;
#else
  return 0;
#endif
}

}  // namespace tnn
