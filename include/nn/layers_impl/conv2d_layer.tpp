/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "device/device_type.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"

#include "device/task.hpp"
#include "nn/layers_impl/cpu/conv2d_ops.hpp"
#include "nn/layers_impl/cuda/conv2d_ops.hpp"
#include "ops/ops.hpp"
#ifdef USE_CUDNN
#include "nn/layers_impl/cuda/cudnn_conv2d_ops.hpp"
#endif
#include "tensor/tensor_ops.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace tnn {

template <typename T>
Conv2DLayer<T>::Conv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h,
                            size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                            size_t pad_w, bool use_bias, const std::string &name)
    : ParameterizedLayer<T>(name), in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w),
      pad_h_(pad_h), pad_w_(pad_w), use_bias_(use_bias) {}

template <typename T> Conv2DLayer<T>::~Conv2DLayer() {
#ifdef USE_CUDNN
  if (cudnn_handle_) {
    cuda::cudnn_conv2d::destroy_convolution_handle(cudnn_handle_);
    cudnn_handle_ = nullptr;
  }
#endif
}

template <typename T> void Conv2DLayer<T>::initialize_params() {
  weights_ = Tensor<T>({out_channels_, in_channels_, kernel_h_, kernel_w_}, this->device_);
  weight_gradients_ = Tensor<T>({out_channels_, in_channels_, kernel_h_, kernel_w_}, this->device_);
  weight_gradients_.fill(T(0));

  if (use_bias_) {
    bias_ = Tensor<T>({out_channels_, 1, 1, 1}, this->device_);
    bias_gradients_ = Tensor<T>({out_channels_, 1, 1, 1}, this->device_);
    bias_.fill(T(0));
    bias_gradients_.fill(T(0));
  }

#ifdef USE_CUDNN
  // cuDNN workspace will be initialized on first forward pass
  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_workspace_ = make_array_ptr<T[]>(this->device_, 0);
  } else
#endif
  {
    temp_output_buffer_ = make_array_ptr<T[]>(this->device_, 0);
    temp_gradient_buffer_ = make_array_ptr<T[]>(this->device_, 0);
    temp_col_grad_matrix_buffer_ = make_array_ptr<T[]>(this->device_, 0);
  }

  T fan_in = static_cast<T>(in_channels_ * kernel_h_ * kernel_w_);
  T fan_out = static_cast<T>(out_channels_ * kernel_h_ * kernel_w_);
  T std_dev = std::sqrt(static_cast<T>(2.0) / (fan_in + fan_out));
  if (this->use_seed_) {
    weights_.fill_random_normal(T(0), std_dev, this->srand_seed_);
  } else {
    weights_.fill_random_normal(T(0), std_dev);
  }
}

template <typename T>
const Tensor<T> &Conv2DLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  if (!this->initialized_) {
    throw std::runtime_error("Conv2DLayer must be initialized before forward pass.");
  }
  if (input.channels() != in_channels_) {
    std::cerr << "Input shape: " << input.channels() << " channels, expected: " << in_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in Conv2DLayer");
  }

  const Tensor<T> *current = &input;
  Tensor<T> device_input;
  if (input.device() != this->device_) {
    device_input = input.to_device(this->device_);
    current = &device_input;
  }

#ifdef USE_CUDNN
  if (this->device_->device_type() == DeviceType::GPU) {
    return cudnn_forward(current, micro_batch_id);
  } else
#endif
  {
    return def_forward(current, micro_batch_id);
  }
}

template <typename T>
const Tensor<T> &Conv2DLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  if (!this->initialized_) {
    throw std::runtime_error("Conv2DLayer must be initialized before backward pass.");
  }
  const Tensor<T> *current_gradient = &gradient;
  Tensor<T> device_gradient;
  if (gradient.device() != this->device_) {
    device_gradient = gradient.to_device(this->device_);
    current_gradient = &device_gradient;
  }

#ifdef USE_CUDNN
  if (this->device_->device_type() == DeviceType::GPU) {
    return cudnn_backward(current_gradient, micro_batch_id);
  } else
#endif
  {
    return def_backward(current_gradient, micro_batch_id);
  }
}

template <typename T>
const Tensor<T> &Conv2DLayer<T>::def_forward(const Tensor<T> *current, size_t micro_batch_id) {
  const size_t batch_size = current->batch_size();
  const size_t channels_ = current->channels();
  const size_t input_h = current->height();
  const size_t input_w = current->width();

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  micro_batch_input_shapes_[micro_batch_id] = {batch_size, channels_, input_h, input_w};
  Tensor<T> &output = this->get_buffer({batch_size, out_channels_, output_h, output_w});

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;
  size_t col_matrix_size = kernel_size * output_size;

  // Ensure per-microbatch col buffer is allocated
  auto col_buffer_it = micro_batch_col_buffers_.find(micro_batch_id);
  if (col_buffer_it == micro_batch_col_buffers_.end()) {
    micro_batch_col_buffers_[micro_batch_id] = make_array_ptr<T[]>(this->device_, col_matrix_size);
  } else {
    col_buffer_it->second.ensure(col_matrix_size);
  }

  size_t output_buffer_size = out_channels_ * output_size;
  temp_output_buffer_.ensure(output_buffer_size);

  im2col_task_ = im2col(*current, micro_batch_col_buffers_[micro_batch_id], kernel_h_, kernel_w_,
                        stride_h_, stride_w_, pad_h_, pad_w_, "default");

  forward_task_ =
      compute_conv_forward(micro_batch_col_buffers_[micro_batch_id], weights_.data_ptr(),
                           temp_output_buffer_, output_size, kernel_size, out_channels_, "default");

  cnhw_to_nchw_task_ = ops::cnhw_to_nchw(temp_output_buffer_, output.data_ptr(), batch_size,
                                         out_channels_, output_h, output_w);

  if (use_bias_) {
    add_bias_task_ = add_bias_to_output(output.data_ptr(), bias_.data_ptr(), batch_size, output_h,
                                        output_w, out_channels_, "default");
  }

  return output;
}

template <typename T>
const Tensor<T> &Conv2DLayer<T>::def_backward(const Tensor<T> *current_gradient,
                                              size_t micro_batch_id) {
  auto it_input_shape = micro_batch_input_shapes_.find(micro_batch_id);

  if (it_input_shape == micro_batch_input_shapes_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  const auto &input_shape = it_input_shape->second;

  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[2];
  const size_t input_w = input_shape[3];
  const size_t output_h = current_gradient->height();
  const size_t output_w = current_gradient->width();

  Tensor<T> &grad_input = this->get_buffer(input_shape);
  grad_input.fill(T(0));

  // Fallback to im2col + GEMM approach
  auto it_col_buffer = micro_batch_col_buffers_.find(micro_batch_id);
  if (it_col_buffer == micro_batch_col_buffers_.end()) {
    throw std::runtime_error("No cached col buffer found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;
  size_t col_grad_matrix_size = kernel_size * output_size;

  size_t gradient_buffer_size = out_channels_ * output_size;
  temp_gradient_buffer_.ensure(gradient_buffer_size);
  temp_col_grad_matrix_buffer_.ensure(col_grad_matrix_size);

  nchw_to_cnhw_task_ = ops::nchw_to_cnhw(current_gradient->data_ptr(), temp_gradient_buffer_,
                                         batch_size, out_channels_, output_h, output_w, "default");

  weight_grad_task_ = compute_weight_gradients(it_col_buffer->second, temp_gradient_buffer_,
                                               weight_gradients_.data_ptr(), output_size,
                                               kernel_size, out_channels_, "default");

  input_grad_task_ = compute_input_gradients(temp_gradient_buffer_, weights_.data_ptr(),
                                             temp_col_grad_matrix_buffer_, output_size, kernel_size,
                                             out_channels_, "default");

  col2im_task_ =
      col2im(temp_col_grad_matrix_buffer_, grad_input.data_ptr(), batch_size, in_channels_, input_h,
             input_w, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, "default");

  if (use_bias_) {
    bias_grad_task_ =
        compute_bias_gradients(current_gradient->data_ptr(), bias_gradients_.data_ptr(), batch_size,
                               output_h, output_w, out_channels_, "default");
  }

  return grad_input;
}

#ifdef USE_CUDNN
template <typename T>
const Tensor<T> &Conv2DLayer<T>::cudnn_forward(const Tensor<T> *current, size_t micro_batch_id) {
  const size_t batch_size = current->batch_size();
  const size_t input_h = current->height();
  const size_t input_w = current->width();

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  Tensor<T> &output = this->get_buffer({batch_size, out_channels_, output_h, output_w});

  if (!cudnn_handle_) {
    cudnn_handle_ = cuda::cudnn_conv2d::initialize_convolution_handle(
        batch_size, in_channels_, input_h, input_w, out_channels_, kernel_h_, kernel_w_, stride_h_,
        stride_w_, pad_h_, pad_w_);

    auto workspace_sizes = cuda::cudnn_conv2d::get_workspace_sizes(cudnn_handle_, batch_size);
    size_t max_workspace = std::max(
        {workspace_sizes.fwd_size, workspace_sizes.bwd_data_size, workspace_sizes.bwd_filter_size});
    cudnn_workspace_.ensure(max_workspace);
  } else {
    cuda::cudnn_conv2d::update_batch_size(cudnn_handle_, batch_size, in_channels_, input_h, input_w,
                                          out_channels_, output_h, output_w);
    // Recalculate and ensure workspace size after batch size update
    auto workspace_sizes = cuda::cudnn_conv2d::get_workspace_sizes(cudnn_handle_, batch_size);
    size_t max_workspace = std::max(
        {workspace_sizes.fwd_size, workspace_sizes.bwd_data_size, workspace_sizes.bwd_filter_size});
    cudnn_workspace_.ensure(max_workspace);
  }

  auto it_input_cache = micro_batch_inputs_cache_.find(micro_batch_id);
  if (it_input_cache == micro_batch_inputs_cache_.end()) {
    micro_batch_inputs_cache_[micro_batch_id] = current->clone();
  } else {
    // Resize the cached tensor to match current input shape
    it_input_cache->second.ensure(current->shape());
    ops::copy(current->data_ptr(), it_input_cache->second.data_ptr(), current->size());
  }

  // Use cuDNN forward
  const T *bias_ptr = use_bias_ ? bias_.data_ptr().get() : nullptr;
  forward_task_ =
      cudnn_forward(current->data_ptr(), weights_.data_ptr(), bias_ptr, output.data_ptr(),
                    batch_size, input_h, input_w, output_h, output_w, "default");

  return output;
}

template <typename T>
const Tensor<T> &Conv2DLayer<T>::cudnn_backward(const Tensor<T> *current_gradient,
                                                size_t micro_batch_id) {
  auto it_input_cache = micro_batch_inputs_cache_.find(micro_batch_id);
  if (it_input_cache == micro_batch_inputs_cache_.end()) {
    throw std::runtime_error("No cached input found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  const auto &input_shape = it_input_cache->second.shape();

  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[2];
  const size_t input_w = input_shape[3];
  const size_t output_h = current_gradient->height();
  const size_t output_w = current_gradient->width();

  Tensor<T> &grad_input = this->get_buffer(input_shape);
  grad_input.fill(T(0));

  const Tensor<T> &cached_input = it_input_cache->second;

  // cuDNN backward for weights
  weight_grad_task_ = cudnn_backward_filter(cached_input.data_ptr(), current_gradient->data_ptr(),
                                            weight_gradients_.data_ptr(), batch_size, input_h,
                                            input_w, output_h, output_w, "default");

  // cuDNN backward for input
  input_grad_task_ =
      cudnn_backward_data(current_gradient->data_ptr(), weights_.data_ptr(), grad_input.data_ptr(),
                          batch_size, input_h, input_w, output_h, output_w, "default");

  // cuDNN backward for bias
  if (use_bias_) {
    bias_grad_task_ = cudnn_backward_bias(current_gradient->data_ptr(), bias_gradients_.data_ptr(),
                                          batch_size, output_h, output_w, out_channels_, "default");
  }

  return grad_input;
}

template <typename T>
std::unique_ptr<Task>
Conv2DLayer<T>::cudnn_forward(const device_ptr<T[]> &input_data, const device_ptr<T[]> &weight_data,
                              const T *bias_data, device_ptr<T[]> &output_data, size_t batch_size,
                              size_t input_h, size_t input_w, size_t output_h, size_t output_w,
                              const std::string &flow_id) {
  if (input_data.device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN forward requires GPU device");
  }

  return create_gpu_task(flow_id, cuda::cudnn_conv2d::forward_with_bias<T>, cudnn_handle_,
                         input_data.get(), weight_data.get(), bias_data, output_data.get(),
                         batch_size, in_channels_, input_h, input_w, out_channels_, output_h,
                         output_w, cudnn_workspace_.get(), cudnn_workspace_.capacity());
}

template <typename T>
std::unique_ptr<Task> Conv2DLayer<T>::cudnn_backward_data(
    const device_ptr<T[]> &gradient_data, const device_ptr<T[]> &weight_data,
    device_ptr<T[]> &input_grad_data, size_t batch_size, size_t input_h, size_t input_w,
    size_t output_h, size_t output_w, const std::string &flow_id) {
  if (gradient_data.device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN backward data requires GPU device");
  }

  return create_gpu_task(flow_id, cuda::cudnn_conv2d::backward_data<T>, cudnn_handle_,
                         gradient_data.get(), weight_data.get(), input_grad_data.get(), batch_size,
                         in_channels_, input_h, input_w, out_channels_, output_h, output_w,
                         cudnn_workspace_.get(), cudnn_workspace_.capacity());
}

template <typename T>
std::unique_ptr<Task> Conv2DLayer<T>::cudnn_backward_filter(
    const device_ptr<T[]> &input_data, const device_ptr<T[]> &gradient_data,
    device_ptr<T[]> &weight_grad_data, size_t batch_size, size_t input_h, size_t input_w,
    size_t output_h, size_t output_w, const std::string &flow_id) {
  if (gradient_data.device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN backward filter requires GPU device");
  }

  return create_gpu_task(flow_id, cuda::cudnn_conv2d::backward_filter<T>, cudnn_handle_,
                         input_data.get(), gradient_data.get(), weight_grad_data.get(), batch_size,
                         in_channels_, input_h, input_w, out_channels_, output_h, output_w,
                         cudnn_workspace_.get(), cudnn_workspace_.capacity());
}

template <typename T>
std::unique_ptr<Task> Conv2DLayer<T>::cudnn_backward_bias(const device_ptr<T[]> &gradient_data,
                                                          device_ptr<T[]> &bias_grad_data,
                                                          size_t batch_size, size_t output_h,
                                                          size_t output_w, size_t out_channels,
                                                          const std::string &flow_id) {
  if (gradient_data.device_type() != DeviceType::GPU) {
    throw std::runtime_error("cuDNN backward bias requires GPU device");
  }

  return create_gpu_task(flow_id, cuda::cudnn_conv2d::backward_bias<T>, cudnn_handle_,
                         gradient_data.get(), bias_grad_data.get(), batch_size, out_channels,
                         output_h, output_w);
}
#endif

template <typename T>
std::unique_ptr<Task> Conv2DLayer<T>::compute_conv_forward(
    const device_ptr<T[]> &col_data, const device_ptr<T[]> &weight_data,
    device_ptr<T[]> &output_data, const size_t output_size, const size_t kernel_size,
    const size_t out_channels, const std::string &flow_id) {
  if (col_data.device_type() != weight_data.device_type() ||
      weight_data.device_type() != output_data.device_type()) {
    throw std::runtime_error("All tensors must be on the same device for conv forward");
  }

  if (col_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::conv2d::compute_conv_forward<T>, col_data.get(),
                           weight_data.get(), output_data.get(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (col_data.device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::conv2d::compute_conv_forward<T>, col_data.get(),
                           weight_data.get(), output_data.get(), output_size, kernel_size,
                           out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv forward");
  }
}

template <typename T>
std::unique_ptr<Task> Conv2DLayer<T>::compute_weight_gradients(
    const device_ptr<T[]> &col_data, const device_ptr<T[]> &gradient_data,
    device_ptr<T[]> &weight_grad_data, const size_t output_size, const size_t kernel_size,
    const size_t out_channels, const std::string &flow_id) {
  if (col_data.device_type() != gradient_data.device_type() ||
      gradient_data.device_type() != weight_grad_data.device_type()) {
    throw std::runtime_error("All tensors must be on the same device for conv weight gradients");
  }

  if (col_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::conv2d::compute_weight_gradients<T>, col_data.get(),
                           gradient_data.get(), weight_grad_data.get(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (col_data.device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::conv2d::compute_weight_gradients<T>, col_data.get(),
                           gradient_data.get(), weight_grad_data.get(), output_size, kernel_size,
                           out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv weight gradients");
  }
}

template <typename T>
std::unique_ptr<Task> Conv2DLayer<T>::compute_input_gradients(
    const device_ptr<T[]> &gradient_data, const device_ptr<T[]> &weight_data,
    device_ptr<T[]> &col_grad_data, const size_t output_size, const size_t kernel_size,
    const size_t out_channels, const std::string &flow_id) const {
  if (gradient_data.device_type() != weight_data.device_type() ||
      weight_data.device_type() != col_grad_data.device_type()) {
    throw std::runtime_error("All tensors must be on the same device for conv input gradients");
  }

  if (gradient_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::conv2d::compute_input_gradients<T>, gradient_data.get(),
                           weight_data.get(), col_grad_data.get(), output_size, kernel_size,
                           out_channels);
  }
#ifdef USE_CUDA
  else if (gradient_data.device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::conv2d::compute_input_gradients<T>, gradient_data.get(),
                           weight_data.get(), col_grad_data.get(), output_size, kernel_size,
                           out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv input gradients");
  }
}

template <typename T>
std::unique_ptr<Task>
Conv2DLayer<T>::compute_bias_gradients(const device_ptr<T[]> &gradient_data,
                                       device_ptr<T[]> &bias_grad_data, const size_t batch_size,
                                       const size_t output_h, const size_t output_w,
                                       const size_t out_channels, const std::string &flow_id) {
  if (gradient_data.device_type() != bias_grad_data.device_type()) {
    throw std::runtime_error("Gradient and bias gradient tensors must be on the same device");
  }

  if (gradient_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::conv2d::compute_bias_gradients<T>, gradient_data.get(),
                           bias_grad_data.get(), batch_size, output_h, output_w, out_channels);
  }
#ifdef USE_CUDA
  else if (gradient_data.device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::conv2d::compute_bias_gradients<T>, gradient_data.get(),
                           bias_grad_data.get(), batch_size, output_h, output_w, out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv bias gradients");
  }
}

template <typename T>
std::unique_ptr<Task>
Conv2DLayer<T>::add_bias_to_output(device_ptr<T[]> &output_data, const device_ptr<T[]> &bias_data,
                                   const size_t batch_size, const size_t output_h,
                                   const size_t output_w, const size_t out_channels,
                                   const std::string &flow_id) const {
  if (output_data.device_type() != bias_data.device_type()) {
    throw std::runtime_error("Output and bias tensors must be on the same device");
  }

  if (output_data.device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::conv2d::add_bias_to_output<T>, output_data.get(),
                           bias_data.get(), batch_size, output_h, output_w, out_channels);
  }
#ifdef USE_CUDA
  else if (output_data.device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::conv2d::add_bias_to_output<T>, output_data.get(),
                           bias_data.get(), batch_size, output_h, output_w, out_channels);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for conv add bias to output");
  }
}

template <typename T> std::string Conv2DLayer<T>::type() const { return "conv2d"; }

template <typename T> LayerConfig Conv2DLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["in_channels"] = in_channels_;
  config.parameters["out_channels"] = out_channels_;
  config.parameters["kernel_h"] = kernel_h_;
  config.parameters["kernel_w"] = kernel_w_;
  config.parameters["stride_h"] = stride_h_;
  config.parameters["stride_w"] = stride_w_;
  config.parameters["pad_h"] = pad_h_;
  config.parameters["pad_w"] = pad_w_;
  config.parameters["use_bias"] = use_bias_;
  config.parameters["optimized"] = std::string("cudnn");
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> Conv2DLayer<T>::clone() const {
  return std::make_unique<Conv2DLayer<T>>(in_channels_, out_channels_, kernel_h_, kernel_w_,
                                          stride_h_, stride_w_, pad_h_, pad_w_, use_bias_,
                                          this->name_);
}

template <typename T>
std::vector<size_t>
Conv2DLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("Conv2DLayer expects 4D input");
  }

  size_t output_h = (input_shape[2] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[3] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {input_shape[0], out_channels_, output_h, output_w};
}

template <typename T> void Conv2DLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.push_back(&weights_);
  if (use_bias_) {
    params.push_back(&bias_);
  }
}

template <typename T> void Conv2DLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.push_back(&weight_gradients_);
  if (use_bias_) {
    grads.push_back(&bias_gradients_);
  }
}

template <typename T> void Conv2DLayer<T>::clear_gradients() {
  weight_gradients_.fill(T(0))->sync();
  if (use_bias_) {
    bias_gradients_.fill(T(0))->sync();
  }
}

template <typename T>
uint64_t Conv2DLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];
  size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  size_t output_size = batch_size * output_h * output_w;
  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;

  // Main convolution computation: 2 FLOPs per MAC (multiply-add)
  uint64_t conv_flops = 2ULL * out_channels_ * kernel_size * output_size;

  // Bias addition: 1 FLOP per output element
  uint64_t bias_flops = use_bias_ ? (batch_size * out_channels_ * output_h * output_w) : 0;

  return conv_flops + bias_flops;
}

template <typename T>
uint64_t Conv2DLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[2];
  size_t input_w = input_shape[3];
  size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  size_t output_size = batch_size * output_h * output_w;
  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;

  // weight gradients: gradient × im2col_input^T (2 FLOPs per MAC)
  uint64_t weight_grad_flops = 2ULL * out_channels_ * kernel_size * output_size;

  // input gradients: weights^T × gradient (2 FLOPs per MAC)
  uint64_t input_grad_flops = 2ULL * out_channels_ * kernel_size * output_size;

  // bias gradients: reduction across batch and spatial dimensions (1 FLOP per add)
  uint64_t bias_grad_flops = use_bias_ ? (batch_size * out_channels_ * output_h * output_w) : 0;

  return weight_grad_flops + input_grad_flops + bias_grad_flops;
}

template <typename T>
uint64_t Conv2DLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) const {
  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t Conv2DLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) const {
  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T> size_t Conv2DLayer<T>::cached_memory_bytes() const {
  size_t total_bytes = 0;
  for (const auto &pair : micro_batch_col_buffers_) {
    total_bytes += pair.second.capacity() * sizeof(T);
  }
#ifdef USE_CUDNN
  size_t input_cache_bytes = 0;
  for (const auto &pair : micro_batch_inputs_cache_) {
    input_cache_bytes += pair.second.size() * sizeof(T);
  }
  // std::cout << "Cudnn input cache bytes: " << input_cache_bytes << std::endl;
  total_bytes += input_cache_bytes;
#endif
  total_bytes += temp_output_buffer_.capacity() * sizeof(T);
  // std::cout << "Temp output buffer bytes: " << temp_output_buffer_.capacity() * sizeof(T)
  //           << std::endl;
  total_bytes += temp_gradient_buffer_.capacity() * sizeof(T);
  // std::cout << "Temp gradient buffer bytes: " << temp_gradient_buffer_.capacity() * sizeof(T)
  //           << std::endl;
  total_bytes += temp_col_grad_matrix_buffer_.capacity() * sizeof(T);
  // std::cout << "Temp col grad matrix buffer bytes: "
  //           << temp_col_grad_matrix_buffer_.capacity() * sizeof(T) << std::endl;
  total_bytes += cudnn_workspace_.capacity() * sizeof(T);
  // std::cout << "Cudnn workspace bytes: " << cudnn_workspace_.capacity() * sizeof(T) << std::endl;
  total_bytes += Layer<T>::cached_memory_bytes();
  // std::cout << "base layer cached bytes: " << Layer<T>::cached_memory_bytes() << std::endl;
  return total_bytes;
}

// Explicit template instantiations
template class Conv2DLayer<float>;

} // namespace tnn
