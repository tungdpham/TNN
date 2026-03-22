/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/conv2d_layer.hpp"

#include <numeric>

#include "device/device_type.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/cpu/conv2d_nhwc_ops.hpp"
#include "tensor/tensor.hpp"
#include "utils/misc.hpp"
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

namespace tnn {

Conv2DLayer::Conv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
                         size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                         bool use_bias, const std::string &name)
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

Conv2DLayer::~Conv2DLayer() {
#ifdef USE_CUDNN
  for (auto &pair : fe_handle_cache) {
    if (pair.second) {
      cuda::cudnn_conv2d::destroy_fe_handle(pair.second);
    }
  }
  fe_handle_cache.clear();
  stats_cache.clear();
#endif
}

void Conv2DLayer::init_impl() {
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
 * ! Only support GPU device with cuDNN backend. CPU implementation is to be added.
 * @tparam T
 * @param input input tensor in NHWC format
 * @param output output tensor in NHWC format
 * @param mb_id micro batch id for caching input
 */

void Conv2DLayer::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  if (input->dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NHWC)");
  }

  size_t channels = input->dimension(3);

  if (channels != in_channels_) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << in_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in Conv2DLayer");
  }

  const size_t batch_size = input->dimension(0);
  const size_t input_h = input->dimension(1);
  const size_t input_w = input->dimension(2);

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  output->ensure({batch_size, output_h, output_w, out_channels_});

  if (this->is_training_) {
    set_immutable_cache(mb_id, "input", input);
  }

#ifdef USE_CUDNN
  if (this->device().device_type() == DeviceType::GPU) {
    cudnn_forward(input, output, mb_id);
    return;
  }
#endif

  def_forward(input, output, mb_id);
}

/**
 * @brief Perform convolution 2d backward on grad_output and save it to grad_input.
 * ! Only support GPU device with cuDNN backend. CPU implementation is to be added.
 * @tparam T
 * @param grad_output upstream grad_output tensor in NHWC format
 * @param grad_input output grad_output tensor in NHWC format
 * @param mb_id micro batch id for caching input
 */

void Conv2DLayer::backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                                size_t mb_id) {
  if (grad_output->dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NHWC)");
  }

  size_t channels = grad_output->dimension(3);

  if (channels != out_channels_) {
    std::cerr << "Gradient shape: " << channels << " channels, expected: " << out_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Gradient channel size mismatch in Conv2DLayer");
  }

  ConstTensor &input = this->get_immutable_cache(mb_id, "input");
  const auto &input_shape = input->shape();
  grad_input->ensure(input_shape);

#ifdef USE_CUDNN
  if (this->device().device_type() == DeviceType::GPU) {
    cudnn_backward(grad_output, grad_input, mb_id);
    return;
  }
#endif

  def_backward(grad_output, grad_input, mb_id);
}

void Conv2DLayer::def_forward(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  if (this->device().device_type() == DeviceType::CPU) {
    DISPATCH_DTYPE(io_dtype_, T, {
      create_cpu_task(this->flow_handle_, cpu::conv2d_nhwc::forward<T>, input->data_as<T>(),
                      weights_->data_as<T>(), use_bias_ ? bias_->data_as<T>() : nullptr,
                      output->data_as<T>(), input->dimension(0), input->dimension(1),
                      input->dimension(2), in_channels_, out_channels_, kernel_h_, kernel_w_,
                      stride_h_, stride_w_, pad_h_, pad_w_, output->dimension(1),
                      output->dimension(2), use_bias_);
    });
  } else {
    throw std::runtime_error("Conv2DLayer only supports CPU device in def_forward");
  }
}

void Conv2DLayer::def_backward(const ConstTensor &grad_output, const Tensor &grad_input,
                               size_t mb_id) {
  ConstTensor &input = this->get_immutable_cache(mb_id, "input");

  if (this->device().device_type() == DeviceType::CPU) {
    DISPATCH_DTYPE(io_dtype_, T, {
      create_cpu_task(this->flow_handle_, cpu::conv2d_nhwc::backward_data<T>,
                      grad_output->data_as<T>(), weights_->data_as<T>(), grad_input->data_as<T>(),
                      grad_output->dimension(0), grad_output->dimension(1),
                      grad_output->dimension(2), in_channels_, out_channels_, kernel_h_, kernel_w_,
                      stride_h_, stride_w_, pad_h_, pad_w_, grad_output->dimension(1),
                      grad_output->dimension(2));

      create_cpu_task(this->flow_handle_, cpu::conv2d_nhwc::backward_weights<T>,
                      input->data_as<T>(), grad_output->data_as<T>(),
                      weight_gradients_->data_as<T>(), grad_output->dimension(0),
                      input->dimension(1), input->dimension(2), in_channels_, out_channels_,
                      kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                      grad_output->dimension(1), grad_output->dimension(2));

      if (use_bias_) {
        create_cpu_task(this->flow_handle_, cpu::conv2d_nhwc::backward_bias<T>,
                        grad_output->data_as<T>(), bias_gradients_->data_as<T>(),
                        grad_output->dimension(0), grad_output->dimension(1),
                        grad_output->dimension(2), out_channels_);
      }
    });
  }
}

#ifdef USE_CUDNN
template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> Conv2DLayer::conv2d_forward_task(
    cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats, const ConstTensor &input,
    const Tensor &output, const ConstTensor &weights, const ConstTensor &bias,
    const Tensor &workspace, size_t batch_size, size_t input_h, size_t input_w, size_t output_h,
    size_t output_w, flowHandle_t handle) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("Conv2DLayer IO_T and Param_T must be the same type");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("Conv2DLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(handle, cuda::cudnn_conv2d::run_forward, fe_handle, stats, input->data(),
                          weights->data(), bias_ != nullptr ? bias_->data() : nullptr,
                          output->data(), workspace->data());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> Conv2DLayer::conv2d_backward_data_task(
    cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats,
    const ConstTensor &grad_output, const ConstTensor &weights, const Tensor &grad_input,
    const Tensor &workspace, size_t batch_size, size_t input_h, size_t input_w, size_t output_h,
    size_t output_w, flowHandle_t handle) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("Conv2DLayer IO_T and Param_T must be the same type");
  }
  if (grad_output->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("Conv2DLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(handle, cuda::cudnn_conv2d::run_backward_data, fe_handle, stats,
                          grad_output->data(), weights->data(), grad_input->data(),
                          workspace->data());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> Conv2DLayer::conv2d_backward_weights_and_bias_task(
    cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats, const ConstTensor &input,
    const ConstTensor &grad_output, const Tensor &weight_gradients, const Tensor &bias_gradients,
    const Tensor &workspace, size_t batch_size, size_t input_h, size_t input_w, size_t output_h,
    size_t output_w, flowHandle_t handle) const {
  if (!std::is_same_v<IO_T, Param_T>) {
    throw std::runtime_error("Conv2DLayer IO_T and Param_T must be the same type");
  }
  if (input->data_type() != dtype_of<IO_T>() || grad_output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("Conv2DLayer input/grad_output dtype mismatch with dispatch IO_T");
  }

  return create_cuda_task(handle, cuda::cudnn_conv2d::run_backward_weights_and_bias, fe_handle,
                          stats, input->data(), grad_output->data(), weight_gradients->data(),
                          use_bias_ ? bias_gradients->data() : nullptr, workspace->data());
}

void Conv2DLayer::build_graph(const Vec<size_t> &input_shape) const {
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
    auto io_data_type = cuda::cudnn::to_cudnn_datatype(io_dtype_);
    auto compute_type = cuda::cudnn::to_cudnn_datatype(compute_dtype_);
    fe_handle_cache[shape_key] = cuda::cudnn_conv2d::initialize_fe_handle(
        shared_handle, io_data_type, compute_type, new_stats);
    stats_cache[shape_key] = new_stats;
  }
}

void Conv2DLayer::cudnn_forward(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  const size_t batch_size = input->dimension(0);
  const size_t input_h = input->dimension(1);
  const size_t input_w = input->dimension(2);

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  output->ensure({batch_size, output_h, output_w, out_channels_});

  size_t shape_key = get_shape_hash(input->shape());

  build_graph(input->shape());

  cuda::cudnn_conv2d::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  ConvolutionStats &current_stats = stats_cache.at(shape_key);

  size_t ws_bytes = current_stats.fwd_workspace_size;
  Tensor cudnn_workspace = this->get_workspace({ws_bytes}, DType_t::BYTE);

  DISPATCH_ON_3_DTYPES_TO_METHOD(conv2d_forward_task, fe_handle, current_stats, input, output,
                                 weights_, bias_, cudnn_workspace, batch_size, input_h, input_w,
                                 output_h, output_w, this->flow_handle_);
}

void Conv2DLayer::cudnn_backward(const ConstTensor &grad_output, const Tensor &grad_input,
                                 size_t mb_id) {
  ConstTensor &input = this->get_immutable_cache(mb_id, "input");
  if (!input) {
    throw std::runtime_error("No cached input found for micro-batch ID: " + std::to_string(mb_id));
  }

  const auto &input_shape = input->shape();
  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[1];
  const size_t input_w = input_shape[2];
  const auto &grad_shape = grad_output->shape();
  const size_t output_h = grad_shape[1];
  const size_t output_w = grad_shape[2];

  grad_input->ensure(input_shape);

  size_t shape_key = get_shape_hash(input_shape);

  cuda::cudnn_conv2d::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  ConvolutionStats &current_stats = stats_cache.at(shape_key);

  size_t max_workspace_size =
      std::max({current_stats.wgrad_workspace_size, current_stats.dgrad_workspace_size,
                current_stats.bgrad_workspace_size});

  Tensor cudnn_workspace = this->get_workspace({max_workspace_size}, DType_t::BYTE);

  DISPATCH_ON_3_DTYPES_TO_METHOD(conv2d_backward_weights_and_bias_task, fe_handle, current_stats,
                                 input, grad_output, weight_gradients_, bias_gradients_,
                                 cudnn_workspace, batch_size, input_h, input_w, output_h, output_w,
                                 this->flow_handle_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(conv2d_backward_data_task, fe_handle, current_stats, grad_output,
                                 weights_, grad_input, cudnn_workspace, batch_size, input_h,
                                 input_w, output_h, output_w, this->flow_handle_);
}
#endif

LayerConfig Conv2DLayer::get_config() const {
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

Vec<size_t> Conv2DLayer::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("Conv2DLayer expects 4D input including batch size");
  }

  size_t batch_size = input_shape[0];
  size_t output_h = (input_shape[1] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[2] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {batch_size, output_h, output_w, out_channels_};
}

std::unique_ptr<Conv2DLayer> Conv2DLayer::create_from_config(const LayerConfig &config) {
  size_t in_channels = config.get<size_t>("in_channels");
  size_t out_channels = config.get<size_t>("out_channels");
  size_t kernel_h = config.get<size_t>("kernel_h");
  size_t kernel_w = config.get<size_t>("kernel_w");
  size_t stride_h = config.get<size_t>("stride_h", 1);
  size_t stride_w = config.get<size_t>("stride_w", 1);
  size_t pad_h = config.get<size_t>("pad_h", 0);
  size_t pad_w = config.get<size_t>("pad_w", 0);
  bool use_bias = config.get<bool>("use_bias", true);
  return std::make_unique<Conv2DLayer>(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                       stride_w, pad_h, pad_w, use_bias, config.name);
}

size_t Conv2DLayer::fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const {
  // Cache the input for backward pass
  auto &shape = input_shapes[0];
  size_t input_bytes = std::accumulate(shape.begin(), shape.end(), get_dtype_size(io_dtype_),
                                       std::multiplies<size_t>());
  return input_bytes;
}

size_t Conv2DLayer::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
#ifdef USE_CUDNN
  build_graph(shape);
  const size_t shape_key = get_shape_hash(shape);
  const ConvolutionStats &stats = stats_cache.at(shape_key);
  auto output_shapes = this->output_shapes(input_shapes);
  return stats.fwd_workspace_size + get_shapes_bytes(output_shapes, io_dtype_);
#else
  return 0;
#endif
}

size_t Conv2DLayer::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
#ifdef USE_CUDNN
  if (!allocator_ || allocator_->device().device_type() != DeviceType::GPU) {
    std::cerr << "Conv2DLayer::inf_workspace: No GPU allocator available, returning 0\n";
    return 0;
  }
  build_graph(shape);
  const size_t shape_key = get_shape_hash(shape);
  const ConvolutionStats &stats = stats_cache.at(shape_key);
  auto output_shapes = this->output_shapes(input_shapes);
  return stats.fwd_workspace_size + get_shapes_bytes(output_shapes, io_dtype_);
#else
  return 0;
#endif
}

size_t Conv2DLayer::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  auto &shape = input_shapes[0];
#ifdef USE_CUDNN
  build_graph(shape);
  const size_t shape_key = get_shape_hash(shape);
  const ConvolutionStats &stats = stats_cache.at(shape_key);
  size_t max_workspace_size = std::max(
      {stats.wgrad_workspace_size, stats.dgrad_workspace_size, stats.bgrad_workspace_size});
  size_t grad_input_bytes = 0;
  for (const auto &in_shape : input_shapes) {
    grad_input_bytes += std::accumulate(in_shape.begin(), in_shape.end(), get_dtype_size(io_dtype_),
                                        std::multiplies<size_t>());
  }
  return max_workspace_size + grad_input_bytes;
#else
  return 0;
#endif
}

}  // namespace tnn
