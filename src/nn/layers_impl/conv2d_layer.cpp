/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/conv2d_layer.hpp"
#include "device/cuda/cuda_context.hpp"
#include "device/device_type.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/common/conv2d.hpp"
#include "nn/layers_impl/cuda/cudnn_conv2d_ops.hpp"
#include "type/type.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace tnn {

Conv2DLayer::Conv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
                         size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                         bool use_bias, const std::string &name)
    : ParameterizedLayer(name), in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w),
      pad_h_(pad_h), pad_w_(pad_w), use_bias_(use_bias) {}

Conv2DLayer::~Conv2DLayer() {
  for (auto &pair : fe_handle_cache) {
    if (pair.second) {
      cuda::cudnn_conv2d::destroy_fe_handle(pair.second);
    }
  }
  fe_handle_cache.clear();
}

void Conv2DLayer::init_params() {
  weights_ = make_param_tensor({
      out_channels_,
      kernel_h_,
      kernel_w_,
      in_channels_,
  });
  weight_gradients_ = make_param_tensor({
      out_channels_,
      kernel_h_,
      kernel_w_,
      in_channels_,
  });
  weight_gradients_->fill(0.0f);

  if (use_bias_) {
    bias_ = make_param_tensor({out_channels_});
    bias_gradients_ = make_param_tensor({out_channels_});
    bias_gradients_->fill(0.0f);
  }

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
}

size_t Conv2DLayer::get_shape_hash(size_t n, size_t c, size_t h, size_t w) const {
  size_t seed = 0;
  auto hash_combine = [&](size_t v) {
    seed ^= std::hash<size_t>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  };
  hash_combine(n);
  hash_combine(c);
  hash_combine(h);
  hash_combine(w);
  return seed;
}

/**
 * @brief Perform convolution 2d forward on input and save it to output->
 * ! Only support GPU device with cuDNN backend. CPU implementation is to be added.
 * @tparam T
 * @param input input tensor in NHWC format
 * @param output output tensor in NHWC format
 * @param micro_batch_id micro batch id for caching input
 */

void Conv2DLayer::forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  if (input->dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NHWC)");
  }

  size_t channels = input->dimension(3);

  if (channels != in_channels_) {
    std::cerr << "Input shape: " << channels << " channels, expected: " << in_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Input channel size mismatch in Conv2DLayer");
  }

  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_forward(input, output, micro_batch_id);
  } else {
    throw std::runtime_error("CPU implementation for Conv2DLayer not implemented");
  }
}

/**
 * @brief Perform convolution 2d backward on gradient and save it to grad_input.
 * ! Only support GPU device with cuDNN backend. CPU implementation is to be added.
 * @tparam T
 * @param gradient upstream gradient tensor in NHWC format
 * @param grad_input output gradient tensor in NHWC format
 * @param micro_batch_id micro batch id for caching input
 */

void Conv2DLayer::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t micro_batch_id) {
  if (gradient->dims() != 4) {
    throw std::invalid_argument("Conv2D: Input tensor must be 4-dimensional (NHWC)");
  }

  size_t channels = gradient->dimension(3);

  if (channels != out_channels_) {
    std::cerr << "Gradient shape: " << channels << " channels, expected: " << out_channels_
              << " channels" << std::endl;
    throw std::invalid_argument("Gradient channel size mismatch in Conv2DLayer");
  }

  if (this->device_->device_type() == DeviceType::GPU) {
    cudnn_backward(gradient, grad_input, micro_batch_id);
  } else {
    throw std::runtime_error("CPU implementation for Conv2DLayer not implemented");
  }
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task>
Conv2DLayer::conv2d_forward_task(cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats,
                                 const Tensor &input, Tensor &output, const Tensor &weights,
                                 const Tensor &bias, Tensor &workspace, size_t batch_size,
                                 size_t input_h, size_t input_w, size_t output_h, size_t output_w,
                                 const std::string &flow_id) const {
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("Conv2DLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_gpu_task(flow_id, cuda::cudnn_conv2d::run_forward, fe_handle, stats, input->data(),
                         weights->data(), bias_ != nullptr ? bias_->data() : nullptr,
                         output->data(), workspace->data());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> Conv2DLayer::conv2d_backward_data_task(
    cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats, const Tensor &gradient,
    const Tensor &weights, Tensor &grad_input, Tensor &workspace, size_t batch_size, size_t input_h,
    size_t input_w, size_t output_h, size_t output_w, const std::string &flow_id) const {
  if (gradient->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("Conv2DLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  return create_gpu_task(flow_id, cuda::cudnn_conv2d::run_backward_data, fe_handle, stats,
                         gradient->data(), weights->data(), grad_input->data(), workspace->data());
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> Conv2DLayer::conv2d_backward_weights_and_bias_task(
    cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats, const Tensor &input,
    const Tensor &gradient, Tensor &weight_gradients, Tensor &bias_gradients, Tensor &workspace,
    size_t batch_size, size_t input_h, size_t input_w, size_t output_h, size_t output_w,
    const std::string &flow_id) const {
  if (input->data_type() != dtype_of<IO_T>() || gradient->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("Conv2DLayer input/gradient dtype mismatch with dispatch IO_T");
  }

  return create_gpu_task(flow_id, cuda::cudnn_conv2d::run_backward_weights_and_bias, fe_handle,
                         stats, input->data(), gradient->data(), weight_gradients->data(),
                         use_bias_ ? bias_gradients->data() : nullptr, workspace->data());
}

void Conv2DLayer::cudnn_forward(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  const size_t batch_size = input->dimension(0);
  const size_t input_h = input->dimension(1);
  const size_t input_w = input->dimension(2);

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  output->ensure({batch_size, output_h, output_w, out_channels_}, this->device_);

  size_t shape_key = get_shape_hash(batch_size, in_channels_, input_h, input_w);

  cuda::cudnn_conv2d::feHandle_t *fe_handle = nullptr;
  size_t io_dtype_size = get_dtype_size(io_dtype_);

  if (fe_handle_cache.find(shape_key) == fe_handle_cache.end()) {
    ConvolutionStats new_stats;
    init_convolution_stats(new_stats, batch_size, in_channels_, input_h, input_w, out_channels_,
                           kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, use_bias_);

    auto cuda_context = dynamic_cast<CUDAContext *>(this->device_->context());
    if (!cuda_context) {
      throw std::runtime_error("Conv2DLayer requires CUDAContext for cuDNN operations");
    }
    cudnnHandle_t shared_handle = cuda_context->getCudnnHandle();
    auto io_data_type = cuda::cudnn_conv2d::get_cudnn_data_type(io_dtype_);
    auto compute_type = cuda::cudnn_conv2d::get_cudnn_data_type(compute_dtype_);
    fe_handle_cache[shape_key] = cuda::cudnn_conv2d::initialize_fe_handle(
        shared_handle, io_data_type, compute_type, new_stats);
    stats_cache[shape_key] = new_stats;
  }

  fe_handle = fe_handle_cache.at(shape_key);
  ConvolutionStats &current_stats = stats_cache.at(shape_key);

  size_t workspace_elements =
      (current_stats.fwd_workspace_size + io_dtype_size - 1) / io_dtype_size;
  Tensor cudnn_workspace = this->get_buffer({workspace_elements});

  if (this->is_training_) {
    micro_batch_inputs_cache_[micro_batch_id] = input;
  }

  DISPATCH_ON_3_DTYPES_TO_METHOD(conv2d_forward_task, fe_handle, current_stats, input, output,
                                 weights_, bias_, cudnn_workspace, batch_size, input_h, input_w,
                                 output_h, output_w, "default");
}

void Conv2DLayer::cudnn_backward(const Tensor &gradient, Tensor &grad_input,
                                 size_t micro_batch_id) {
  auto it_input_cache = micro_batch_inputs_cache_.find(micro_batch_id);
  if (it_input_cache == micro_batch_inputs_cache_.end()) {
    throw std::runtime_error("No cached input found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  const auto &input_shape = it_input_cache->second->shape();
  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[1];
  const size_t input_w = input_shape[2];
  const auto &grad_shape = gradient->shape();
  const size_t output_h = grad_shape[1];
  const size_t output_w = grad_shape[2];

  grad_input->ensure(input_shape, this->device_);

  size_t shape_key = get_shape_hash(batch_size, in_channels_, input_h, input_w);
  cuda::cudnn_conv2d::feHandle_t *fe_handle = fe_handle_cache.at(shape_key);
  ConvolutionStats &current_stats = stats_cache.at(shape_key);

  size_t io_dtype_size = get_dtype_size(io_dtype_);
  size_t max_backward_workspace =
      std::max({current_stats.wgrad_workspace_size, current_stats.dgrad_workspace_size,
                current_stats.bgrad_workspace_size});

  size_t workspace_elements = (max_backward_workspace + io_dtype_size - 1) / io_dtype_size;
  Tensor cudnn_workspace = this->get_buffer({workspace_elements});

  const Tensor &cached_input = it_input_cache->second;

  DISPATCH_ON_3_DTYPES_TO_METHOD(conv2d_backward_weights_and_bias_task, fe_handle, current_stats,
                                 cached_input, gradient, weight_gradients_, bias_gradients_,
                                 cudnn_workspace, batch_size, input_h, input_w, output_h, output_w,
                                 "default");

  DISPATCH_ON_3_DTYPES_TO_METHOD(conv2d_backward_data_task, fe_handle, current_stats, gradient,
                                 weights_, grad_input, cudnn_workspace, batch_size, input_h,
                                 input_w, output_h, output_w, "default");
}


LayerConfig Conv2DLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
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

std::unique_ptr<Layer> Conv2DLayer::clone() const {
  return std::make_unique<Conv2DLayer>(in_channels_, out_channels_, kernel_h_, kernel_w_, stride_h_,
                                       stride_w_, pad_h_, pad_w_, use_bias_, this->name_);
}

std::vector<size_t>
Conv2DLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("Conv2DLayer expects 4D input including batch size");
  }

  size_t batch_size = input_shape[0];
  size_t output_h = (input_shape[1] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[2] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {batch_size, output_h, output_w, out_channels_};
}

void Conv2DLayer::collect_parameters(std::vector<Tensor> &params) {
  params.push_back(weights_);
  if (use_bias_) {
    params.push_back(bias_);
  }
}

void Conv2DLayer::collect_gradients(std::vector<Tensor> &grads) {
  grads.push_back(weight_gradients_);
  if (use_bias_) {
    grads.push_back(bias_gradients_);
  }
}

uint64_t Conv2DLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[1];
  size_t input_w = input_shape[2];
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

uint64_t Conv2DLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  assert(input_shape.size() == 4 && "Input shape must be 4D");
  size_t batch_size = input_shape[0];
  size_t input_h = input_shape[1];
  size_t input_w = input_shape[2];
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

size_t Conv2DLayer::cached_memory_bytes() const {
  size_t total_bytes = 0;
  size_t input_cache_bytes = 0;
  for (const auto &pair : micro_batch_inputs_cache_) {
    input_cache_bytes += pair.second->size() * get_dtype_size(pair.second->data_type());
  }
  total_bytes += input_cache_bytes;
  return total_bytes;
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

} // namespace tnn
