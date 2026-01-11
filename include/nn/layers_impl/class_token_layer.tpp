/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/class_token_layer.hpp"

#include "nn/layers_impl/cpu/class_token_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/class_token_ops.hpp"
#endif

#include <cmath>
#include <stdexcept>

namespace tnn {

template <typename T>
ClassTokenLayer<T>::ClassTokenLayer(size_t embed_dim, const std::string &name)
    : ParameterizedLayer<T>(name), embed_dim_(embed_dim) {}

template <typename T> void ClassTokenLayer<T>::init_params() {
  class_token_ = Tensor<T>({1, embed_dim_, 1, 1}, this->device_);
  class_token_gradients_ = Tensor<T>({1, embed_dim_, 1, 1}, this->device_);

  T fan_in = static_cast<T>(embed_dim_);
  T bound = static_cast<T>(1.0) / std::sqrt(fan_in);

  if (this->use_seed_) {
    class_token_.fill_random_uniform(-bound, bound, this->srand_seed_);
  } else {
    class_token_.fill_random_uniform(-bound, bound);
  }
  class_token_gradients_.fill(T(0));
}

template <typename T>
void ClassTokenLayer<T>::forward_impl(const Tensor<T> &input, Tensor<T> &output,
                                      size_t micro_batch_id) {
  if (!this->initialized_) {
    throw std::runtime_error("Layer parameters not initialized. Call initialize() before forward.");
  }

  const auto &shape = input.shape();
  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t height = shape[2];
  size_t width = shape[3];
  size_t len = height * width;

  if (channels != embed_dim_) {
    throw std::runtime_error("ClassTokenLayer: Input channels must match embed_dim");
  }

  output.ensure({batch_size, channels, len + 1, 1}, this->device_);

  auto &out_ptr = output.data_ptr();
  const auto &in_ptr = input.data_ptr();
  auto &token_ptr = class_token_.data_ptr();

  if (this->device_->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::class_token_forward<T>, in_ptr.get(), token_ptr.get(),
                    out_ptr.get(), batch_size, channels, len);
  } else {
#ifdef USE_CUDA
    // Use create_gpu_task to ensure correct flow/stream usage, consistent with other layers
    create_gpu_task("default", cuda::class_token_forward<T>, in_ptr.get(), token_ptr.get(),
                    out_ptr.get(), batch_size, channels, len);
#else
    throw std::runtime_error("CUDA support for ClassTokenLayer not implemented");
#endif
  }
}

template <typename T>
void ClassTokenLayer<T>::backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                                       size_t micro_batch_id) {
  const auto &shape = gradient.shape();
  size_t batch_size = shape[0];
  size_t channels = shape[1];
  size_t len_plus_1 = shape[2];
  size_t len = len_plus_1 - 1;

  grad_input.ensure({batch_size, channels, len, 1}, this->device_);

  const auto &grad_ptr = gradient.data_ptr();
  auto &grad_in_ptr = grad_input.data_ptr();
  auto &token_grad_ptr = class_token_gradients_.data_ptr();

  if (this->device_->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::class_token_backward<T>, grad_ptr.get(), grad_in_ptr.get(),
                    token_grad_ptr.get(), batch_size, channels, len);
  } else {
#ifdef USE_CUDA
    create_gpu_task("default", cuda::class_token_backward<T>, grad_ptr.get(), grad_in_ptr.get(),
                    token_grad_ptr.get(), batch_size, channels, len);
#else
    throw std::runtime_error("CUDA support for ClassTokenLayer not implemented");
#endif
  }
}

template <typename T>
uint64_t ClassTokenLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

template <typename T>
uint64_t ClassTokenLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  return 0;
}

template <typename T> std::string ClassTokenLayer<T>::type() const { return "class_token"; }

template <typename T> LayerConfig ClassTokenLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["embed_dim"] = embed_dim_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> ClassTokenLayer<T>::clone() const {
  return std::make_unique<ClassTokenLayer<T>>(embed_dim_, this->name_);
}

template <typename T>
std::vector<size_t>
ClassTokenLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("ClassTokenLayer expects 4D input including batch size");
  }

  // input_shape: [N, C, H, W]
  // output_shape: [N, C, H*W + 1, 1]
  size_t N = input_shape[0];
  size_t C = input_shape[1];
  size_t H = input_shape[2];
  size_t W = input_shape[3];
  return {N, C, H * W + 1, 1};
}

template <typename T>
void ClassTokenLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.push_back(&class_token_);
}

template <typename T> void ClassTokenLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.push_back(&class_token_gradients_);
}

} // namespace tnn
