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
  class_token_ = Tensor<T>({embed_dim_}, this->device_);
  class_token_gradients_ = Tensor<T>({embed_dim_}, this->device_);

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
  if (input.dims() != 3) {
    throw std::runtime_error(
        "ClassTokenLayer: Input tensor must have 3 dimensions (Batch, Seq, Embed)");
  }
  size_t batch_size = input.dimension(0);
  size_t seq_len = input.dimension(1);
  size_t embed_dim = input.dimension(2);

  if (embed_dim != embed_dim_) {
    throw std::runtime_error("ClassTokenLayer: Input embed_dim must match layer embed_dim");
  }

  output.ensure({batch_size, seq_len + 1, embed_dim}, this->device_);

  auto &out_ptr = output.data_ptr();
  const auto &in_ptr = input.data_ptr();
  auto &token_ptr = class_token_.data_ptr();

  if (this->device_->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::class_token_forward<T>, in_ptr.get(), token_ptr.get(),
                    out_ptr.get(), batch_size, seq_len, embed_dim);
  } else {
#ifdef USE_CUDA
    // Use create_gpu_task to ensure correct flow/stream usage, consistent with other layers
    create_gpu_task("default", cuda::class_token_forward<T>, in_ptr.get(), token_ptr.get(),
                    out_ptr.get(), batch_size, seq_len, embed_dim);
#else
    throw std::runtime_error("CUDA support for ClassTokenLayer not implemented");
#endif
  }
}

template <typename T>
void ClassTokenLayer<T>::backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                                       size_t micro_batch_id) {
  if (gradient.dims() != 3) {
    throw std::runtime_error(
        "ClassTokenLayer: Gradient tensor must have 3 dimensions (Batch, Seq, Embed)");
  }
  size_t batch_size = gradient.dimension(0);
  size_t seq_len_plus_1 = gradient.dimension(1);
  size_t embed_dim = gradient.dimension(2);
  size_t seq_len = seq_len_plus_1 - 1;

  grad_input.ensure({batch_size, seq_len, embed_dim}, this->device_);

  const auto &grad_ptr = gradient.data_ptr();
  auto &grad_in_ptr = grad_input.data_ptr();
  auto &token_grad_ptr = class_token_gradients_.data_ptr();

  if (this->device_->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::class_token_backward<T>, grad_ptr.get(), grad_in_ptr.get(),
                    token_grad_ptr.get(), batch_size, seq_len, embed_dim);
  } else {
#ifdef USE_CUDA
    create_gpu_task("default", cuda::class_token_backward<T>, grad_ptr.get(), grad_in_ptr.get(),
                    token_grad_ptr.get(), batch_size, seq_len, embed_dim);
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
  if (input_shape.size() < 3) {
    throw std::runtime_error("ClassTokenLayer: Input shape must have at least 3 dimensions");
  }
  size_t batch_size = input_shape[0];
  size_t seq_len = input_shape[1];
  size_t embed_dim = input_shape[2];
  return {batch_size, seq_len + 1, embed_dim};
}

template <typename T>
void ClassTokenLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.push_back(&class_token_);
}

template <typename T> void ClassTokenLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.push_back(&class_token_gradients_);
}

} // namespace tnn
