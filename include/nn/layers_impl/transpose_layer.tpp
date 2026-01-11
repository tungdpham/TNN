/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/blocks_impl/cpu/permute_heads.hpp"
#include "nn/layers_impl/transpose_layer.hpp"
#ifdef USE_CUDA
#include "nn/blocks_impl/cuda/permute_heads.hpp"
#endif

namespace tnn {

template <typename T>
TransposeLayer<T>::TransposeLayer(const std::string &name) : StatelessLayer<T>(name) {}

template <typename T>
void TransposeLayer<T>::forward_impl(const Tensor<T> &input, Tensor<T> &output,
                                     size_t micro_batch_id) {
  if (input.dims() != 3) {
    throw std::runtime_error("TransposeLayer expects 3D input (Batch, D1, D2)");
  }
  size_t B = input.dimension(0);
  size_t L = input.dimension(1);
  size_t H = input.dimension(2);
  size_t D = 1;

  output.ensure({B, H, L}, this->device_);

  // Call permute_heads(input, output, B, L, H, D)
  if (this->device_->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::permute_heads<T>, input.data_ptr().get(),
                    output.data_ptr().get(), B, L, H, D);
  }
#ifdef USE_CUDA
  else {
    // Use create_gpu_task to ensure correct flow/stream usage
    create_gpu_task("default", cuda::permute_heads<T>, input.data_ptr().get(),
                    output.data_ptr().get(), B, L, H, D);
  }
#endif
}

template <typename T>
void TransposeLayer<T>::backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                                      size_t micro_batch_id) {
  // Gradient is (B, H, L). We want (B, L, H).
  if (gradient.dims() != 3) {
    throw std::runtime_error("TransposeLayer: Gradient must be 3D");
  }
  size_t B = gradient.dimension(0);
  // Gradient output shape was {B, H, L}, so dim(1) is H, dim(2) is L
  size_t H = gradient.dimension(1);
  size_t L = gradient.dimension(2);
  size_t D = 1;

  grad_input.ensure({B, L, H}, this->device_);

  if (this->device_->device_type() == DeviceType::CPU) {
    create_cpu_task("default", cpu::permute_heads<T>, gradient.data_ptr().get(),
                    grad_input.data_ptr().get(), B, H, L, D);
  }
#ifdef USE_CUDA
  else {
    create_gpu_task("default", cuda::permute_heads<T>, gradient.data_ptr().get(),
                    grad_input.data_ptr().get(), B, H, L, D);
  }
#endif
}

template <typename T>
std::vector<size_t>
TransposeLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 3)
    throw std::runtime_error("TransposeLayer expects 3 dims (B, D1, D2)");
  return {input_shape[0], input_shape[2], input_shape[1]};
}

template <typename T> std::string TransposeLayer<T>::type() const { return "transpose"; }

template <typename T> LayerConfig TransposeLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> TransposeLayer<T>::clone() const {
  return std::make_unique<TransposeLayer<T>>(this->name_);
}

template <typename T>
std::unique_ptr<Layer<T>> TransposeLayer<T>::create_from_config(const LayerConfig &config) {
  return std::make_unique<TransposeLayer<T>>(config.name);
}

} // namespace tnn
