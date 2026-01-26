/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/transpose_layer.hpp"
#include "nn/blocks_impl/cpu/permute_heads.hpp"
#ifdef USE_CUDA
#include "nn/blocks_impl/cuda/permute_heads.hpp"
#endif

namespace tnn {

TransposeLayer::TransposeLayer(const std::string &name) : StatelessLayer(name) {}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> TransposeLayer::permute(const Tensor &input, Tensor &output, size_t B,
                                              size_t L, size_t H, size_t D,
                                              const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T>) {
    throw std::runtime_error(
        "TransposeLayer mixed dtype dispatch not implemented (io/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("TransposeLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  if (this->device_->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::permute_heads<Compute_T>, input->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), B, L, H, D);
  }
#ifdef USE_CUDA
  else if (this->device_->device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::permute_heads<Compute_T>, input->data_as<Compute_T>(),
                           output->data_as<Compute_T>(), B, L, H, D);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for permute_forward");
  }
  return nullptr;
}

void TransposeLayer::forward_impl(const Tensor &input, Tensor &output, size_t mb_id) {
  if (input->dims() != 3) {
    throw std::runtime_error("TransposeLayer expects 3D input (Batch, D1, D2)");
  }
  size_t B = input->dimension(0);
  size_t L = input->dimension(1);
  size_t H = input->dimension(2);
  size_t D = 1;

  output->ensure({B, H, L});

  DISPATCH_ON_3_DTYPES_TO_METHOD(permute, input, output, B, L, H, D, "default");
}

void TransposeLayer::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id) {
  // Gradient is (B, H, L). We want (B, L, H).
  if (gradient->dims() != 3) {
    throw std::runtime_error("TransposeLayer: Gradient must be 3D");
  }
  size_t B = gradient->dimension(0);
  // Gradient output shape was {B, H, L}, so dim(1) is H, dim(2) is L
  size_t H = gradient->dimension(1);
  size_t L = gradient->dimension(2);
  size_t D = 1;

  grad_input->ensure({B, L, H});

  DISPATCH_ON_3_DTYPES_TO_METHOD(permute, gradient, grad_input, B, H, L, D, "default");
}

std::vector<size_t>
TransposeLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 3)
    throw std::runtime_error("TransposeLayer expects 3 dims (B, D1, D2)");
  return {input_shape[0], input_shape[2], input_shape[1]};
}

LayerConfig TransposeLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::unique_ptr<Layer> TransposeLayer::clone() const {
  return std::make_unique<TransposeLayer>(this->name_);
}

std::unique_ptr<TransposeLayer> TransposeLayer::create_from_config(const LayerConfig &config) {
  return std::make_unique<TransposeLayer>(config.name);
}

} // namespace tnn
