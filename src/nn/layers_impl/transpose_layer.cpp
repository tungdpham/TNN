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

namespace synet {

TransposeLayerImpl::TransposeLayerImpl(const std::string &name)
    : SISOLayerImpl(name) {}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> TransposeLayerImpl::permute(const Tensor &input, Tensor &output, size_t B,
                                                  size_t L, size_t H, size_t D,
                                                  flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T>) {
    throw std::runtime_error(
        "TransposeLayerImpl mixed dtype dispatch not implemented (io/compute must match).");
  }
  if (input.data_type() != dtype_of<IO_T>() || output.data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("TransposeLayerImpl IO tensor dtype mismatch with dispatch IO_T");
  }

  if (get_engine_type() == EngineType::CPU) {
    return create_cpu_task(handle, cpu::permute_heads<Compute_T, Compute_T>,
                           input.data_as<Compute_T>(), output.data_as<Compute_T>(), B, L, H, D);
  }
#ifdef USE_CUDA
  else if (get_engine_type() == EngineType::CUDA) {
    return create_cuda_task(handle, cuda::permute_heads<Compute_T, Compute_T>,
                            input.data_as<Compute_T>(), output.data_as<Compute_T>(), B, L, H, D);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for permute_forward");
  }
  return nullptr;
}

Tensor TransposeLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  if (input.dims() != 3) {
    throw std::runtime_error("TransposeLayerImpl expects 3D input (Batch, D1, D2)");
  }
  size_t B = input.dimension(0);
  size_t L = input.dimension(1);
  size_t H = input.dimension(2);
  size_t D = 1;

  Tensor output = get_tensor({B, H, L}, input.data_type());

  DISPATCH_ON_3_DTYPES_TO_METHOD(permute, input, output, B, L, H, D, this->flow_handle_);
  return output;
}

Tensor TransposeLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  // Gradient is (B, H, L). We want (B, L, H).
  if (grad_output.dims() != 3) {
    throw std::runtime_error("TransposeLayerImpl: Gradient must be 3D");
  }
  size_t B = grad_output.dimension(0);
  // Gradient output shape was {B, H, L}, so dim(1) is H, dim(2) is L
  size_t H = grad_output.dimension(1);
  size_t L = grad_output.dimension(2);
  size_t D = 1;

  Tensor grad_input = get_tensor({B, L, H}, grad_output.data_type());

  DISPATCH_ON_3_DTYPES_TO_METHOD(permute, grad_output, grad_input, B, H, L, D, this->flow_handle_);
  return grad_input;
}

Vec<size_t> TransposeLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.size() != 3)
    throw std::runtime_error("TransposeLayerImpl expects 3 dims (B, D1, D2)");
  return {input_shape[0], input_shape[2], input_shape[1]};
}

LayerConfig TransposeLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

std::shared_ptr<TransposeLayerImpl> TransposeLayerImpl::create_from_config(
    const LayerConfig &config) {
  return std::make_shared<TransposeLayerImpl>(config.name);
}

}  // namespace synet
