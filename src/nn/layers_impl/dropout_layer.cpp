/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/dropout_layer.hpp"

#include <memory>
#include <stdexcept>

#include "device/task.hpp"
#include "nn/layers_impl/cpu/dropout_ops.hpp"
#include "type/type.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/dropout_ops.hpp"
#endif

namespace tnn {

DropoutLayer::DropoutLayer(float dropout_rate, const std::string &name)
    : StatelessLayer(name),
      dropout_rate_(dropout_rate),
      generator_(std::random_device{}()) {
  if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
    throw std::invalid_argument("Dropout rate must be in [0, 1)");
  }
}

Tensor DropoutLayer::forward_impl(const ConstTensor &input, size_t mb_id) {
  if (!this->is_training_) {
    Tensor output = get_output_tensor(input->shape());
    output->share_from(input);
    return output;
  }

  Tensor mask = this->get_cache_tensor(input->shape(), DType_t::BOOL);
  set_mutable_cache(mb_id, "mask", mask);

  Tensor output = get_output_tensor(input->shape());

  DISPATCH_ON_3_DTYPES_TO_METHOD(run_forward, input, output, mask, this->flow_handle_);
  return output;
}

Tensor DropoutLayer::backward_impl(const ConstTensor &grad_output, size_t mb_id) {
  const ConstTensor &mask = this->get_mutable_cache(mb_id, "mask");
  if (mask == nullptr) {
    throw std::runtime_error("No cached mask found for micro-batch ID in DropoutLayer: " +
                             std::to_string(mb_id));
  }

  Tensor grad_input = get_output_tensor(grad_output->shape());
  DISPATCH_ON_3_DTYPES_TO_METHOD(run_backward, grad_output, grad_input, mask, this->flow_handle_);
  return grad_input;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> DropoutLayer::run_forward(const ConstTensor &input, const Tensor &output,
                                                const Tensor &mask, flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T>) {
    throw std::runtime_error(
        "DropoutLayer mixed dtype dispatch not implemented (io/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("DropoutLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  size_t batch_size = input->dimension(0);
  size_t channels = input->dimension(1);
  size_t spatial_size = input->stride(1);

  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::dropout::run_forward<Compute_T>,
                           input->data_as<Compute_T>(), output->data_as<Compute_T>(),
                           mask->data_as<bool>(), batch_size, channels, spatial_size,
                           dropout_rate_);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::dropout::run_forward<Compute_T>,
                            input->data_as<Compute_T>(), output->data_as<Compute_T>(),
                            mask->data_as<bool>(), batch_size, channels, spatial_size,
                            dropout_rate_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for run_forward");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> DropoutLayer::run_backward(const ConstTensor &grad_output,
                                                 const Tensor &grad_input, const ConstTensor &mask,
                                                 flowHandle_t handle) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T>) {
    throw std::runtime_error(
        "DropoutLayer mixed dtype dispatch not implemented (io/compute must match).");
  }

  size_t batch_size = grad_output->dimension(0);
  size_t channels = grad_output->dimension(1);
  size_t spatial_size = grad_output->stride(1);

  Compute_T scale = Compute_T(1) / (Compute_T(1) - static_cast<Compute_T>(dropout_rate_));

  if (grad_output->device_type() == DeviceType::CPU) {
    return create_cpu_task(handle, cpu::dropout::run_backward<Compute_T>,
                           grad_output->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                           mask->data_as<bool>(), batch_size, channels, spatial_size, scale);
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    return create_cuda_task(handle, cuda::dropout::run_backward<Compute_T>,
                            grad_output->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                            mask->data_as<bool>(), batch_size, channels, spatial_size, scale);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for run_backward");
  }
  return nullptr;
}

LayerConfig DropoutLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("dropout_rate", dropout_rate_);
  return config;
}

Vec<size_t> DropoutLayer::compute_output_shape(const Vec<size_t> &input_shape) const {
  return input_shape;
}

std::unique_ptr<DropoutLayer> DropoutLayer::create_from_config(const LayerConfig &config) {
  float dropout_rate = config.get<float>("dropout_rate");
  return std::make_unique<DropoutLayer>(dropout_rate, config.name);
}

}  // namespace tnn
