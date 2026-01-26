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
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/dropout_ops.hpp"
#endif

namespace tnn {

DropoutLayer::DropoutLayer(float dropout_rate, const std::string &name)
    : StatelessLayer(name), dropout_rate_(dropout_rate), generator_(std::random_device{}()) {
  if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
    throw std::invalid_argument("Dropout rate must be in [0, 1)");
  }
}

void DropoutLayer::forward_impl(const Tensor &input, Tensor &output, size_t mb_id) {
  if (!this->is_training_) {
    output->ensure(input->shape());
    input->copy_to(output);
    return;
  }

  Tensor &mask = micro_batch_masks_[mb_id];
  if (mask == nullptr)
    mask = Tensor::create<float>(input->shape(), this->device_);
  else {
    mask->ensure(input->shape());
  }
  output->ensure(input->shape());

  DISPATCH_ON_3_DTYPES_TO_METHOD(compute_dropout_forward, input, output, mask, "default");
}

void DropoutLayer::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id) {
  if (!this->is_training_) {
    grad_input->ensure(gradient->shape());
    gradient->copy_to(grad_input);
    return;
  }

  auto it_mask = micro_batch_masks_.find(mb_id);
  if (it_mask == micro_batch_masks_.end()) {
    throw std::runtime_error("No cached mask found for micro-batch ID in DropoutLayer: " +
                             std::to_string(mb_id));
  }
  const Tensor &mask = it_mask->second;

  grad_input->ensure(gradient->shape());
  gradient->copy_to(grad_input);
  grad_input->mul(mask);
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> DropoutLayer::compute_dropout_forward(const Tensor &input, Tensor &output,
                                                            Tensor &mask,
                                                            const std::string &flow_id) const {
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
    return create_cpu_task(flow_id, cpu::dropout::compute_dropout_forward<Compute_T>,
                           input->data_as<Compute_T>(), output->data_as<Compute_T>(),
                           mask->data_as<Compute_T>(), batch_size, channels, spatial_size,
                           dropout_rate_);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::dropout::compute_dropout_forward<Compute_T>,
                           input->data_as<Compute_T>(), output->data_as<Compute_T>(),
                           mask->data_as<Compute_T>(), batch_size, channels, spatial_size,
                           dropout_rate_);
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for compute_dropout_forward");
  }
  return nullptr;
}

LayerConfig DropoutLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["dropout_rate"] = dropout_rate_;
  return config;
}

std::unique_ptr<Layer> DropoutLayer::clone() const {
  return std::make_unique<DropoutLayer>(dropout_rate_, this->name_);
}

std::vector<size_t>
DropoutLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

std::unique_ptr<DropoutLayer> DropoutLayer::create_from_config(const LayerConfig &config) {
  float dropout_rate = config.get<float>("dropout_rate");
  return std::make_unique<DropoutLayer>(dropout_rate, config.name);
}

uint64_t DropoutLayer::forward_flops(const std::vector<size_t> &input_shape) const {
  if (!this->is_training_) {
    return 0;
  }

  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  uint64_t rng_flops = num_elements;
  uint64_t mask_flops = num_elements;
  uint64_t scale_flops =
      static_cast<uint64_t>((1.0 - static_cast<double>(dropout_rate_)) * num_elements);

  return rng_flops + mask_flops + scale_flops;
}

uint64_t DropoutLayer::backward_flops(const std::vector<size_t> &input_shape) const {
  if (!this->is_training_) {
    return 0;
  }

  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  return num_elements;
}

} // namespace tnn
