/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/slice_layer.hpp"
#include "device/task.hpp"
#include "nn/layers_impl/cpu/slice_ops.hpp"
#include "nn/layers_impl/cuda/slice_ops.hpp"

#include <stdexcept>

namespace tnn {

SliceLayer::SliceLayer(size_t axis, size_t start, size_t length, const std::string &name)
    : StatelessLayer(name), axis_(axis), start_(start), length_(length) {}

SliceLayer::~SliceLayer() = default;

void SliceLayer::forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  micro_batch_original_shapes_[micro_batch_id] = input->shape();

  std::vector<size_t> output_shape = compute_output_shape(input->shape());
  output->ensure(output_shape, this->device_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(slice_forward, input, output, "default");
}

void SliceLayer::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t micro_batch_id) {
  auto it = micro_batch_original_shapes_.find(micro_batch_id);
  if (it == micro_batch_original_shapes_.end()) {
    throw std::runtime_error("No cached shape found for micro-batch ID in SliceLayer");
  }
  const std::vector<size_t> &original_shape = it->second;

  grad_input->ensure(original_shape, this->device_);

  DISPATCH_ON_3_DTYPES_TO_METHOD(slice_backward, gradient, grad_input, original_shape, "default");
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> SliceLayer::slice_forward(const Tensor &input, Tensor &output,
                                                const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T>) {
    throw std::runtime_error(
        "SliceLayer mixed dtype dispatch not implemented (io/compute must match).");
  }
  if (input->data_type() != dtype_of<IO_T>() || output->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("SliceLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  if (input->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::slice::slice_forward<Compute_T>,
                           input->data_as<Compute_T>(), output->data_as<Compute_T>(),
                           input->shape(), axis_, start_, length_);
  }
#ifdef USE_CUDA
  else if (input->device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::slice::slice_forward<Compute_T>,
                           input->data_as<Compute_T>(), output->data_as<Compute_T>(),
                           input->shape(), axis_, start_, length_);
  }
#endif
  else {
    if (input->device_type() == DeviceType::GPU) {
      throw std::runtime_error("SliceLayer: GPU execution requires building with USE_CUDA");
    }
    throw std::runtime_error("SliceLayer: Unsupported device type");
  }
  return nullptr;
}

template <typename IO_T, typename Param_T, typename Compute_T>
std::unique_ptr<Task> SliceLayer::slice_backward(const Tensor &gradient, Tensor &grad_input,
                                                 const std::vector<size_t> &original_shape,
                                                 const std::string &flow_id) const {
  if constexpr (!std::is_same_v<IO_T, Compute_T>) {
    throw std::runtime_error(
        "SliceLayer mixed dtype dispatch not implemented (io/compute must match).");
  }
  if (gradient->data_type() != dtype_of<IO_T>() || grad_input->data_type() != dtype_of<IO_T>()) {
    throw std::runtime_error("SliceLayer IO tensor dtype mismatch with dispatch IO_T");
  }

  if (gradient->device_type() == DeviceType::CPU) {
    return create_cpu_task(flow_id, cpu::slice::slice_backward<Compute_T>,
                           gradient->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                           original_shape, axis_, start_, length_);
  }
#ifdef USE_CUDA
  else if (gradient->device_type() == DeviceType::GPU) {
    return create_gpu_task(flow_id, cuda::slice::slice_backward<Compute_T>,
                           gradient->data_as<Compute_T>(), grad_input->data_as<Compute_T>(),
                           original_shape, axis_, start_, length_);
  }
#endif
  else {
    if (gradient->device_type() == DeviceType::GPU) {
      throw std::runtime_error("SliceLayer: GPU execution requires building with USE_CUDA");
    }
    throw std::runtime_error("SliceLayer: Unsupported device type");
  }
  return nullptr;
}

std::vector<size_t> SliceLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (axis_ >= input_shape.size()) {
    throw std::invalid_argument("Slice axis out of bounds");
  }
  if (start_ + length_ > input_shape[axis_]) {
    throw std::invalid_argument("Slice range out of bounds");
  }

  std::vector<size_t> output_shape = input_shape;
  output_shape[axis_] = length_;
  return output_shape;
}


LayerConfig SliceLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["axis"] = (int)axis_;
  config.parameters["start"] = (int)start_;
  config.parameters["length"] = (int)length_;
  return config;
}

std::unique_ptr<Layer> SliceLayer::clone() const {
  return std::make_unique<SliceLayer>(axis_, start_, length_, this->name_);
}

std::unique_ptr<SliceLayer> SliceLayer::create_from_config(const LayerConfig &config) {
  size_t axis = (size_t)config.get<int>("axis", 0);
  size_t start = (size_t)config.get<int>("start", 0);
  size_t length = (size_t)config.get<int>("length", 1);
  return std::make_unique<SliceLayer>(axis, start, length, config.name);
}

uint64_t SliceLayer::forward_flops(const std::vector<size_t> &input_shape) const { return 0; }

uint64_t SliceLayer::backward_flops(const std::vector<size_t> &input_shape) const { return 0; }

} // namespace tnn
