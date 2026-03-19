/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/n_ary_layer.hpp"

#include "device/task.hpp"
#include "nn/layers_impl/cpu/n_ary_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/n_ary_ops.hpp"
#endif
#include <cmath>
#include <stdexcept>

namespace tnn {

void NAryOpLayer::forward_impl(const std::vector<ConstTensor> &inputs,
                               const std::vector<Tensor> &outputs, size_t mb_id) {
  if (inputs.size() < 2) {
    throw std::runtime_error("NAryOpLayer requires at least 2 inputs");
  }
  if (outputs.size() != 1) {
    throw std::runtime_error("NAryOpLayer produces exactly 1 output");
  }

  const auto &first_shape = inputs[0]->shape();
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i]->shape() != first_shape) {
      throw std::runtime_error("NAryOpLayer: all inputs must have the same shape");
    }
  }

  outputs[0]->ensure(first_shape);

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto key = std::string("fwd_input_") + std::to_string(i);
    auto &cached = get_cached_tensor(mb_id, key);
    cached = inputs[i];
  }

  DISPATCH_IO_DTYPE(compute_nary_forward_impl, inputs, outputs[0], first_shape, this->flow_handle_);
}

void NAryOpLayer::backward_impl(const std::vector<ConstTensor> &grad_outputs,
                                const std::vector<Tensor> &grad_inputs, size_t mb_id) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("NAryOpLayer backward: expects 1 grad output");
  }
  if (grad_inputs.size() < 2) {
    throw std::runtime_error("NAryOpLayer backward: requires at least 2 grad inputs");
  }

  const auto &output_shape = grad_outputs[0]->shape();

  for (auto &grad_input : grad_inputs) {
    grad_input->ensure(output_shape);
    grad_input->fill(0);
  }

  std::vector<ConstTensor> fwd_inputs;
  for (size_t i = 0; i < grad_inputs.size(); ++i) {
    auto key = std::string("fwd_input_") + std::to_string(i);
    fwd_inputs.push_back(get_cached_tensor(mb_id, key));
  }

  DISPATCH_IO_DTYPE(compute_nary_backward_impl, grad_outputs[0], grad_inputs, fwd_inputs,
                    output_shape, this->flow_handle_);
}

template <typename Compute_T>
std::unique_ptr<Task> NAryOpLayer::compute_nary_forward_impl(const std::vector<ConstTensor> &inputs,
                                                             const Tensor &output,
                                                             const std::vector<size_t> &shape,
                                                             flowHandle_t handle) {
  if (inputs[0]->data_type() != dtype_of<Compute_T>() ||
      output->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("NAryOpLayer: data type mismatch in forward pass");
  }

  std::vector<const Compute_T *> input_ptrs;
  for (const auto &input : inputs) {
    input_ptrs.push_back(input->data_as<Compute_T>());
  }

  if (inputs[0]->device_type() == DeviceType::CPU) {
    cpu::nary_forward<Compute_T>(input_ptrs, output->data_as<Compute_T>(), shape, op_type_);
  }
#ifdef USE_CUDA
  else if (inputs[0]->device_type() == DeviceType::GPU) {
    size_t ws_bytes = cuda::nary_forward_workspace_bytes(input_ptrs.size());
    Tensor ws = this->get_workspace({ws_bytes}, DType_t::BYTE);
    return create_cuda_task(handle, cuda::nary_forward<Compute_T>, input_ptrs,
                            output->data_as<Compute_T>(), shape, op_type_, ws->data());
  }
#endif
  else {
    throw std::runtime_error("NAryOpLayer: unsupported device type");
  }

  return nullptr;
}

template <typename Compute_T>
std::unique_ptr<Task> NAryOpLayer::compute_nary_backward_impl(
    const ConstTensor &grad_output, const std::vector<Tensor> &grad_inputs,
    const std::vector<ConstTensor> &fwd_inputs, const std::vector<size_t> &shape,
    flowHandle_t handle) {
  if (grad_output->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("NAryOpLayer: data type mismatch in backward pass");
  }

  std::vector<const Compute_T *> fwd_input_ptrs;
  for (const auto &input : fwd_inputs) {
    fwd_input_ptrs.push_back(input->data_as<Compute_T>());
  }

  std::vector<Compute_T *> grad_input_ptrs;
  for (auto &grad_input : grad_inputs) {
    grad_input_ptrs.push_back(grad_input->data_as<Compute_T>());
  }

  if (grad_output->device_type() == DeviceType::CPU) {
    cpu::nary_backward<Compute_T>(grad_output->data_as<Compute_T>(), grad_input_ptrs,
                                  fwd_input_ptrs, shape, op_type_);
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    size_t ws_bytes = cuda::nary_backward_workspace_bytes(fwd_input_ptrs.size());
    Tensor ws = this->get_workspace({ws_bytes}, DType_t::BYTE);
    return create_cuda_task(handle, cuda::nary_backward<Compute_T>,
                            grad_output->data_as<Compute_T>(), grad_input_ptrs, fwd_input_ptrs,
                            shape, op_type_, ws->data());
  }
#endif
  else {
    throw std::runtime_error("NAryOpLayer: unsupported device type");
  }

  return nullptr;
}

Vec<Vec<size_t>> NAryOpLayer::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (input_shapes.size() < 2) {
    throw std::runtime_error("NAryOpLayer: requires at least 2 inputs");
  }

  for (size_t i = 1; i < input_shapes.size(); ++i) {
    if (input_shapes[i] != input_shapes[0]) {
      throw std::runtime_error("NAryOpLayer: all inputs must have the same shape");
    }
  }

  return {input_shapes[0]};
}

LayerConfig NAryOpLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  return config;
}

size_t NAryOpLayer::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
#ifdef USE_CUDA
  if (allocator_ && allocator_->device().device_type() == DeviceType::GPU) {
    return cuda::nary_forward_workspace_bytes(input_shapes.size());
  }
#endif
  return 0;
}

size_t NAryOpLayer::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  return fwd_workspace(input_shapes);
}

size_t NAryOpLayer::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
#ifdef USE_CUDA
  if (allocator_ && allocator_->device().device_type() == DeviceType::GPU) {
    return cuda::nary_backward_workspace_bytes(input_shapes.size());
  }
#endif
  return 0;
}

std::unique_ptr<AddLayer> AddLayer::create_from_config(const LayerConfig &config) {
  return std::make_unique<AddLayer>(config.name);
}

std::unique_ptr<SubLayer> SubLayer::create_from_config(const LayerConfig &config) {
  return std::make_unique<SubLayer>(config.name);
}

std::unique_ptr<MulLayer> MulLayer::create_from_config(const LayerConfig &config) {
  return std::make_unique<MulLayer>(config.name);
}

std::unique_ptr<DivLayer> DivLayer::create_from_config(const LayerConfig &config) {
  return std::make_unique<DivLayer>(config.name);
}

}  // namespace tnn
