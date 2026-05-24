/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/n_ary_layer.hpp"

#include "device/task.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/cpu/n_ary_ops.hpp"
#ifdef USE_CUDA
#include "nn/layers_impl/cuda/n_ary_ops.hpp"
#endif
#include <cmath>
#include <stdexcept>

namespace tnn {

Vec<Tensor> NAryOpLayer::forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) {
  if (inputs.size() < 2) {
    throw std::runtime_error("NAryOpLayer requires at least 2 inputs");
  }

  const auto &first_shape = inputs[0]->shape();
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i]->shape() != first_shape) {
      throw std::runtime_error("NAryOpLayer: all inputs must have the same shape");
    }
  }

  Tensor output = get_output_tensor(first_shape);

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto key = std::string("fwd_input_") + std::to_string(i);
    auto &cached = get_immutable_cache(mb_id, key);
    cached = inputs[i];
  }

  DISPATCH_IO_DTYPE(compute_nary_forward_impl, inputs, output, first_shape, this->flow_handle_);
  return {output};
}

Vec<Tensor> NAryOpLayer::backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("NAryOpLayer backward: expects 1 grad output");
  }

  const auto &output_shape = grad_outputs[0]->shape();

  Vec<ConstTensor> fwd_inputs;
  size_t num_inputs = 2;  // Default, will be determined from cache
  for (size_t i = 0;; ++i) {
    auto key = std::string("fwd_input_") + std::to_string(i);
    try {
      fwd_inputs.push_back(get_immutable_cache(mb_id, key));
      num_inputs = i + 1;
    } catch (...) {
      break;
    }
  }

  Vec<Tensor> grad_inputs;
  for (size_t i = 0; i < num_inputs; ++i) {
    Tensor grad_input = get_output_tensor(output_shape);
    grad_input->fill(0);
    grad_inputs.push_back(grad_input);
  }

  DISPATCH_IO_DTYPE(compute_nary_backward_impl, grad_outputs[0], grad_inputs, fwd_inputs,
                    output_shape, this->flow_handle_);
  return grad_inputs;
}

template <typename Compute_T>
std::unique_ptr<Task> NAryOpLayer::compute_nary_forward_impl(const Vec<ConstTensor> &inputs,
                                                             const Tensor &output,
                                                             const Vec<size_t> &shape,
                                                             flowHandle_t handle) {
  if (inputs[0]->data_type() != dtype_of<Compute_T>() ||
      output->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("NAryOpLayer: data type mismatch in forward pass");
  }

  Vec<const Compute_T *> input_ptrs;
  for (const auto &input : inputs) {
    input_ptrs.push_back(input->data_as<Compute_T>());
  }

  if (inputs[0]->device_type() == DeviceType::CPU) {
    cpu::nary::run_forward<Compute_T>(input_ptrs, output->data_as<Compute_T>(), shape, op_type_);
  }
#ifdef USE_CUDA
  else if (inputs[0]->device_type() == DeviceType::GPU) {
    size_t ws_bytes = cuda::nary::nary_forward_workspace_bytes(input_ptrs.size());
    Tensor ws = this->get_workspace({ws_bytes}, DType_t::BYTE);
    return create_cuda_task(handle, cuda::nary::run_forward<Compute_T>, input_ptrs,
                            output->data_as<Compute_T>(), shape, op_type_, ws->data());
  }
#endif
  else {
    throw std::runtime_error("NAryOpLayer: unsupported device type");
  }

  return nullptr;
}

template <typename Compute_T>
std::unique_ptr<Task> NAryOpLayer::compute_nary_backward_impl(const ConstTensor &grad_output,
                                                              const Vec<Tensor> &grad_inputs,
                                                              const Vec<ConstTensor> &fwd_inputs,
                                                              const Vec<size_t> &shape,
                                                              flowHandle_t handle) {
  if (grad_output->data_type() != dtype_of<Compute_T>()) {
    throw std::runtime_error("NAryOpLayer: data type mismatch in backward pass");
  }

  Vec<const Compute_T *> fwd_input_ptrs;
  for (const auto &input : fwd_inputs) {
    fwd_input_ptrs.push_back(input->data_as<Compute_T>());
  }

  Vec<Compute_T *> grad_input_ptrs;
  for (auto &grad_input : grad_inputs) {
    grad_input_ptrs.push_back(grad_input->data_as<Compute_T>());
  }

  if (grad_output->device_type() == DeviceType::CPU) {
    cpu::nary::run_backward<Compute_T>(grad_output->data_as<Compute_T>(), grad_input_ptrs,
                                       fwd_input_ptrs, shape, op_type_);
  }
#ifdef USE_CUDA
  else if (grad_output->device_type() == DeviceType::GPU) {
    size_t ws_bytes = cuda::nary::nary_backward_workspace_bytes(fwd_input_ptrs.size());
    Tensor ws = this->get_workspace({ws_bytes}, DType_t::BYTE);
    return create_cuda_task(handle, cuda::nary::run_backward<Compute_T>,
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
