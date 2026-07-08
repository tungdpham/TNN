/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layer.hpp"

#include <fmt/ranges.h>

#include "device/flow.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_ops.hpp"
#include "type/type.hpp"

namespace tunx {

void LayerImpl::set_engine_type(EngineType engine_type) {
  engine_type_ = engine_type;
  on_set_engine_type(engine_type);
}

EngineType LayerImpl::get_engine_type() const { return engine_type_; }

void LayerImpl::init() {
  if (initialized_) {
    throw std::runtime_error("Cannot initalize LayerImpl more than once. ");
  }
  if (engine_type_ == EngineType::UNKNOWN) {
    throw std::runtime_error(
        "Engine type must be set to a valid value before initializing LayerImpl.");
  }
  init_impl();
  initialized_ = true;
}

Vec<Tensor> LayerImpl::forward(const Vec<Tensor> &inputs) {
  Residuals dummy_residuals;
  return forward(inputs, dummy_residuals);
}

Vec<Tensor> LayerImpl::forward(const Vec<Tensor> &inputs, Residuals &residuals) {
  if (!initialized_) {
    throw std::runtime_error("LayerImpl must be initialized before calling forward");
  }
  is_fwd_ = true;
  Vec<Tensor> current_inputs;
  for (auto &input : inputs) {
    if (input.device() == this->device())
      current_inputs.push_back(input);
    else
      current_inputs.push_back(input.to_device(this->device()));
  }
  Vec<Tensor> outputs = forward_impl(current_inputs, residuals);
#ifndef NDEBUG
  this->device().getFlow(flow_handle_)->synchronize();
#endif
  return outputs;
}

Vec<Tensor> LayerImpl::backward(const Vec<Tensor> &grad_outputs, Residuals &residuals) {
  if (!initialized_) {
    throw std::runtime_error("LayerImpl must be initialized before calling backward");
  }
  is_fwd_ = false;
  Vec<Tensor> current_grad_outputs;
  for (auto &grad : grad_outputs) {
    if (grad.device() == this->device())
      current_grad_outputs.push_back(grad);
    else
      current_grad_outputs.push_back(grad.to_device(this->device()));
  }
  auto grad_inputs = backward_impl(current_grad_outputs, residuals);
#ifndef NDEBUG
  this->device().getFlow(flow_handle_)->synchronize();
#endif
  return grad_inputs;
}

LayerImpl &LayerImpl::set_allocator(DELAllocatorV2 &allocator) {
  allocator_ = &allocator;
  on_set_allocator(allocator);
  return *this;
}

DELAllocatorV2 *LayerImpl::get_allocator() const { return allocator_; }

LayerImpl &LayerImpl::set_flow_handle(flowHandle_t handle) {
  flow_handle_ = handle;
  on_set_flow_handle(handle);
  return *this;
}

flowHandle_t LayerImpl::get_flow_handle() const { return flow_handle_; }

LayerImpl &LayerImpl::set_io_dtype(DType_t dtype) {
  io_dtype_ = dtype;
  on_set_io_dtype(dtype);
  return *this;
}

DType_t LayerImpl::get_io_dtype() const { return io_dtype_; }

LayerImpl &LayerImpl::set_param_dtype(DType_t dtype) {
  param_dtype_ = dtype;
  on_set_param_dtype(dtype);
  return *this;
}

DType_t LayerImpl::get_param_dtype() const { return param_dtype_; }

LayerImpl &LayerImpl::set_compute_dtype(DType_t dtype) {
  compute_dtype_ = dtype;
  on_set_compute_dtype(dtype);
  return *this;
}

DType_t LayerImpl::get_compute_dtype() const { return compute_dtype_; }

LayerImpl &LayerImpl::set_seed(unsigned long long seed) {
  use_seed_ = true;
  srand_seed_ = seed;
  on_set_seed(seed);
  return *this;
}

LayerImpl &LayerImpl::set_training(bool training) {
  is_training_ = training;
  on_set_training(training);
  return *this;
}

bool LayerImpl::is_training() const { return is_training_; }

LayerImpl &LayerImpl::set_engine(Engine engine) {
  engine_ = engine;
  on_set_engine(engine);
  return *this;
}

Engine LayerImpl::get_engine() {
  if (!engine_) {
    throw std::runtime_error("Engine is not set");
  }
  return engine_;
}

void LayerImpl::set_backend_handle(void *backend_handle) {
  backend_handle_ = backend_handle;
  on_set_backend_handle(backend_handle);
}

void *LayerImpl::get_backend_handle() const { return backend_handle_; }

void LayerImpl::save_state(std::ostream &out) const {
  auto config = get_config();
  nlohmann::json j = config.to_json();
  std::string j_str = j.dump();
  size_t j_size = j_str.size();
  out.write(reinterpret_cast<const char *>(&j_size), sizeof(size_t));
  out.write(j_str.c_str(), j_size);
  auto descs = param_descriptors();
  for (const auto &desc : descs) {
    Tensor param = *desc.data_ptr;
    save(param, out);
  }
}

Tensor LayerImpl::get_tensor(const Vec<size_t> &shape, DType_t dtype) {
  if (!allocator_) {
    throw std::runtime_error("Allocator is not set");
  }
  return Tensor(shape, dtype, *allocator_);
}

}  // namespace tunx
