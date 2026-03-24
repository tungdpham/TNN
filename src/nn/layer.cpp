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
#include "type/type.hpp"

namespace tnn {

void Layer::init() {
  if (initialized_) {
    throw std::runtime_error("Cannot initalize Layer more than once. ");
  }
  init_impl();
  initialized_ = true;
}

Vec<Tensor> Layer::forward(const Vec<ConstTensor> &inputs, size_t mb_id) {
  if (!initialized_) {
    throw std::runtime_error("Layer must be initialized before calling forward");
  }
  is_fwd_ = true;
  Vec<ConstTensor> current_inputs;
  for (auto &input : inputs) {
    if (input->device() == this->device())
      current_inputs.push_back(input);
    else
      current_inputs.push_back(input->to_device(this->device()));
  }
  return forward_impl(current_inputs, mb_id);
}

Vec<Tensor> Layer::backward(const Vec<ConstTensor> &grad_outputs, size_t mb_id) {
  if (!initialized_) {
    throw std::runtime_error("Layer must be initialized before calling backward");
  }
  is_fwd_ = false;
  Vec<ConstTensor> current_grad_outputs;
  for (auto &grad : grad_outputs) {
    if (grad->device() == this->device())
      current_grad_outputs.push_back(grad);
    else
      current_grad_outputs.push_back(grad->to_device(this->device()));
  }
  auto grad_inputs = backward_impl(current_grad_outputs, mb_id);
  clear_cache(mb_id);
  return grad_inputs;
}

Layer &Layer::set_allocator(DELAllocatorV2 &allocator) {
  allocator_ = &allocator;
  on_set_allocator(allocator);
  return *this;
}

DELAllocatorV2 *Layer::get_allocator() const { return allocator_; }

Layer &Layer::set_flow_handle(flowHandle_t handle) {
  flow_handle_ = handle;
  on_set_flow_handle(handle);
  return *this;
}

flowHandle_t Layer::get_flow_handle() const { return flow_handle_; }

Layer &Layer::set_io_dtype(DType_t dtype) {
  io_dtype_ = dtype;
  on_set_io_dtype(dtype);
  return *this;
}

DType_t Layer::get_io_dtype() const { return io_dtype_; }

Layer &Layer::set_param_dtype(DType_t dtype) {
  param_dtype_ = dtype;
  on_set_param_dtype(dtype);
  return *this;
}

DType_t Layer::get_param_dtype() const { return param_dtype_; }

Layer &Layer::set_compute_dtype(DType_t dtype) {
  compute_dtype_ = dtype;
  on_set_compute_dtype(dtype);
  return *this;
}

DType_t Layer::get_compute_dtype() const { return compute_dtype_; }

Layer &Layer::set_seed(unsigned long long seed) {
  use_seed_ = true;
  srand_seed_ = seed;
  on_set_seed(seed);
  return *this;
}

Layer &Layer::set_training(bool training) {
  is_training_ = training;
  on_set_training(training);
  return *this;
}

bool Layer::is_training() const { return is_training_; }

void Layer::save_state(std::ofstream &file) {
  auto config = get_config();
  nlohmann::json j = config.to_json();
  std::string j_str = j.dump();
  size_t j_size = j_str.size();
  file.write(reinterpret_cast<const char *>(&j_size), sizeof(size_t));
  file.write(j_str.c_str(), j_size);
  auto descs = param_descriptors();
  for (const auto &desc : descs) {
    Tensor param = *desc.data_ptr;
    param->save(file);
  }
}

Vec<Tensor> Layer::parameters() {
  Vec<Tensor> params;
  auto descs = this->param_descriptors();
  for (const auto &desc : descs) {
    params.push_back(*desc.data_ptr);
  }
  return params;
}

Vec<Tensor> Layer::gradients() {
  Vec<Tensor> grads;
  auto descs = this->param_descriptors();
  for (const auto &desc : descs) {
    grads.push_back(*desc.grad_ptr);
  }
  return grads;
}

Tensor Layer::get_tensor(const Vec<size_t> &shape, DType_t dtype) {
  if (!allocator_) {
    throw std::runtime_error("Allocator is not set");
  }
  return make_tensor(dtype, shape, device());
}

void Layer::set_immutable_cache(size_t mb_id, const std::string &key, ConstTensor value) {
  if (!is_training_) {
    return;  // no need to cache in inference mode
  }
  immutable_cache_[{mb_id, key}] = std::move(value);
}

ConstTensor &Layer::get_immutable_cache(size_t mb_id, const std::string &key) {
  return immutable_cache_[{mb_id, key}];
}

void Layer::set_mutable_cache(size_t mb_id, const std::string &key, Tensor value) {
  if (!is_training_) {
    return;  // no need to cache in inference mode
  }
  mutable_cache_[{mb_id, key}] = std::move(value);
}

Tensor &Layer::get_mutable_cache(size_t mb_id, const std::string &key) {
  return mutable_cache_[{mb_id, key}];
}

Tensor Layer::get_output_tensor(const Vec<size_t> &shape) {
  if (!allocator_) {
    throw std::runtime_error("Allocator is not set");
  }
  Tensor output_tensor = make_tensor(*allocator_, io_dtype_, shape);
  return output_tensor;
}

Tensor Layer::get_cache_tensor(const Vec<size_t> &shape, DType_t dtype) {
  if (!allocator_) {
    throw std::runtime_error("Allocator is not set");
  }
  int old_side = allocator_->side();
  if (is_training_ && is_fwd_) {
    allocator_->set_side(0);
  }
  Tensor cache_tensor = make_tensor(*allocator_, dtype, shape);
  allocator_->set_side(old_side);  // reset to original side after allocation
  return cache_tensor;
}

Tensor Layer::get_workspace(const Vec<size_t> &shape, DType_t dtype) {
  if (!allocator_) {
    throw std::runtime_error("Allocator is not set");
  }
  int old_side = allocator_->side();
  if (is_training_ && is_fwd_) {
    allocator_->set_side(1);
  }
  Tensor workspace_tensor = make_tensor(*allocator_, dtype, shape);
  allocator_->set_side(old_side);
  return workspace_tensor;
}

void Layer::clear_cache(size_t mb_id) {
  for (auto it = immutable_cache_.begin(); it != immutable_cache_.end();) {
    if (it->first.first == mb_id) {
      it = immutable_cache_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = mutable_cache_.begin(); it != mutable_cache_.end();) {
    if (it->first.first == mb_id) {
      it = mutable_cache_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace tnn
