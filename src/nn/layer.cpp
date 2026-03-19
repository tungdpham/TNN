/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layer.hpp"

#include "device/flow.hpp"
#include "type/type.hpp"

namespace tnn {

void Layer::init() {
  if (initialized_) {
    throw std::runtime_error("Cannot initalize Layer more than once. ");
  }
  init_impl();
  initialized_ = true;
}

void Layer::forward(const std::vector<ConstTensor> &inputs, const std::vector<Tensor> &outputs,
                    size_t mb_id) {
  if (!initialized_) {
    throw std::runtime_error("Layer must be initialized before calling forward");
  }
  std::vector<ConstTensor> current_inputs;
  for (auto &input : inputs) {
    if (input->device() == this->device())
      current_inputs.push_back(input);
    else
      current_inputs.push_back(input->to_device(this->device()));
  }
  forward_impl(current_inputs, outputs, mb_id);
}

void Layer::backward(const std::vector<ConstTensor> &grad_outputs,
                     const std::vector<Tensor> &grad_inputs, size_t mb_id) {
  if (!initialized_) {
    throw std::runtime_error("Layer must be initialized before calling backward");
  }
  std::vector<ConstTensor> current_grad_outputs;
  for (auto &grad : grad_outputs) {
    if (grad->device() == this->device())
      current_grad_outputs.push_back(grad);
    else
      current_grad_outputs.push_back(grad->to_device(this->device()));
  }
  backward_impl(current_grad_outputs, grad_inputs, mb_id);
  clear_cache(mb_id);
}

Layer &Layer::set_allocator(DELAllocator &allocator) {
  allocator_ = &allocator;
  on_set_allocator(allocator);
  return *this;
}

DELAllocator *Layer::get_allocator() const { return allocator_; }

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

std::vector<Tensor> Layer::parameters() {
  std::vector<Tensor> params;
  auto descs = this->param_descriptors();
  for (const auto &desc : descs) {
    params.push_back(*desc.data_ptr);
  }
  return params;
}

std::vector<Tensor> Layer::gradients() {
  std::vector<Tensor> grads;
  auto descs = this->param_descriptors();
  for (const auto &desc : descs) {
    grads.push_back(*desc.grad_ptr);
  }
  return grads;
}

Tensor Layer::make_io_tensor(std::vector<size_t> shape) {
  return make_tensor(io_dtype_, shape, device());
}

Tensor Layer::make_compute_tensor(std::vector<size_t> shape) {
  return make_tensor(compute_dtype_, shape, device());
}

ConstTensor &Layer::get_cached_tensor(size_t mb_id, const std::string &key) {
  return cached_tensors_[{mb_id, key}];
}

Tensor &Layer::get_mutable_tensor(size_t mb_id, const std::string &key) {
  return mutable_tensors_[{mb_id, key}];
}

Tensor Layer::get_act(const std::vector<size_t> &shape) {
  if (!allocator_) {
    throw std::runtime_error("Allocator is not set");
  }
  if (is_training_) {
    allocator_->set_side(0);
  }
  return make_tensor(*allocator_, io_dtype_, shape);
}

Tensor Layer::get_workspace(const std::vector<size_t> &shape, DType_t dtype) {
  if (!allocator_) {
    throw std::runtime_error("Allocator is not set");
  }
  if (is_training_) {
    allocator_->set_side(1);
  }
  return make_tensor(*allocator_, dtype, shape);
}

void Layer::clear_cache(size_t mb_id) {
  for (auto it = cached_tensors_.begin(); it != cached_tensors_.end();) {
    if (it->first.first == mb_id) {
      it = cached_tensors_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace tnn
