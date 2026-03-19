/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <cstddef>
#include <cstring>

#include "common/config.hpp"
#include "device/del_allocator.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tnn {
using LayerConfig = TConfig;

struct ParamDescriptor {
  DType_t dtype;  // data type of the parameter
  std::vector<size_t> shape;
  Tensor *data_ptr;  // pointer to the actual param
  Tensor *grad_ptr;  // pointer to the actual grad_output
};

// Single input/output layer interface. Can be easily extended to multiple inputs/outputs later if
// needed.
class Layer {
public:
  Layer() = default;
  Layer(const std::string &name)
      : name_(name) {}

  virtual ~Layer() = default;

  void init();
  void forward(const std::vector<ConstTensor> &inputs, const std::vector<Tensor> &outputs,
               size_t mb_id = 0);
  void backward(const std::vector<ConstTensor> &grad_outputs,
                const std::vector<Tensor> &grad_inputs, size_t mb_id = 0);

  // Note: have to call init again after changing param dtype
  Layer &set_allocator(DELAllocator &allocator);
  DELAllocator *get_allocator() const;
  Layer &set_flow_handle(flowHandle_t handle);
  flowHandle_t get_flow_handle() const;
  Layer &set_seed(unsigned long long seed);
  Layer &set_io_dtype(DType_t dtype);
  DType_t get_io_dtype() const;
  Layer &set_param_dtype(DType_t dtype);
  DType_t get_param_dtype() const;
  Layer &set_compute_dtype(DType_t dtype);
  DType_t get_compute_dtype() const;
  Layer &set_training(bool training);
  bool is_training() const;

  virtual Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const = 0;
  virtual size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const { return 0; }
  virtual size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const { return 0; }
  virtual size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const { return 0; }
  virtual size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const { return 0; }
  std::string name() const { return name_; }
  void save_state(std::ofstream &file);
  virtual std::vector<ParamDescriptor> param_descriptors() { return {}; }
  virtual std::string type() const = 0;
  virtual LayerConfig get_config() const = 0;
  const Device &device() const {
    if (!allocator_) {
      throw std::runtime_error("Allocator is not set");
    }
    return allocator_->device();
  }

  std::vector<Tensor> parameters();
  std::vector<Tensor> gradients();

protected:
  virtual void init_impl() {}
  virtual void on_set_allocator(DELAllocator &allocator) {}
  virtual void on_set_flow_handle(flowHandle_t handle) {}
  virtual void on_set_seed(unsigned long long seed) {}
  virtual void on_set_training(bool training) {}
  virtual void on_set_io_dtype(DType_t dtype) {}
  virtual void on_set_param_dtype(DType_t dtype) {}
  virtual void on_set_compute_dtype(DType_t dtype) {}
  virtual void forward_impl(const std::vector<ConstTensor> &inputs,
                            const std::vector<Tensor> &outputs, size_t mb_id) = 0;
  virtual void backward_impl(const std::vector<ConstTensor> &grad_outputs,
                             const std::vector<Tensor> &grad_inputs, size_t mb_id) = 0;

protected:
  bool initialized_ = false;
  DELAllocator *allocator_ = nullptr;
  bool is_training_ = true;
  bool use_seed_ = false;
  unsigned long long srand_seed_ = 0;
  std::map<std::pair<size_t, std::string>, Tensor> mutable_tensors_;
  // for immutable cache (e.g., inputs)
  std::map<std::pair<size_t, std::string>, ConstTensor> cached_tensors_;
  flowHandle_t flow_handle_;
  std::string name_;
  DType_t io_dtype_ = DType_t::FP32;       // data type for input/output tensors
  DType_t param_dtype_ = DType_t::FP32;    // data type for parameters/gradients
  DType_t compute_dtype_ = DType_t::FP32;  // data type for internal computations

  // helpers
  Tensor make_io_tensor(std::vector<size_t> shape);
  Tensor make_compute_tensor(std::vector<size_t> shape);
  ConstTensor &get_cached_tensor(size_t mb_id, const std::string &key);
  Tensor &get_mutable_tensor(size_t mb_id, const std::string &key);
  Tensor get_act(const std::vector<size_t> &shape);
  Tensor get_workspace(const std::vector<size_t> &shape, DType_t dtype = DType_t::FP32);
  void clear_cache(size_t mb_id);
};

#define DISPATCH_IO_DTYPE(method_name, ...)                                \
  do {                                                                     \
    DISPATCH_DTYPE(this->io_dtype_, IO_T, method_name<IO_T>(__VA_ARGS__)); \
  } while (0)

#define DISPATCH_ON_3_DTYPES_TO_METHOD(method_name, ...)                                   \
  do {                                                                                     \
    DISPATCH_DTYPE(                                                                        \
        this->io_dtype_, IO_T,                                                             \
        DISPATCH_DTYPE(this->param_dtype_, PARAM_T,                                        \
                       DISPATCH_DTYPE(this->compute_dtype_, COMP_T,                        \
                                      method_name<IO_T, PARAM_T, COMP_T>(__VA_ARGS__);))); \
  } while (0)
}  // namespace tnn