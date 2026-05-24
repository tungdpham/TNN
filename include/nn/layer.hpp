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
#include "device/del_allocator_v2.hpp"
#include "device/engine.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tnn {
using LayerConfig = TConfig;

struct ParamDescriptor {
  DType_t dtype;  // data type of the parameter
  Vec<size_t> shape;
  Tensor *data_ptr;  // pointer to the actual param
  Tensor *grad_ptr;  // pointer to the actual grad_output
};

inline size_t get_shapes_bytes(const Vec<Vec<size_t>> &shapes, DType_t dtype) {
  size_t total_bytes = 0;
  size_t dtype_size = get_dtype_size(dtype);
  for (const auto &shape : shapes) {
    size_t shape_bytes =
        std::accumulate(shape.begin(), shape.end(), dtype_size, std::multiplies<size_t>());
    shape_bytes = align_up(shape_bytes, 256);
    total_bytes += shape_bytes;
  }
  return total_bytes;
}

// Single input/output layer interface. Can be easily extended to multiple inputs/outputs later if
// needed.
class Layer {
public:
  Layer() = default;
  Layer(const std::string &name)
      : name_(name) {}

  virtual ~Layer() = default;

  void set_engine_type(EngineType engine_type);
  EngineType get_engine_type() const;

  void init();
  Vec<Tensor> forward(const Vec<ConstTensor> &inputs, size_t mb_id = 0);
  Vec<Tensor> backward(const Vec<ConstTensor> &grad_outputs, size_t mb_id = 0);

  // Note: have to call init again after changing param dtype
  Layer &set_allocator(DELAllocatorV2 &allocator);
  DELAllocatorV2 *get_allocator() const;
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
  std::string name() const { return name_; }
  void save_state(std::ofstream &file);
  virtual Vec<ParamDescriptor> param_descriptors() { return {}; }
  virtual std::string type() const = 0;
  virtual LayerConfig get_config() const = 0;

  Vec<Tensor> parameters();
  Vec<Tensor> gradients();
  void clear_cache(size_t mb_id);

  const Device &device() const {
    if (!allocator_) {
      throw std::runtime_error("Layer: Allocator is not set to get device.");
    }
    return allocator_->device();
  }

protected:
  virtual void on_set_engine_type(EngineType engine_type) {}
  virtual void init_impl() {}
  virtual void on_set_allocator(DELAllocatorV2 &allocator) {}
  virtual void on_set_flow_handle(flowHandle_t handle) {}
  virtual void on_set_seed(unsigned long long seed) {}
  virtual void on_set_training(bool training) {}
  virtual void on_set_io_dtype(DType_t dtype) {}
  virtual void on_set_param_dtype(DType_t dtype) {}
  virtual void on_set_compute_dtype(DType_t dtype) {}
  virtual Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) = 0;
  virtual Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) = 0;

protected:
  bool initialized_ = false;
  EngineType engine_type_ = EngineType::UNKNOWN;
  DELAllocatorV2 *allocator_ = nullptr;
  bool is_training_ = true;
  bool is_fwd_ = false;
  bool use_seed_ = false;
  unsigned long long srand_seed_ = 0;
  std::map<std::pair<size_t, std::string>, ConstTensor> immutable_cache_;
  std::map<std::pair<size_t, std::string>, Tensor> mutable_cache_;
  flowHandle_t flow_handle_;
  std::string name_;
  DType_t io_dtype_ = DType_t::FP32;       // data type for input/output tensors
  DType_t param_dtype_ = DType_t::FP32;    // data type for parameters/gradients
  DType_t compute_dtype_ = DType_t::FP32;  // data type for internal computations

  // helpers
  void set_immutable_cache(size_t mb_id, const std::string &key, ConstTensor value);
  ConstTensor &get_immutable_cache(size_t mb_id, const std::string &key);
  void set_mutable_cache(size_t mb_id, const std::string &key, Tensor value);
  Tensor &get_mutable_cache(size_t mb_id, const std::string &key);
  Tensor get_tensor(const Vec<size_t> &shape, DType_t dtype);
  Tensor get_output_tensor(const Vec<size_t> &shape);
  Tensor get_cache_tensor(const Vec<size_t> &shape = {}, DType_t dtype = DType_t::FP32);
  Tensor get_workspace(const Vec<size_t> &shape, DType_t dtype = DType_t::FP32);
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