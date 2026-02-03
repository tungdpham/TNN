/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <any>
#include <cstddef>
#include <cstring>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "profiling/profiler.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tnn {

struct LayerConfig {
  std::string name;
  std::string type;
  std::unordered_map<std::string, std::any> parameters;

  template <typename T>
  T get(const std::string &key, const T &default_value = T{}) const;

  nlohmann::json to_json() const;

  static LayerConfig from_json(const nlohmann::json &j);
};

class Layer {
public:
  Layer();

  virtual ~Layer() = default;

  // Note: have to reinit after changing device
  void set_device(const Device &device);

  const Device &get_device() const;

  void set_io_dtype(DType_t dtype);

  DType_t get_io_dtype() const;

  void set_param_dtype(DType_t dtype);

  // Note: have to call init again after changing param dtype
  DType_t get_param_dtype() const;

  void set_compute_dtype(DType_t dtype);

  DType_t get_compute_dtype() const;

  /**
   * @brief Initialize the layer (e.g., allocate parameters)
   * ! Must set io, param, compute dtypes and device ptr prior to this to ensure proper math.
   * ! Must be called before forward/backward.
   */
  void init();

  void set_seed(unsigned long long seed);

  void forward(const ConstTensor &input, const Tensor &output, size_t mb_id = 0);

  void backward(const ConstTensor &gradient, const Tensor &grad_input, size_t mb_id = 0);

  virtual std::vector<Tensor> parameters();

  virtual std::vector<Tensor> gradients();

  void save_state(std::ofstream &file);

  virtual uint64_t forward_flops(const std::vector<size_t> &input_shape) const = 0;

  virtual uint64_t backward_flops(const std::vector<size_t> &input_shape) const = 0;

  virtual bool has_parameters() const { return false; }

  virtual std::string type() const = 0;

  virtual LayerConfig get_config() const = 0;

  virtual std::unique_ptr<Layer> clone() const = 0;

  void set_training(bool training);

  bool is_training() const;

  virtual std::vector<size_t> compute_output_shape(
      const std::vector<size_t> &input_shape) const = 0;

  void enable_profiling(bool enable);

  bool is_profiling_enabled() const;

  void print_profiling_info() const;

  void set_mem_pool(PoolAllocator *mem_pool);

  const PoolAllocator *get_mem_pool() const;

  size_t nbytes_params();

  virtual size_t cached_memory_bytes() const;

  void reset_profiling_info();

  std::string name() const;

protected:
  bool initialized_ = false;
  bool is_training_ = true;
  bool enable_profiling_ = false;
  bool use_seed_ = false;
  unsigned long long srand_seed_ = 0;
  std::map<std::pair<size_t, std::string>, Tensor> mutable_tensors_;
  std::map<std::pair<size_t, std::string>, ConstTensor>
      cached_tensors_;  // for immutable cache (e.g., inputs)
  Profiler profiler_;
  PoolAllocator *mem_pool_;
  csref<Device> device_ = getCPU();
  std::string name_;
  DType_t io_dtype_ = DType_t::FP32;       // data type for input/output tensors
  DType_t param_dtype_ = DType_t::FP32;    // data type for parameters/gradients
  DType_t compute_dtype_ = DType_t::FP32;  // data type for internal computations

  virtual void on_set_device(const Device &device) {}
  virtual void on_set_training(bool training) {}
  virtual void on_set_io_dtype(DType_t dtype) {}
  virtual void on_set_param_dtype(DType_t dtype) {}
  virtual void on_set_compute_dtype(DType_t dtype) {}
  virtual void init_impl() = 0;
  virtual void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) = 0;
  virtual void backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                             size_t mb_id = 0) = 0;

  // helpers
  Tensor make_param_tensor(std::vector<size_t> shape);

  Tensor make_io_tensor(std::vector<size_t> shape);

  Tensor make_compute_tensor(std::vector<size_t> shape);

  ConstTensor &get_cached_tensor(size_t mb_id, const std::string &key);

  Tensor &get_mutable_tensor(size_t mb_id, const std::string &key);

  Tensor get_buffer(const std::vector<size_t> &shape, DType_t dtype = DType_t::FP32);

private:
  void clear_cache(size_t mb_id);
};

#define DISPATCH_ON_DTYPE_TO_METHOD(method_name, ...)                         \
  do {                                                                        \
    DISPATCH_ON_DTYPE(this->io_dtype_, IO_T, method_name<IO_T>(__VA_ARGS__)); \
  } while (0)

#define DISPATCH_ON_3_DTYPES_TO_METHOD(method_name, ...)                                         \
  do {                                                                                           \
    DISPATCH_ON_DTYPE(                                                                           \
        this->io_dtype_, IO_T,                                                                   \
        DISPATCH_ON_DTYPE(this->param_dtype_, PARAM_T,                                           \
                          DISPATCH_ON_DTYPE(this->compute_dtype_, COMP_T,                        \
                                            method_name<IO_T, PARAM_T, COMP_T>(__VA_ARGS__);))); \
  } while (0)
}  // namespace tnn