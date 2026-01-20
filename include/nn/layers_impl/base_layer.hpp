/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/mem_pool.hpp"
#include "logging/logger.hpp"
#include "profiling/event.hpp"
#include "profiling/profiler.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"
#include <any>
#include <cstring>
#include <fmt/core.h>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

struct LayerConfig {
  std::string name;
  std::unordered_map<std::string, std::any> parameters;

  template <typename T> T get(const std::string &key, const T &default_value = T{}) const {
    auto it = parameters.find(key);
    if (it != parameters.end()) {
      try {
        return std::any_cast<T>(it->second);
      } catch (const std::bad_any_cast &) {
        return default_value;
      }
    }
    return default_value;
  }

  nlohmann::json to_json() const {
    nlohmann::json j;
    j["name"] = name;
    for (const auto &[key, value] : parameters) {
      if (value.type() == typeid(size_t)) {
        j[key] = std::any_cast<size_t>(value);
      } else if (value.type() == typeid(float)) {
        j[key] = std::any_cast<float>(value);
      } else if (value.type() == typeid(bool)) {
        j[key] = std::any_cast<bool>(value);
      } else if (value.type() == typeid(std::string)) {
        j[key] = std::any_cast<std::string>(value);
      } else if (value.type() == typeid(std::vector<nlohmann::json>)) {
        j[key] = std::any_cast<std::vector<nlohmann::json>>(value);
      }
    }
    return j;
  }

  static LayerConfig from_json(const nlohmann::json &j) {
    LayerConfig config;
    config.name = j.value("name", "");
    if (j.contains("parameters")) {
      for (const auto &[key, value] : j["parameters"].items()) {
        if (value.is_number_integer()) {
          config.parameters[key] = value.template get<size_t>();
        } else if (value.is_number_float()) {
          config.parameters[key] = value.template get<float>();
        } else if (value.is_boolean()) {
          config.parameters[key] = value.template get<bool>();
        } else if (value.is_string()) {
          config.parameters[key] = value.template get<std::string>();
        }
      }
    }
    return config;
  }
};

class Layer {
public:
  Layer() {
    this->device_ = &getCPU();
    this->mem_pool_ = &global_mem_pool();
  }

  virtual ~Layer() = default;

  // Note: have to reinit after changing device
  void set_device(const Device &device) {
    device_ = &device;
    on_set_device(device);
  }

  const Device *get_device() const { return device_; }

  void set_io_dtype(DType_t dtype) {
    io_dtype_ = dtype;
    on_set_io_dtype(dtype);
  }

  DType_t get_io_dtype() const { return io_dtype_; }

  void set_param_dtype(DType_t dtype) {
    param_dtype_ = dtype;
    on_set_param_dtype(dtype);
  }

  // Note: have to call init again after changing param dtype
  DType_t get_param_dtype() const { return param_dtype_; }

  void set_compute_dtype(DType_t dtype) {
    compute_dtype_ = dtype;
    on_set_compute_dtype(dtype);
  }

  DType_t get_compute_dtype() const { return compute_dtype_; }

  /**
   * @brief Initialize the layer (e.g., allocate parameters)
   * ! Must set io, param, compute dtypes and device ptr prior to this to ensure proper math.
   * ! Must be called before forward/backward.
   */
  void init() {
    if (initialized_) {
      return;
    }
    init_impl();
    initialized_ = true;
  };

  void set_seed(unsigned long long seed) {
    use_seed_ = true;
    srand_seed_ = seed;
  }

  void forward(const Tensor &input, Tensor &output, size_t micro_batch_id = 0) {
    if (!initialized_) {
      std::cerr << "Warning: Layer " << name_ << " is not initialized. Call init() before forward."
                << std::endl;
      return;
    }
    Clock::time_point start_time = Clock::now();
    if (input == nullptr || output == nullptr) {
      throw std::runtime_error("Layer " + name_ + " received null IO tensor.");
    }
    if (input->data_type() != io_dtype_) {
      throw std::runtime_error("Layer " + name_ +
                               " input tensor dtype does not match layer io_dtype.");
    }
    if (output->data_type() != io_dtype_) {
      throw std::runtime_error("Layer " + name_ +
                               " output tensor dtype does not match layer io_dtype.");
    }
    const Tensor *current = &input;
    Tensor device_input;
    if (input->device() != this->device_) {
      device_input = this->get_buffer(input->shape());
      input->copy_to(device_input);
      current = &device_input;
    }
    forward_impl(*current, output, micro_batch_id);
    // this->device_->getFlow("default")->synchronize();
    Clock::time_point end_time = Clock::now();
    profiler_.add_event(Event{EventType::COMPUTE, start_time, end_time, "forward"});
  }

  void backward(const Tensor &gradient, Tensor &grad_input, size_t micro_batch_id = 0) {
    if (!initialized_) {
      std::cerr << "Warning: Layer " << name_ << " is not initialized. Call init() before backward."
                << std::endl;
      return;
    }
    auto start_time = Clock::now();
    if (gradient == nullptr || grad_input == nullptr) {
      throw std::runtime_error("Layer " + name_ + " received null gradient or grad_input tensor.");
    }
    if (gradient->data_type() != io_dtype_) {
      throw std::runtime_error("Layer " + name_ +
                               " gradient tensor dtype does not match layer io_dtype.");
    }
    if (grad_input->data_type() != io_dtype_) {
      throw std::runtime_error("Layer " + name_ +
                               " grad_input tensor dtype does not match layer io_dtype.");
    }
    const Tensor *current_gradient = &gradient;
    Tensor device_gradient;
    if (gradient->device() != this->device_) {
      device_gradient = this->get_buffer(gradient->shape());
      gradient->copy_to(device_gradient);
      current_gradient = &device_gradient;
    }
    backward_impl(*current_gradient, grad_input, micro_batch_id);
    // this->device_->getFlow("default")->synchronize();
    Clock::time_point end_time = Clock::now();
    profiler_.add_event(Event{EventType::COMPUTE, start_time, end_time, "backward"});
  }

  virtual std::vector<Tensor> parameters() { return {}; }

  virtual std::vector<Tensor> gradients() { return {}; }

  virtual void save_state(std::ofstream &file) {
    auto params = parameters();
    for (const auto &param : params) {
      param->save(file);
    }
  }

  virtual void load_state(std::ifstream &file) {
    auto params = parameters();
    for (auto &param : params) {
      if (this->device_ == nullptr) {
        std::cerr << "ERR: Device not set for Layer " << name_ << " when loading state."
                  << std::endl;
        return;
      }
      param = load(file, this->device_);
      set_param_dtype(param->data_type());
    }
    this->initialized_ = true;
  }

  virtual uint64_t forward_flops(const std::vector<size_t> &input_shape) const = 0;

  virtual uint64_t backward_flops(const std::vector<size_t> &input_shape) const = 0;

  virtual bool has_parameters() const { return false; }

  virtual std::string type() const = 0;

  virtual LayerConfig get_config() const = 0;

  virtual std::unique_ptr<Layer> clone() const = 0;

  void set_training(bool training) {
    is_training_ = training;
    on_set_training(training);
  }

  bool is_training() const { return is_training_; }

  virtual std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const = 0;

  void enable_profiling(bool enable) { enable_profiling_ = enable; }

  bool is_profiling_enabled() const { return enable_profiling_; }

  void print_profiling_info() const {
    const auto &events = profiler_.get_events();
    std::string header = "Profiling info for Layer: " + name_;
    std::string format_str = "{:=^70}\n";
    std::string output = fmt::format(fmt::runtime(format_str), header);
    output += fmt::format(fmt::runtime("{:<20}{:<25}{:<25}\n"), "Event", "Duration (ms)", "");
    output += fmt::format(fmt::runtime("{:-^70}\n"), "");
    for (const auto &event : events) {
      float duration_ms =
          Time::duration_cast<Time::microseconds>(event.end_time - event.start_time).count() /
          1000.0f;
      output += fmt::format(fmt::runtime("{:<20}{:<25.3f}\n"), event.name, duration_ms);
    }
    output += fmt::format(fmt::runtime("{:=^70}"), "");
    GlobalLogger::info(output);
  }

  void set_mem_pool(MemPool *mem_pool) { mem_pool_ = mem_pool; }

  const MemPool *get_mem_pool() const { return mem_pool_; }

  virtual size_t cached_memory_bytes() const { return 0; }

  void reset_profiling_info() { profiler_.reset(); }

  std::string name() const { return name_; }

protected:
  bool initialized_ = false;
  bool is_training_ = true;
  bool enable_profiling_ = false;
  bool use_seed_ = false;
  unsigned long long srand_seed_ = 0;
  std::map<std::string, Tensor> cached_tensors_;
  Profiler profiler_;
  MemPool *mem_pool_;
  const Device *device_;
  std::string name_;
  DType_t io_dtype_ = DType_t::FP32;      // data type for input/output tensors
  DType_t param_dtype_ = DType_t::FP32;   // data type for parameters/gradients
  DType_t compute_dtype_ = DType_t::FP32; // data type for internal computations

  virtual void on_set_device(const Device &device) {}
  virtual void on_set_training(bool training) {}
  virtual void on_set_io_dtype(DType_t dtype) {}
  virtual void on_set_param_dtype(DType_t dtype) {}
  virtual void on_set_compute_dtype(DType_t dtype) {}
  virtual void init_impl() = 0;
  virtual void forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id = 0) = 0;
  virtual void backward_impl(const Tensor &gradient, Tensor &grad_input,
                             size_t micro_batch_id = 0) = 0;

  // helpers
  Tensor make_param_tensor(std::vector<size_t> shape) {
    return make_tensor_from_dtype(param_dtype_, shape, this->device_);
  }

  Tensor make_io_tensor(std::vector<size_t> shape) {
    return make_tensor_from_dtype(io_dtype_, shape, this->device_);
  }

  Tensor make_compute_tensor(std::vector<size_t> shape) {
    return make_tensor_from_dtype(compute_dtype_, shape, this->device_);
  }

  Tensor &get_cached_tensor(size_t mb_id, const std::string &key, const std::vector<size_t> &shape,
                            DType_t dtype = DType_t::FP32) {
    auto aggregate_key = std::to_string(mb_id) + key;
    auto it = cached_tensors_.find(aggregate_key);
    if (it != cached_tensors_.end()) {
      it->second->ensure(shape, this->device_);
      return it->second;
    }
    Tensor tensor = make_tensor_from_dtype(dtype, shape, this->device_);
    cached_tensors_[aggregate_key] = std::move(tensor);
    return cached_tensors_[aggregate_key];
  }

  Tensor get_buffer(const std::vector<size_t> &shape, DType_t dtype = DType_t::FP32) {
    if (!mem_pool_) {
      throw std::runtime_error("MemPool not set for layer: " + name_);
    }
    if (!this->device_) {
      throw std::runtime_error("Device not set for layer: " + name_);
    }
    return make_pooled_tensor_from_dtype(*mem_pool_, dtype, shape, this->device_);
  }
};

#define DISPATCH_ON_DTYPE_TO_METHOD(method_name, ...)                                              \
  do {                                                                                             \
    DISPATCH_ON_DTYPE(this->compute_dtype_, IO_T, method_name<IO_T>(__VA_ARGS__));                 \
  } while (0)

#define DISPATCH_ON_3_DTYPES_TO_METHOD(method_name, ...)                                           \
  do {                                                                                             \
    DISPATCH_ON_DTYPE(                                                                             \
        this->io_dtype_, IO_T,                                                                     \
        DISPATCH_ON_DTYPE(this->param_dtype_, PARAM_T,                                             \
                          DISPATCH_ON_DTYPE(this->compute_dtype_, COMP_T,                          \
                                            method_name<IO_T, PARAM_T, COMP_T>(__VA_ARGS__);)));   \
  } while (0)
} // namespace tnn