/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tensor/tensor.hpp"
#include <any>
#include <iostream>
#include <map>
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
};

template <typename T = float> class Layer {
public:
  Layer() { this->device_ = &getCPU(); }
  virtual ~Layer() = default;

  virtual void initialize() {};

  void set_seed(unsigned long long seed) {
    use_seed_ = true;
    srand_seed_ = seed;
  }

  virtual const Tensor<T> &forward(const Tensor<T> &input, size_t micro_batch_id = 0) = 0;
  virtual const Tensor<T> &backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) = 0;

  virtual std::vector<Tensor<T> *> parameters() { return {}; }
  virtual std::vector<Tensor<T> *> gradients() { return {}; }

  virtual void clear_gradients() {}

  virtual uint64_t
  forward_complexity(const std::vector<size_t> &input_shape) const = 0; // relative complexity
  virtual uint64_t
  backward_complexity(const std::vector<size_t> &input_shape) const = 0; // relative complexity

  virtual uint64_t forward_flops(const std::vector<size_t> &input_shape) const = 0;
  virtual uint64_t backward_flops(const std::vector<size_t> &input_shape) const = 0;

  virtual bool has_parameters() const { return false; }

  virtual std::string type() const = 0;
  virtual LayerConfig get_config() const = 0;
  virtual std::unique_ptr<Layer<T>> clone() const = 0;

  virtual void set_training(bool training) { is_training_ = training; }
  virtual bool is_training() const { return is_training_; }

  virtual void set_device(const Device *device) { device_ = device; }
  const Device *get_device() const { return device_; }

  virtual std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const = 0;

  void enable_profiling(bool enable) { enable_profiling_ = enable; }

  void print_profiling_info() const {
    std::cout << "Profiling info for layer: " << name_ << std::endl;
    for (const auto &[key, value] : perf_timers_) {
      std::cout << "  " << key << ": " << value << " ms" << std::endl;
    }
  }

  void reset_profiling_info() { perf_timers_.clear(); }

  std::string name() const { return name_; }

protected:
  bool is_training_ = true;
  bool enable_profiling_ = false;
  bool use_seed_ = false;
  unsigned long long srand_seed_ = 0;
  mutable std::map<std::string, float> perf_timers_; // For profiling layer's internal performance
  // buffers for storing intermediate results per micro-batch
  std::unordered_map<size_t, Tensor<T>> output_buffers_;
  std::unordered_map<size_t, Tensor<T>> gradient_buffers_;
  const Device *device_;
  std::string name_;

  Tensor<T> &get_output_buffer(size_t micro_batch_id, const std::vector<size_t> &shape) {
    auto it = output_buffers_.find(micro_batch_id);
    if (it == output_buffers_.end()) {
      output_buffers_.emplace(micro_batch_id, Tensor<T>(shape, this->device_));
      return output_buffers_[micro_batch_id];
    } else {
      auto &buf = it->second;
      if (buf.shape() != shape) {
        buf.resize(shape);
      }
      return buf;
    }
  }

  Tensor<T> &get_gradient_buffer(size_t micro_batch_id, const std::vector<size_t> &shape) {
    auto it = gradient_buffers_.find(micro_batch_id);
    if (it == gradient_buffers_.end()) {
      gradient_buffers_.emplace(micro_batch_id, Tensor<T>(shape, this->device_));
      return gradient_buffers_[micro_batch_id];
    } else {
      auto &buf = it->second;
      if (buf.shape() != shape) {
        buf.resize(shape);
      }
      return buf;
    }
  }
};

} // namespace tnn