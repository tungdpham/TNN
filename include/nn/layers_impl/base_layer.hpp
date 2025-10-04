/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "../optimizers.hpp"
#include <any>
#include <map>
#include <string>
#include <unordered_map>

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
  virtual ~Layer() = default;

  virtual Tensor<T> forward(const Tensor<T> &input, size_t micro_batch_id = 0) = 0;
  virtual Tensor<T> backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) = 0;

  virtual void forward_inplace(Tensor<T> &input, size_t micro_batch_id = 0) {
    input = forward(input, micro_batch_id);
  }

  virtual void backward_inplace(Tensor<T> &gradient, size_t micro_batch_id = 0) {
    gradient = backward(gradient, micro_batch_id);
  }

  virtual std::vector<Tensor<T> *> parameters() { return {}; }
  virtual std::vector<Tensor<T> *> gradients() { return {}; }

  virtual uint64_t
  forward_complexity(const std::vector<size_t> &input_shape) = 0; // relative complexity
  virtual uint64_t
  backward_complexity(const std::vector<size_t> &input_shape) = 0; // relative complexity

  virtual uint64_t forward_flops(const std::vector<size_t> &input_shape) const = 0;
  virtual uint64_t backward_flops(const std::vector<size_t> &input_shape) const = 0;
  virtual uint64_t forward_memory_traffic(const std::vector<size_t> &input_shape) const = 0;
  virtual uint64_t backward_memory_traffic(const std::vector<size_t> &input_shape) const = 0;
  virtual double forward_arithmetic_intensity(const std::vector<size_t> &input_shape) const = 0;
  virtual double backward_arithmetic_intensity(const std::vector<size_t> &input_shape) const = 0;

  virtual bool has_parameters() const { return false; }

  virtual std::string type() const = 0;
  virtual LayerConfig get_config() const = 0;
  virtual std::unique_ptr<Layer<T>> clone() const = 0;

  virtual void set_training(bool training) { is_training_ = training; }
  virtual bool is_training() const { return is_training_; }

  virtual std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const = 0;

  virtual void update_parameters() {}

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
  mutable std::map<std::string, float> perf_timers_; // For profiling layer's internal performance
  std::string name_;
};

} // namespace tnn