#pragma once

#include <string>
#include <unordered_map>
#include <any>
#include "../optimizers.hpp"

namespace tnn {

// Base configuration for all layers
struct LayerConfig {
  std::string name;
  std::unordered_map<std::string, std::any> parameters;

  template <typename T>
  T get(const std::string &key, const T &default_value = T{}) const {
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

// Abstract base layer interface
template <typename T = float> class Layer {
public:
  virtual ~Layer() = default;

  // Core forward/backward operations
  virtual Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) = 0;
  virtual Tensor<T> backward(const Tensor<T> &grad_output,
                             int micro_batch_id = 0) = 0;

  // Parameter management
  virtual std::vector<Tensor<T> *> parameters() { return {}; }
  virtual std::vector<Tensor<T> *> gradients() { return {}; }
  virtual bool has_parameters() const { return false; }

  // Configuration and introspection
  virtual std::string type() const = 0;
  virtual LayerConfig get_config() const = 0;
  virtual std::unique_ptr<Layer<T>> clone() const = 0;

  // Training state
  virtual void set_training(bool training) { is_training_ = training; }
  virtual bool is_training() const { return is_training_; }

  // Output shape inference for different Tensor<T> types
  virtual std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const = 0;

  // Optional: custom parameter update (for layers that need special handling)
  virtual void update_parameters(Optimizer<T> &optimizer) {}

  std::string name() const { return name_; }

protected:
  bool is_training_ = true;
  std::string name_;
};

} // namespace tnn