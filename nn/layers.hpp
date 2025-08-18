#pragma once

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../tensor/tensor.hpp"
#include "activations.hpp"
#include "optimizers.hpp"

namespace tnn {

// Convenience function for creating tensor activation functions
template <typename T = float>
std::unique_ptr<ActivationFunction<T>>
create_activation(const std::string &name);

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

// Base class for layers without parameters (activation, pooling, etc.)
template <typename T = float> class StatelessLayer : public Layer<T> {
public:
  explicit StatelessLayer(const std::string &name = "") { this->name_ = name; }
  std::vector<Tensor<T> *> parameters() override { return {}; }
  std::vector<Tensor<T> *> gradients() override { return {}; }
  bool has_parameters() const override { return false; }
};

// Base class for layers with parameters (dense, conv, etc.)
template <typename T = float> class ParameterizedLayer : public Layer<T> {
public:
  explicit ParameterizedLayer(const std::string &name = "") {
    this->name_ = name;
  }

  std::vector<Tensor<T> *> parameters() override;
  std::vector<Tensor<T> *> gradients() override;
  bool has_parameters() const override { return true; }
  void update_parameters(Optimizer<T> &optimizer) override;

protected:
  virtual void collect_parameters(std::vector<Tensor<T> *> &params) = 0;
  virtual void collect_gradients(std::vector<Tensor<T> *> &grads) = 0;
  virtual void update_parameters_impl(Optimizer<T> &optimizer) = 0;
};

// Forward declarations for specific layers
template <typename T> class DenseLayer;
template <typename T> class ActivationLayer;
template <typename T> class Conv2DLayer;
template <typename T> class MaxPool2DLayer;
template <typename T> class DropoutLayer;
template <typename T> class FlattenLayer;
template <typename T> class LayerFactory;
template <typename T> class BatchNormLayer;

} // namespace tnn


// Include the implementation for the create_activation function
template <typename T>
std::unique_ptr<ActivationFunction<T>>
tnn::create_activation(const std::string &name) {
  // Ensure factory has default activations registered
  ActivationFactory<T>::register_defaults();
  return ActivationFactory<T>::create(name);
}

#include "layers_impl/parameterized_layer.cpp"

// Include specific layer headers
#include "layers_impl/dense_layer.hpp"
#include "layers_impl/activation_layer.hpp"
#include "layers_impl/conv2d_layer.hpp"
#include "layers_impl/maxpool2d_layer.hpp"
#include "layers_impl/dropout_layer.hpp"
#include "layers_impl/flatten_layer.hpp"
#include "layers_impl/layer_factory.hpp"
#include "layers_impl/batchnorm_layer.hpp"

// Convenience functions for creating layers
namespace tnn {

template <typename T = float>
std::unique_ptr<Layer<T>>
dense(size_t input_features, size_t output_features,
      const std::string &activation = "none", bool use_bias = true,
      const std::string &name = "dense") {
  std::unique_ptr<ActivationFunction<T>> act = nullptr;
  if (activation != "none") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    act = factory.create(activation);
  }

  return std::make_unique<DenseLayer<T>>(input_features, output_features,
                                              std::move(act), use_bias, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>>
conv2d(size_t in_channels, size_t out_channels, size_t kernel_h,
       size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
       size_t pad_h = 0, size_t pad_w = 0,
       const std::string &activation = "none", bool use_bias = true,
       const std::string &name = "conv2d") {
  std::unique_ptr<ActivationFunction<T>> act = nullptr;
  if (activation != "none") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    act = factory.create(activation);
  }
  return std::make_unique<Conv2DLayer<T>>(
      in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h,
      pad_w, use_bias, std::move(act), name);
}

template <typename T = float>
std::unique_ptr<Layer<T>>
activation(const std::string &activation_name,
           const std::string &name = "activation") {
  auto factory = ActivationFactory<T>();
  factory.register_defaults();
  auto act = factory.create(activation_name);
  return std::make_unique<ActivationLayer<T>>(std::move(act), name);
}

template <typename T = float>
std::unique_ptr<Layer<T>>
maxpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 0,
          size_t stride_w = 0, size_t pad_h = 0, size_t pad_w = 0,
          const std::string &name = "maxpool2d") {
  return std::make_unique<MaxPool2DLayer<T>>(pool_h, pool_w, stride_h,
                                                  stride_w, pad_h, pad_w, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>> dropout(T dropout_rate,
                                       const std::string &name = "dropout") {
  return std::make_unique<DropoutLayer<T>>(dropout_rate, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>>
batchnorm(size_t num_features, T epsilon = T(1e-5), T momentum = T(0.1),
          bool affine = true, const std::string &name = "batchnorm") {
  return std::make_unique<BatchNormLayer<T>>(num_features, epsilon,
                                                  momentum, affine, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>> flatten(const std::string &name = "flatten") {
  return std::make_unique<FlattenLayer<T>>(name);
}

} // namespace tnn