#pragma once
#include <any>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../matrix/matrix.hpp"
#include "activations.hpp"

// Forward declarations
class Optimizer;

namespace layers {

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
template <typename T> class Layer {
public:
  virtual ~Layer() = default;

  // Core forward/backward operations
  virtual Matrix forward(const Matrix &input) = 0;
  virtual Matrix backward(const Matrix &grad_output) = 0;

  // Parameter management
  virtual std::vector<Matrix *> parameters() { return {}; }

  virtual std::vector<Matrix *> gradients() { return {}; }

  virtual bool has_parameters() const { return false; }

  // Configuration and introspection
  virtual std::string type() const = 0;
  virtual LayerConfig get_config() const = 0;
  virtual std::unique_ptr<Layer<T>> clone() const = 0;

  // Training state
  virtual void set_training(bool training) { is_training_ = training; }

  virtual bool is_training() const { return is_training_; }

  // Output shape inference
  virtual std::tuple<int, int, int>
  compute_output_shape(int input_rows, int input_cols,
                       int input_channels) const = 0;

  // Optional: custom parameter update (for layers that need special handling)
  virtual void update_parameters(const Optimizer &optimizer) {}

protected:
  bool is_training_ = true;
  std::string name_;
};

// Base class for layers without parameters (activation, pooling, etc.)
template <typename T> class StatelessLayer : public Layer<T> {
public:
  explicit StatelessLayer(const std::string &name = "") { this->name_ = name; }

  std::vector<Matrix *> parameters() override { return {}; }

  std::vector<Matrix *> gradients() override { return {}; }

  bool has_parameters() const override { return false; }
};

// Base class for layers with parameters (dense, conv, etc.)
template <typename T> class ParameterizedLayer : public Layer<T> {
public:
  explicit ParameterizedLayer(const std::string &name = "") {
    this->name_ = name;
  }

  std::vector<Matrix *> parameters() override {
    std::vector<Matrix *> params;
    collect_parameters(params);
    return params;
  }

  std::vector<Matrix *> gradients() override {
    std::vector<Matrix *> grads;
    collect_gradients(grads);
    return grads;
  }

  bool has_parameters() const override { return true; }

  void update_parameters(const Optimizer &optimizer) override {
    update_parameters_impl(optimizer);
  }

protected:
  virtual void collect_parameters(std::vector<Matrix *> &params) = 0;
  virtual void collect_gradients(std::vector<Matrix *> &grads) = 0;
  virtual void update_parameters_impl(const Optimizer &optimizer) = 0;

  Matrix last_input_; // Cache for gradient computation
};

// Dense/Fully Connected Layer
template <typename T> class DenseLayer : public ParameterizedLayer<T> {
private:
  int input_size_;
  int output_size_;
  bool use_bias_;
  std::unique_ptr<ActivationFunction<T>> activation_;
  Matrix weights_;
  Matrix bias_;
  Matrix weight_gradients_;
  Matrix bias_gradients_;
  Matrix pre_activation_output_;  // Store pre-activation values for gradient
                                  // computation
  Matrix post_activation_output_; // Store post-activation values for derivative
                                  // computation

public:
  DenseLayer(int input_size, int output_size,
             std::unique_ptr<ActivationFunction<T>> activation = nullptr,
             bool use_bias = true, const std::string &name = "dense")
      : ParameterizedLayer<T>(name), input_size_(input_size),
        output_size_(output_size), use_bias_(use_bias),
        activation_(std::move(activation)) {
    // Initialize matrices with proper dimensions
    weights_ = Matrix(output_size, input_size, 1);
    weight_gradients_ = Matrix(output_size, input_size, 1);

    if (use_bias_) {
      bias_ = Matrix(output_size, 1, 1);
      bias_gradients_ = Matrix(output_size, 1, 1);
      bias_.fill(0.0);
    }

    // Xavier initialization
    T fan_in = static_cast<T>(input_size);
    T fan_out = static_cast<T>(output_size);
    T std_dev = std::sqrt(T(2.0) / (fan_in + fan_out));
    weights_.fill_random_normal(std_dev);
  }

  Matrix forward(const Matrix &input) override {
    if (input.cols * input.channels != input_size_) {
      throw std::invalid_argument("Input size mismatch in DenseLayer");
    }

    this->last_input_ = input;

    // Flatten input if needed (batch_size, features)
    Matrix flattened_input =
        input.reshape(input.rows, input.cols * input.channels, 1);

    // Linear transformation: output = input * weights^T + bias
    Matrix output(input.rows, output_size_, 1);

    // Perform matrix multiplication: input (batch_size x input_size) *
    // weights^T (input_size x output_size)
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int batch = 0; batch < input.rows; ++batch) {
      for (int out = 0; out < output_size_; ++out) {
        double sum = 0.0;
        for (int in = 0; in < input_size_; ++in) {
          sum += flattened_input(batch, in, 0) * weights_(out, in, 0);
        }
        output(batch, out, 0) = sum;

        // Add bias
        if (use_bias_) {
          output(batch, out, 0) += bias_(out, 0, 0);
        }
      }
    }

    pre_activation_output_ = output;

    // Apply activation if provided and store post-activation values
    if (activation_) {
      activation_->apply(output);
      post_activation_output_ = output;
    } else {
      post_activation_output_ = output;
    }

    return output;
  }

  Matrix backward(const Matrix &grad_output) override {
    Matrix grad_input(this->last_input_.rows, this->last_input_.cols,
                      this->last_input_.channels);

    Matrix current_grad = grad_output;

    // Backprop through activation using stored post-activation values
    if (activation_) {
      Matrix activation_derivative =
          activation_->derivative(post_activation_output_, &current_grad);
      // Element-wise multiplication of grad_output with activation derivative
      for (int i = 0; i < current_grad.size(); ++i) {
        current_grad.data[i] *= activation_derivative.data[i];
      }
    }

    // Flatten the last input for gradient computation
    Matrix flattened_input = this->last_input_.reshape(
        this->last_input_.rows,
        this->last_input_.cols * this->last_input_.channels, 1);

    // Compute weight gradients: dW = grad_output^T * input

    weight_gradients_.resize(output_size_, input_size_, 1);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int out = 0; out < output_size_; ++out) {
      for (int in = 0; in < input_size_; ++in) {
        double grad_sum = 0.0;
        for (int batch = 0; batch < current_grad.rows; ++batch) {
          grad_sum +=
              current_grad(batch, out, 0) * flattened_input(batch, in, 0);
        }
        weight_gradients_(out, in, 0) = grad_sum;
      }
    }

    // Compute bias gradients
    if (use_bias_) {
      bias_gradients_.resize(output_size_, 1, 1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int out = 0; out < output_size_; ++out) {
        double grad_sum = 0.0;
        for (int batch = 0; batch < current_grad.rows; ++batch) {
          grad_sum += current_grad(batch, out, 0);
        }
        bias_gradients_(out, 0, 0) = grad_sum;
      }
    }

    // Compute input gradients: grad_input = grad_output * weights
    Matrix flattened_grad_input(this->last_input_.rows, input_size_, 1);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int batch = 0; batch < current_grad.rows; ++batch) {
      for (int in = 0; in < input_size_; ++in) {
        double grad_sum = 0.0;
        for (int out = 0; out < output_size_; ++out) {
          grad_sum += current_grad(batch, out, 0) * weights_(out, in, 0);
        }
        flattened_grad_input(batch, in, 0) = grad_sum;
      }
    }

    // Reshape back to original input shape
    return flattened_grad_input.reshape(this->last_input_.rows,
                                        this->last_input_.cols,
                                        this->last_input_.channels);
  }

  std::string type() const override { return "dense"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["input_size"] = input_size_;
    config.parameters["output_size"] = output_size_;
    config.parameters["use_bias"] = use_bias_;
    config.parameters["activation"] =
        activation_ ? activation_->name() : std::string("none");
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    auto activation_clone = activation_ ? activation_->clone() : nullptr;
    return std::make_unique<DenseLayer<T>>(input_size_, output_size_,
                                           std::move(activation_clone),
                                           use_bias_, this->name_);
  }

  std::tuple<int, int, int>
  compute_output_shape(int input_rows, int input_cols,
                       int input_channels) const override {
    return std::make_tuple(input_rows, output_size_, 1);
  }

protected:
  void collect_parameters(std::vector<Matrix *> &params) override {
    params.push_back(&weights_);
    if (use_bias_) {
      params.push_back(&bias_);
    }
  }

  void collect_gradients(std::vector<Matrix *> &grads) override {
    grads.push_back(&weight_gradients_);
    if (use_bias_) {
      grads.push_back(&bias_gradients_);
    }
  }

  void update_parameters_impl(const Optimizer &optimizer) override {
    // This would call optimizer.update(weights_, weight_gradients_) etc.
    // weights_ -= learning_rate * weight_gradients_
    // bias_ -= learning_rate * bias_gradients_
  }

private:
  // No need for matrix_multiply helper since Matrix class has operator*
};

// Activation Layer (stateless)
template <typename T> class ActivationLayer : public StatelessLayer<T> {
private:
  std::unique_ptr<ActivationFunction<T>> activation_;
  Matrix
      last_input_; // For gradient computation (will be initialized when used)

public:
  explicit ActivationLayer(std::unique_ptr<ActivationFunction<T>> activation,
                           const std::string &name = "activation")
      : StatelessLayer<T>(name), activation_(std::move(activation)) {
    if (!activation_) {
      throw std::invalid_argument("Activation function cannot be null");
    }
  }

  Matrix forward(const Matrix &input) override {
    last_input_ = input;
    Matrix output = input; // Copy
    activation_->apply(output);
    return output;
  }

  Matrix backward(const Matrix &grad_output) override {
    Matrix grad_input = activation_->derivative(last_input_, &grad_output);
    // Element-wise multiplication with grad_output
    for (int i = 0; i < grad_input.size(); ++i) {
      grad_input.data[i] *= grad_output.data[i];
    }
    return grad_input;
  }

  std::string type() const override { return "activation"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["activation"] = activation_->name();
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<ActivationLayer<T>>(activation_->clone(),
                                                this->name_);
  }

  std::tuple<int, int, int>
  compute_output_shape(int input_rows, int input_cols,
                       int input_channels) const override {
    return std::make_tuple(input_rows, input_cols, input_channels);
  }
};

// Dropout Layer
template <typename T> class DropoutLayer : public StatelessLayer<T> {
private:
  T dropout_rate_;
  Matrix mask_; // Dropout mask (will be initialized when used)
  mutable std::mt19937 generator_; // Thread-safe random generator

public:
  explicit DropoutLayer(T dropout_rate, const std::string &name = "dropout")
      : StatelessLayer<T>(name), dropout_rate_(dropout_rate),
        generator_(std::random_device{}()) {
    if (dropout_rate < T(0) || dropout_rate >= T(1)) {
      throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
  }

  Matrix forward(const Matrix &input) override {
    if (!this->is_training_) {
      return input; // No dropout during inference
    }

    mask_ = Matrix(input.rows, input.cols, input.channels);
    Matrix output = input;

    std::uniform_real_distribution<T> distribution(T(0), T(1));

    // Generate random mask with proper scaling
    T scale = T(1) / (T(1) - dropout_rate_);
    for (int i = 0; i < mask_.size(); ++i) {
      if (distribution(generator_) < dropout_rate_) {
        mask_.data[i] = T(0);
        output.data[i] = T(0);
      } else {
        mask_.data[i] = scale; // Scale remaining values
        output.data[i] *= scale;
      }
    }

    return output;
  }

  Matrix backward(const Matrix &grad_output) override {
    if (!this->is_training_) {
      return grad_output;
    }

    Matrix grad_input = grad_output;
    for (int i = 0; i < grad_input.size(); ++i) {
      grad_input.data[i] *= mask_.data[i];
    }
    return grad_input;
  }

  std::string type() const override { return "dropout"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["dropout_rate"] = dropout_rate_;
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    return std::make_unique<DropoutLayer<T>>(dropout_rate_, this->name_);
  }

  std::tuple<int, int, int>
  compute_output_shape(int input_rows, int input_cols,
                       int input_channels) const override {
    return std::make_tuple(input_rows, input_cols, input_channels);
  }
};

// Layer Factory (following your activation factory pattern)
template <typename T> class LayerFactory {
private:
  static std::unordered_map<
      std::string,
      std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)>>
      creators_;

public:
  static void register_layer(
      const std::string &type,
      std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)> creator) {
    creators_[type] = creator;
  }

  static std::unique_ptr<Layer<T>> create(const std::string &type,
                                          const LayerConfig &config) {
    auto it = creators_.find(type);
    if (it != creators_.end()) {
      return it->second(config);
    }
    throw std::invalid_argument("Unknown layer type: " + type);
  }

  static std::unique_ptr<Layer<T>> create(const LayerConfig &config) {
    return create(config.get<std::string>("type"), config);
  }

  static void register_defaults() {
    // Dense layer
    register_layer(
        "dense", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
          int input_size = config.get<int>("input_size");
          int output_size = config.get<int>("output_size");
          bool use_bias = config.get<bool>("use_bias", true);
          std::string activation_name =
              config.get<std::string>("activation", "none");

          std::unique_ptr<ActivationFunction<T>> activation = nullptr;
          if (activation_name != "none") {
            auto factory = ActivationFactory<T>();
            factory.register_defaults();
            activation = factory.create(activation_name);
          }

          return std::make_unique<DenseLayer<T>>(input_size, output_size,
                                                 std::move(activation),
                                                 use_bias, config.name);
        });

    // Activation layer
    register_layer("activation",
                   [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
                     std::string activation_name =
                         config.get<std::string>("activation");
                     auto factory = ActivationFactory<T>();
                     factory.register_defaults();
                     auto activation = factory.create(activation_name);
                     if (!activation) {
                       throw std::invalid_argument("Unknown activation: " +
                                                   activation_name);
                     }
                     return std::make_unique<ActivationLayer<T>>(
                         std::move(activation), config.name);
                   });

    // Dropout layer
    register_layer(
        "dropout", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
          T dropout_rate = config.get<T>("dropout_rate");
          return std::make_unique<DropoutLayer<T>>(dropout_rate, config.name);
        });
  }

  static std::vector<std::string> available_types() {
    std::vector<std::string> types;
    for (const auto &pair : creators_) {
      types.push_back(pair.first);
    }
    return types;
  }
};

template <typename T>
std::unordered_map<
    std::string, std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)>>
    LayerFactory<T>::creators_;

} // namespace layers

// Convenience functions similar to your Activations namespace
namespace Layers {

template <typename T = double>
std::unique_ptr<layers::Layer<T>>
dense(int input_size, int output_size, const std::string &activation = "none",
      bool use_bias = true, const std::string &name = "dense") {
  std::unique_ptr<ActivationFunction<T>> act = nullptr;
  if (activation != "none") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    act = factory.create(activation);
  }
  return std::make_unique<layers::DenseLayer<T>>(
      input_size, output_size, std::move(act), use_bias, name);
}

template <typename T = double>
std::unique_ptr<layers::Layer<T>>
activation(const std::string &activation_name,
           const std::string &name = "activation") {
  auto factory = ActivationFactory<T>();
  factory.register_defaults();
  auto act = factory.create(activation_name);
  return std::make_unique<layers::ActivationLayer<T>>(std::move(act), name);
}

template <typename T = double>
std::unique_ptr<layers::Layer<T>> dropout(T dropout_rate,
                                          const std::string &name = "dropout") {
  return std::make_unique<layers::DropoutLayer<T>>(dropout_rate, name);
}
} // namespace Layers

/* Usage Example:

// Method 1: Using factory
auto factory = layers::LayerFactory<double>();
factory.register_defaults();

layers::LayerConfig config;
config.name = "hidden1";
config.parameters["type"] = std::string("dense");
config.parameters["input_size"] = 784;
config.parameters["output_size"] = 128;
config.parameters["activation"] = std::string("relu");

auto layer = factory.create(config);

// Method 2: Using convenience functions
auto dense_layer = Layers::dense<double>(784, 128, "relu", true, "hidden1");
auto activation_layer = Layers::activation<double>("sigmoid",
"output_activation"); auto dropout_layer = Layers::dropout<double>(0.5,
"dropout1");

// Method 3: Direct instantiation
auto relu_activation = std::make_unique<ReLU<double>>();
auto dense = std::make_unique<layers::DenseLayer<double>>(784, 128,
std::move(relu_activation));

// Forward pass
Matrix input(32, 784, 1); // Batch of 32 samples, 784 features each
Matrix output = dense->forward(input);

// Backward pass
Matrix grad_output(32, 128, 1);
Matrix grad_input = dense->backward(grad_output);

*/
