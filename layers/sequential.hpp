#pragma once
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "layer.hpp"

namespace models {

template <typename T> class Sequential {
private:
  std::vector<std::unique_ptr<layers::Layer<T>>> layers_;
  std::string name_;
  bool compiled_ = false;

public:
  explicit Sequential(const std::string &name = "sequential") : name_(name) {}

  // Move constructor and assignment
  Sequential(Sequential &&other) noexcept
      : layers_(std::move(other.layers_)), name_(std::move(other.name_)),
        compiled_(other.compiled_) {}

  Sequential &operator=(Sequential &&other) noexcept {
    if (this != &other) {
      layers_ = std::move(other.layers_);
      name_ = std::move(other.name_);
      compiled_ = other.compiled_;
    }
    return *this;
  }

  // Disable copy constructor and assignment
  Sequential(const Sequential &) = delete;
  Sequential &operator=(const Sequential &) = delete;

  // Add layers
  Sequential &add(std::unique_ptr<layers::Layer<T>> layer) {
    layers_.push_back(std::move(layer));
    compiled_ = false; // Need to recompile after adding layers
    return *this;
  }

  // Convenience method to add multiple layers
  template <typename... Layers>
  Sequential &add(std::unique_ptr<layers::Layer<T>> first, Layers... rest) {
    add(std::move(first));
    if constexpr (sizeof...(rest) > 0) {
      add(std::move(rest)...);
    }
    return *this;
  }

  // Build the model (validate connections)
  void compile(int input_rows, int input_cols, int input_channels) {
    if (layers_.empty()) {
      throw std::runtime_error("Cannot compile empty model");
    }

    int current_rows = input_rows;
    int current_cols = input_cols;
    int current_channels = input_channels;

    for (size_t i = 0; i < layers_.size(); ++i) {
      auto [out_rows, out_cols, out_channels] =
          layers_[i]->compute_output_shape(current_rows, current_cols,
                                           current_channels);

      current_rows = out_rows;
      current_cols = out_cols;
      current_channels = out_channels;
    }

    compiled_ = true;
  }

  // Forward pass
  Matrix<T> forward(const Matrix<T> &input) {
    if (!compiled_) {
      throw std::runtime_error("Model must be compiled before forward pass");
    }

    Matrix<T> current = input;
    for (auto &layer : layers_) {
      current = layer->forward(current);
    }
    return current;
  }

  // Backward pass
  Matrix<T> backward(const Matrix<T> &grad_output) {
    if (!compiled_) {
      throw std::runtime_error("Model must be compiled before backward pass");
    }

    Matrix<T> current_grad = grad_output;

    // Backpropagate through layers in reverse order
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
      current_grad = (*it)->backward(current_grad);
    }

    return current_grad;
  }

  // Training mode
  void set_training(bool training) {
    for (auto &layer : layers_) {
      layer->set_training(training);
    }
  }

  void train() { set_training(true); }

  void eval() { set_training(false); }

  // Parameter access
  std::vector<Matrix<T> *> parameters() {
    std::vector<Matrix<T> *> all_params;
    for (auto &layer : layers_) {
      auto layer_params = layer->parameters();
      all_params.insert(all_params.end(), layer_params.begin(),
                        layer_params.end());
    }
    return all_params;
  }

  std::vector<Matrix<T> *> gradients() {
    std::vector<Matrix<T> *> all_grads;
    for (auto &layer : layers_) {
      auto layer_grads = layer->gradients();
      all_grads.insert(all_grads.end(), layer_grads.begin(), layer_grads.end());
    }
    return all_grads;
  }

  size_t parameter_count() const {
    size_t count = 0;
    for (const auto &layer : layers_) {
      if (layer->has_parameters()) {
        auto params = const_cast<layers::Layer<T> *>(layer.get())->parameters();
        for (const auto *param : params) {
          count += param->size();
        }
      }
    }
    return count;
  }

  // Model introspection
  size_t size() const { return layers_.size(); }

  bool empty() const { return layers_.empty(); }

  layers::Layer<T> *operator[](size_t index) {
    if (index >= layers_.size()) {
      throw std::out_of_range("Layer index out of range");
    }
    return layers_[index].get();
  }

  const layers::Layer<T> *operator[](size_t index) const {
    if (index >= layers_.size()) {
      throw std::out_of_range("Layer index out of range");
    }
    return layers_[index].get();
  }

  // Model summary
  std::string summary() const {
    std::stringstream ss;
    ss << "Model: " << name_ << "\n";
    ss << "================================================================\n";
    ss << "Layer (type)                 Output Shape              Param #   \n";
    ss << "================================================================\n";

    size_t total_params = 0;
    // int current_rows = -1, current_cols = -1, current_channels = -1;

    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto &layer = layers_[i];

      // Get layer info
      auto config = layer->get_config();
      std::string layer_name = config.name.empty()
                                   ? layer->type() + "_" + std::to_string(i)
                                   : config.name;

      // Calculate output shape (this is a simplified version)
      std::string output_shape = "Unknown";
      if (compiled_ && i == 0) {
        // For the first layer, we'd need input shape info
        output_shape = "Need input shape";
      }

      // Count parameters
      size_t layer_params = 0;
      if (layer->has_parameters()) {
        auto params = const_cast<layers::Layer<T> *>(layer.get())->parameters();
        for (const auto *param : params) {
          layer_params += param->size();
        }
      }
      total_params += layer_params;

      ss << std::left << std::setw(30)
         << (layer_name + " (" + layer->type() + ")") << std::setw(25)
         << output_shape << std::setw(10) << layer_params << "\n";
    }

    ss << "================================================================\n";
    ss << "Total params: " << total_params << "\n";
    ss << "================================================================\n";

    return ss.str();
  }

  // Clone the entire model
  std::unique_ptr<Sequential<T>> clone() const {
    auto cloned = std::make_unique<Sequential<T>>(name_ + "_clone");

    for (const auto &layer : layers_) {
      cloned->add(layer->clone());
    }

    cloned->compiled_ = compiled_;
    return cloned;
  }

  // Save/Load configuration (simplified)
  std::vector<layers::LayerConfig> get_config() const {
    std::vector<layers::LayerConfig> configs;
    for (const auto &layer : layers_) {
      configs.push_back(layer->get_config());
    }
    return configs;
  }

  static std::unique_ptr<Sequential<T>>
  from_config(const std::vector<layers::LayerConfig> &configs,
              const std::string &name = "sequential") {
    auto model = std::make_unique<Sequential<T>>(name);
    auto factory = layers::LayerFactory<T>();
    factory.register_defaults();

    for (const auto &config : configs) {
      std::string layer_type = config.get<std::string>("type");
      model->add(factory.create(layer_type, config));
    }

    return model;
  }
};

} // namespace models

/* Usage Example:

// Create a simple neural network
auto model = std::make_unique<models::Sequential<double>>("mnist_classifier");

model->add(Layers::dense<double>(784, 128, "relu", true, "hidden1"))
     .add(Layers::dropout<double>(0.3, "dropout1"))
     .add(Layers::dense<double>(128, 64, "relu", true, "hidden2"))
     .add(Layers::dropout<double>(0.3, "dropout2"))
     .add(Layers::dense<double>(64, 10, "softmax", true, "output"));

// Compile the model
model->compile(1, 784, 1); // Input: batch_size x 784 features

// Print model summary
std::cout << model->summary() << std::endl;

// Training loop
model->train();
Matrix<T> input(32, 784, 1);  // Batch of 32 samples
Matrix<T> target(32, 10, 1);  // One-hot encoded targets

// Forward pass
Matrix<T> prediction = model->forward(input);

// Compute loss gradient (you'd implement this based on your loss function)
Matrix<T> loss_grad = compute_loss_gradient(prediction, target);

// Backward pass
Matrix<T> input_grad = model->backward(loss_grad);

// Update parameters (you'd implement an optimizer)
auto params = model->parameters();
auto grads = model->gradients();
// optimizer.update(params, grads);

// Inference
model->eval();
Matrix<T> test_input(1, 784, 1);
Matrix<T> test_output = model->forward(test_input);

*/
