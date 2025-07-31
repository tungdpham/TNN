#pragma once
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers.hpp"

namespace layers {

// Sequential model for layers
template <typename T = double> class Sequential {
private:
  std::vector<std::unique_ptr<Layer<T>>> layers_;
  std::string name_;
  bool is_training_;

  // Cache for intermediate outputs during forward pass (useful for debugging)
  std::vector<Tensor<T>> layer_outputs_;

  // Performance profiling
  bool enable_profiling_;
  std::vector<double> forward_times_ms_;
  std::vector<double> backward_times_ms_;
  std::vector<std::string> layer_names_;

public:
  explicit Sequential(const std::string &name = "sequential")
      : name_(name), is_training_(true), enable_profiling_(false) {}

  // Copy constructor (deep copy)
  Sequential(const Sequential &other)
      : name_(other.name_), is_training_(other.is_training_),
        enable_profiling_(other.enable_profiling_) {
    for (const auto &layer : other.layers_) {
      layers_.push_back(layer->clone());
    }
  }

  // Move constructor
  Sequential(Sequential &&other) noexcept
      : layers_(std::move(other.layers_)), name_(std::move(other.name_)),
        is_training_(other.is_training_),
        layer_outputs_(std::move(other.layer_outputs_)),
        enable_profiling_(other.enable_profiling_),
        forward_times_ms_(std::move(other.forward_times_ms_)),
        backward_times_ms_(std::move(other.backward_times_ms_)),
        layer_names_(std::move(other.layer_names_)) {}

  // Assignment operators
  Sequential &operator=(const Sequential &other) {
    if (this != &other) {
      layers_.clear();
      for (const auto &layer : other.layers_) {
        layers_.push_back(layer->clone());
      }
      name_ = other.name_;
      is_training_ = other.is_training_;
      enable_profiling_ = other.enable_profiling_;
      layer_outputs_.clear();
      forward_times_ms_.clear();
      backward_times_ms_.clear();
      layer_names_.clear();
    }
    return *this;
  }

  Sequential &operator=(Sequential &&other) noexcept {
    if (this != &other) {
      layers_ = std::move(other.layers_);
      name_ = std::move(other.name_);
      is_training_ = other.is_training_;
      layer_outputs_ = std::move(other.layer_outputs_);
      enable_profiling_ = other.enable_profiling_;
      forward_times_ms_ = std::move(other.forward_times_ms_);
      backward_times_ms_ = std::move(other.backward_times_ms_);
      layer_names_ = std::move(other.layer_names_);
    }
    return *this;
  }

  // Layer management
  void add(std::unique_ptr<Layer<T>> layer) {
    if (!layer) {
      throw std::invalid_argument("Cannot add null layer");
    }
    layer->set_training(is_training_);
    layers_.push_back(std::move(layer));
  }

  template <typename LayerType, typename... Args>
  void add_layer(Args &&...args) {
    auto layer = std::make_unique<LayerType>(std::forward<Args>(args)...);
    add(std::move(layer));
  }

  void insert(size_t index, std::unique_ptr<Layer<T>> layer) {
    if (!layer) {
      throw std::invalid_argument("Cannot insert null layer");
    }
    if (index > layers_.size()) {
      throw std::out_of_range("Insert index out of range");
    }
    layer->set_training(is_training_);
    layers_.insert(layers_.begin() + index, std::move(layer));
  }

  void remove(size_t index) {
    if (index >= layers_.size()) {
      throw std::out_of_range("Remove index out of range");
    }
    layers_.erase(layers_.begin() + index);
  }

  void clear() {
    layers_.clear();
    layer_outputs_.clear();
    forward_times_ms_.clear();
    backward_times_ms_.clear();
    layer_names_.clear();
  }

  // Access layers
  Layer<T> &operator[](size_t index) {
    if (index >= layers_.size()) {
      throw std::out_of_range("Layer index out of range");
    }
    return *layers_[index];
  }

  const Layer<T> &operator[](size_t index) const {
    if (index >= layers_.size()) {
      throw std::out_of_range("Layer index out of range");
    }
    return *layers_[index];
  }

  Layer<T> &at(size_t index) { return operator[](index); }

  const Layer<T> &at(size_t index) const { return operator[](index); }

  size_t size() const { return layers_.size(); }

  bool empty() const { return layers_.empty(); }

  // Training mode
  void set_training(bool training) {
    is_training_ = training;
    for (auto &layer : layers_) {
      layer->set_training(training);
    }
  }

  bool is_training() const { return is_training_; }

  void train() { set_training(true); }

  void eval() { set_training(false); }

  // Performance profiling methods
  void enable_profiling(bool enable = true) {
    enable_profiling_ = enable;
    if (enable) {
      forward_times_ms_.reserve(layers_.size());
      backward_times_ms_.reserve(layers_.size());
      layer_names_.reserve(layers_.size());
    }
  }

  bool is_profiling_enabled() const { return enable_profiling_; }

  void clear_profiling_data() {
    forward_times_ms_.clear();
    backward_times_ms_.clear();
    layer_names_.clear();
  }

  // Get profiling results
  const std::vector<double> &get_forward_times() const {
    return forward_times_ms_;
  }

  const std::vector<double> &get_backward_times() const {
    return backward_times_ms_;
  }

  const std::vector<std::string> &get_layer_names() const {
    return layer_names_;
  }

  void print_profiling_summary() const {
    if (!enable_profiling_ || forward_times_ms_.empty()) {
      std::cout << "No profiling data available. Enable profiling with "
                   "enable_profiling(true)\n";
      return;
    }

    std::cout << "\n==========================================================="
                 "======\n";
    std::cout << "Performance Profile: " << name_ << "\n";
    std::cout << "============================================================="
                 "====\n";
    std::cout << std::left << std::setw(20) << "Layer" << std::setw(15)
              << "Forward (ms)" << std::setw(15) << "Backward (ms)"
              << std::setw(15) << "Total (ms)" << "\n";
    std::cout << "-------------------------------------------------------------"
                 "----\n";

    double total_forward = 0.0, total_backward = 0.0;

    for (size_t i = 0; i < forward_times_ms_.size(); ++i) {
      double forward_time = forward_times_ms_[i];
      double backward_time =
          (i < backward_times_ms_.size()) ? backward_times_ms_[i] : 0.0;
      double total_time = forward_time + backward_time;

      total_forward += forward_time;
      total_backward += backward_time;

      std::cout << std::left << std::setw(20) << layer_names_[i]
                << std::setw(15) << std::fixed << std::setprecision(3)
                << forward_time << std::setw(15) << std::fixed
                << std::setprecision(3) << backward_time << std::setw(15)
                << std::fixed << std::setprecision(3) << total_time << "\n";
    }

    std::cout << "-------------------------------------------------------------"
                 "----\n";
    std::cout << std::left << std::setw(20) << "TOTAL" << std::setw(15)
              << std::fixed << std::setprecision(3) << total_forward
              << std::setw(15) << std::fixed << std::setprecision(3)
              << total_backward << std::setw(15) << std::fixed
              << std::setprecision(3) << (total_forward + total_backward)
              << "\n";
    std::cout << "============================================================="
                 "====\n\n";
  }

  // Forward pass
  Tensor<T> forward(const Tensor<T> &input) {
    if (layers_.empty()) {
      throw std::runtime_error("Cannot forward through empty sequential model");
    }

    // Clear previous outputs and profiling data
    layer_outputs_.clear();
    layer_outputs_.reserve(layers_.size() + 1);

    if (enable_profiling_) {
      forward_times_ms_.clear();
      forward_times_ms_.reserve(layers_.size());
      layer_names_.clear();
      layer_names_.reserve(layers_.size());
    }

    // Store input
    layer_outputs_.push_back(input);

    Tensor<T> current_output = input;

    for (size_t i = 0; i < layers_.size(); ++i) {
      try {
        if (enable_profiling_) {
          auto start_time = std::chrono::high_resolution_clock::now();

          current_output = layers_[i]->forward(current_output);

          auto end_time = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
              end_time - start_time);
          forward_times_ms_.push_back(duration.count() / 1000.0);

          // Store layer name for profiling
          std::string layer_name = layers_[i]->type();
          auto config = layers_[i]->get_config();
          if (!config.name.empty()) {
            layer_name = config.name;
          }
          layer_names_.push_back(layer_name);
        } else {
          current_output = layers_[i]->forward(current_output);
        }

        layer_outputs_.push_back(current_output);
      } catch (const std::exception &e) {
        throw std::runtime_error("Error in layer " + std::to_string(i) + " (" +
                                 layers_[i]->type() + "): " + e.what());
      }
    }

    return current_output;
  }

  // Backward pass
  Tensor<T> backward(const Tensor<T> &grad_output) {
    if (layers_.empty()) {
      throw std::runtime_error(
          "Cannot backward through empty sequential model");
    }

    if (layer_outputs_.empty()) {
      throw std::runtime_error("Must call forward before backward");
    }

    if (enable_profiling_) {
      backward_times_ms_.clear();
      backward_times_ms_.reserve(layers_.size());
    }

    Tensor<T> current_grad = grad_output;

    // Backward through layers in reverse order
    for (int i = layers_.size() - 1; i >= 0; --i) {
      try {
        if (enable_profiling_) {
          auto start_time = std::chrono::high_resolution_clock::now();

          current_grad = layers_[i]->backward(current_grad);

          auto end_time = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
              end_time - start_time);
          // Insert at beginning to maintain order (since we're going backwards)
          backward_times_ms_.insert(backward_times_ms_.begin(),
                                    duration.count() / 1000.0);
        } else {
          current_grad = layers_[i]->backward(current_grad);
        }
      } catch (const std::exception &e) {
        throw std::runtime_error("Error in backward pass of layer " +
                                 std::to_string(i) + " (" + layers_[i]->type() +
                                 "): " + e.what());
      }
    }

    return current_grad;
  }

  // Prediction (convenience method for inference)
  Tensor<T> predict(const Tensor<T> &input) {
    bool was_training = is_training_;
    set_training(false);
    Tensor<T> result = forward(input);
    set_training(was_training);
    return result;
  }

  // Parameter management
  std::vector<Tensor<T> *> parameters() {
    std::vector<Tensor<T> *> all_params;
    for (auto &layer : layers_) {
      auto layer_params = layer->parameters();
      all_params.insert(all_params.end(), layer_params.begin(),
                        layer_params.end());
    }
    return all_params;
  }

  std::vector<Tensor<T> *> gradients() {
    std::vector<Tensor<T> *> all_grads;
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
        auto params = const_cast<Layer<T> *>(layer.get())->parameters();
        for (const auto &param : params) {
          count += param->size();
        }
      }
    }
    return count;
  }

  // Shape inference
  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const {
    if (layers_.empty()) {
      return input_shape;
    }

    std::vector<size_t> current_shape = input_shape;

    for (const auto &layer : layers_) {
      current_shape = layer->compute_output_shape(current_shape);
    }

    return current_shape;
  }

  // Get intermediate outputs (for debugging)
  const std::vector<Tensor<T>> &get_layer_outputs() const {
    return layer_outputs_;
  }

  Tensor<T> get_layer_output(size_t layer_index) const {
    if (layer_index >= layer_outputs_.size()) {
      throw std::out_of_range("Layer output index out of range");
    }
    return layer_outputs_[layer_index];
  }

  // Model information
  void print_summary(const std::vector<size_t> &input_shape = {}) const {
    std::cout << "============================================================="
                 "====\n";
    std::cout << "Model: " << name_ << "\n";
    std::cout << "============================================================="
                 "====\n";
    std::cout << std::left << std::setw(20) << "Layer (type)" << std::setw(25)
              << "Output Shape" << std::setw(15) << "Param #" << "\n";
    std::cout << "============================================================="
                 "====\n";

    std::vector<size_t> current_shape = input_shape;
    size_t total_params = 0;

    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto &layer = layers_[i];

      // Compute output shape
      if (!current_shape.empty()) {
        current_shape = layer->compute_output_shape(current_shape);
      }

      // Count parameters
      size_t layer_params = 0;
      if (layer->has_parameters()) {
        auto params = const_cast<Layer<T> *>(layer.get())->parameters();
        for (const auto &param : params) {
          layer_params += param->size();
        }
      }
      total_params += layer_params;

      // Layer name and type
      std::string layer_name = layer->type();
      if (!layer->get_config().name.empty()) {
        layer_name = layer->get_config().name + " (" + layer->type() + ")";
      }

      // Output shape string
      std::string shape_str = "(";
      if (!current_shape.empty()) {
        for (size_t j = 0; j < current_shape.size(); ++j) {
          if (j > 0)
            shape_str += ", ";
          shape_str += std::to_string(current_shape[j]);
        }
      } else {
        shape_str += "Unknown";
      }
      shape_str += ")";

      std::cout << std::left << std::setw(20) << layer_name << std::setw(25)
                << shape_str << std::setw(15) << layer_params << "\n";
    }

    std::cout << "============================================================="
                 "====\n";
    std::cout << "Total params: " << total_params << "\n";
    std::cout << "============================================================="
                 "====\n";
  }

  void print_config() const {
    std::cout << "Sequential Configuration:\n";
    std::cout << "Name: " << name_ << "\n";
    std::cout << "Training: " << (is_training_ ? "True" : "False") << "\n";
    std::cout << "Layers: " << layers_.size() << "\n\n";

    for (size_t i = 0; i < layers_.size(); ++i) {
      auto config = layers_[i]->get_config();
      std::cout << "Layer " << i << ": " << config.name
                << " (type: " << layers_[i]->type() << ")\n";

      // Print layer-specific parameters
      if (!config.parameters.empty()) {
        std::cout << "  Parameters:\n";
        for (const auto &[key, value] : config.parameters) {
          std::cout << "    " << key << ": ";

          // Try to extract common types
          try {
            if (auto int_val = std::any_cast<int>(&value)) {
              std::cout << *int_val;
            } else if (auto size_val = std::any_cast<size_t>(&value)) {
              std::cout << *size_val;
            } else if (auto double_val = std::any_cast<double>(&value)) {
              std::cout << *double_val;
            } else if (auto float_val = std::any_cast<float>(&value)) {
              std::cout << *float_val;
            } else if (auto bool_val = std::any_cast<bool>(&value)) {
              std::cout << (*bool_val ? "true" : "false");
            } else if (auto str_val = std::any_cast<std::string>(&value)) {
              std::cout << "\"" << *str_val << "\"";
            } else {
              std::cout << "[complex type]";
            }
          } catch (...) {
            std::cout << "[unknown type]";
          }
          std::cout << "\n";
        }
      }
      std::cout << "\n";
    }
  }

  // Serialization support
  void save_to_file(const std::string &path) const {
    // Save configuration to JSON
    nlohmann::json config_json;
    config_json["name"] = name_;
    config_json["training"] = is_training_;
    config_json["profiling"] = enable_profiling_;
    config_json["layers"] = nlohmann::json::array();

    for (const auto &layer : layers_) {
      auto config = layer->get_config();
      nlohmann::json layer_json;
      layer_json["type"] = layer->type();
      layer_json["name"] = config.name;
      // A basic way to handle std::any parameters
      // This might need to be more robust depending on the types stored in
      // `any`
      for (const auto &p : config.parameters) {
        if (p.second.type() == typeid(size_t)) {
          layer_json[p.first] = std::any_cast<size_t>(p.second);
        } else if (p.second.type() == typeid(int)) {
          layer_json[p.first] = std::any_cast<int>(p.second);
        } else if (p.second.type() == typeid(double)) {
          layer_json[p.first] = std::any_cast<double>(p.second);
        } else if (p.second.type() == typeid(float)) {
          layer_json[p.first] = std::any_cast<float>(p.second);
        } else if (p.second.type() == typeid(bool)) {
          layer_json[p.first] = std::any_cast<bool>(p.second);
        } else if (p.second.type() == typeid(std::string)) {
          layer_json[p.first] = std::any_cast<std::string>(p.second);
        }
      }
      config_json["layers"].push_back(layer_json);
    }

    std::ofstream config_file(path + ".json");
    config_file << config_json.dump(4);
    config_file.close();

    // Save weights to a binary file
    std::ofstream weights_file(path + ".bin", std::ios::binary);
    for (const auto &layer : layers_) {
      if (layer->has_parameters()) {
        auto params =
            const_cast<Layer<T> *>(layer.get())->parameters();
        for (const auto &param : params) {
          param->save(weights_file);
        }
      }
    }
    weights_file.close();
  }

  static Sequential<T> from_file(const std::string &path) {
    // Load configuration from JSON
    std::ifstream config_file(path + ".json");
    if (!config_file.is_open()) {
      throw std::runtime_error("Could not open config file: " + path +
                               ".json");
    }
    nlohmann::json config_json;
    config_file >> config_json;
    config_file.close();

    Sequential<T> model(config_json["name"]);
    model.set_training(config_json["training"]);
    model.enable_profiling(config_json["profiling"]);

    auto factory = LayerFactory<T>();
    factory.register_defaults();

    for (const auto &layer_json : config_json["layers"]) {
      LayerConfig config;
      config.name = layer_json["name"];
      for (auto it = layer_json.begin(); it != layer_json.end(); ++it) {
        if (it.key() != "type" && it.key() != "name") {
          if (it.value().is_number_integer()) {
            config.parameters[it.key()] = it.value().get<size_t>();
          } else if (it.value().is_number_float()) {
            config.parameters[it.key()] = it.value().get<double>();
          } else if (it.value().is_boolean()) {
            config.parameters[it.key()] = it.value().get<bool>();
          } else if (it.value().is_string()) {
            config.parameters[it.key()] = it.value().get<std::string>();
          }
        }
      }
      auto layer = factory.create(layer_json["type"], config);
      model.add(std::move(layer));
    }

    // Load weights from binary file
    std::ifstream weights_file(path + ".bin", std::ios::binary);
    if (!weights_file.is_open()) {
      throw std::runtime_error("Could not open weights file: " + path +
                               ".bin");
    }
    for (auto &layer : model.layers_) {
      if (layer->has_parameters()) {
        auto params = layer->parameters();
        for (auto &param : params) {
          *param = Tensor<T>::load(weights_file);
        }
      }
    }
    weights_file.close();

    return model;
  }

  // Serialization support (basic configuration)
  std::vector<LayerConfig> get_all_configs() const {
    std::vector<LayerConfig> configs;
    configs.reserve(layers_.size());

    for (const auto &layer : layers_) {
      configs.push_back(layer->get_config());
    }

    return configs;
  }

  // Static method to create from configs
  static Sequential
  from_configs(const std::vector<LayerConfig> &configs,
               const std::string &name = "sequential") {
    Sequential model(name);

    auto factory = LayerFactory<T>();
    factory.register_defaults();

    for (const auto &config : configs) {
      auto layer = factory.create(config);
      model.add(std::move(layer));
    }

    return model;
  }

  // Clone the entire model
  std::unique_ptr<Sequential> clone() const {
    auto cloned = std::make_unique<Sequential>(name_);
    cloned->set_training(is_training_);

    for (const auto &layer : layers_) {
      cloned->add(layer->clone());
    }

    return cloned;
  }

  // Model name
  const std::string &name() const { return name_; }

  void set_name(const std::string &name) { name_ = name; }

  // Iterator support
  auto begin() { return layers_.begin(); }

  auto end() { return layers_.end(); }

  auto begin() const { return layers_.begin(); }

  auto end() const { return layers_.end(); }

  // Update all parameters using an optimizer
  template <typename OptimizerType>
  void update_parameters(const OptimizerType &optimizer) {
    for (auto &layer : layers_) {
      if (layer->has_parameters()) {
        layer->update_parameters(optimizer);
      }
    }
  }
};

// Builder pattern for easy model construction
template <typename T = double> class SequentialBuilder {
private:
  Sequential<T> model_;

public:
  explicit SequentialBuilder(
      const std::string &name = "sequential")
      : model_(name) {}

  SequentialBuilder &dense(size_t input_features,
                                      size_t output_features,
                                      const std::string &activation = "none",
                                      bool use_bias = true,
                                      const std::string &name = "") {
    auto layer = Layers::dense<T>(
        input_features, output_features, activation, use_bias,
        name.empty() ? "dense_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &conv2d(size_t in_channels, size_t out_channels,
                                       size_t kernel_h, size_t kernel_w,
                                       size_t stride_h = 1, size_t stride_w = 1,
                                       size_t pad_h = 0, size_t pad_w = 0,
                                       const std::string &activation = "none",
                                       bool use_bias = true,
                                       const std::string &name = "") {
    auto layer = Layers::conv2d<T>(
        in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, activation, use_bias, 
        name.empty() ? "conv2d_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &activation(const std::string &activation_name,
                                      const std::string &name = "") {
    auto layer = Layers::activation<T>(
        activation_name,
        name.empty() ? "activation_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &maxpool2d(size_t pool_h, size_t pool_w,
                                     size_t stride_h = 0, size_t stride_w = 0,
                                     size_t pad_h = 0, size_t pad_w = 0,
                                     const std::string &name = "") {
    auto layer = Layers::maxpool2d<T>(
        pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
        name.empty() ? "maxpool2d_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &dropout(T dropout_rate,
                                   const std::string &name = "") {
    auto layer = Layers::dropout<T>(
        dropout_rate,
        name.empty() ? "dropout_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &flatten(const std::string &name = "") {
    auto layer = Layers::flatten<T>(
        name.empty() ? "flatten_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &add_layer(std::unique_ptr<Layer<T>> layer) {
    model_.add(std::move(layer));
    return *this;
  }

  Sequential<T> build() { return std::move(model_); }

  Sequential<T> &get_model() { return model_; }
};

}
