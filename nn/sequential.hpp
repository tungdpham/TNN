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
#include "loss.hpp"
#include "optimizers.hpp"

namespace tnn {

// Sequential model for layers
template <typename T = float> class Sequential {
private:
  std::vector<std::unique_ptr<Layer<T>>> layers_;
  std::string name_;
  std::unique_ptr<Optimizer<T>> optimizer_ = nullptr;
  std::unique_ptr<Loss<T>> loss_ = nullptr;
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
    if (other.optimizer_) {
      optimizer_ = other.optimizer_->clone();
    }
    if (other.loss_) {
      loss_ = other.loss_->clone();
    }
  }

  // Move constructor
  Sequential(Sequential &&other) noexcept
      : layers_(std::move(other.layers_)), name_(std::move(other.name_)),
        optimizer_(std::move(other.optimizer_)), loss_(std::move(other.loss_)),
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
      optimizer_ = other.optimizer_ ? other.optimizer_->clone() : nullptr;
      loss_ = other.loss_ ? other.loss_->clone() : nullptr;
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
      optimizer_ = std::move(other.optimizer_);
      loss_ = std::move(other.loss_);
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
  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) {
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

          current_output = layers_[i]->forward(current_output, micro_batch_id);

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
          current_output = layers_[i]->forward(current_output, micro_batch_id);
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
  Tensor<T> backward(const Tensor<T> &grad_output, int micro_batch_id = 0) {
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

          current_grad = layers_[i]->backward(current_grad, micro_batch_id);

          auto end_time = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
              end_time - start_time);
          // Insert at beginning to maintain order (since we're going backwards)
          backward_times_ms_.insert(backward_times_ms_.begin(),
                                    duration.count() / 1000.0);
        } else {
          current_grad = layers_[i]->backward(current_grad, micro_batch_id);
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
  std::vector<Tensor<T> *> parameters() const {
    std::vector<Tensor<T> *> all_params;
    for (auto &layer : layers_) {
      auto layer_params = layer->parameters();
      all_params.insert(all_params.end(), layer_params.begin(),
                        layer_params.end());
    }
    return all_params;
  }

  std::vector<Tensor<T> *> gradients() const {
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
    std::cout << get_config().dump(2) << std::endl;
  }

  // Serialization support
  void save_to_file(const std::string &path) const {
    // Use the new get_config method that includes optimizer and loss
    nlohmann::json config_json = get_config();

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
    // Load configuration from JSON using new serialization system
    std::ifstream config_file(path + ".json");
    if (!config_file.is_open()) {
      throw std::runtime_error("Could not open config file: " + path +
                               ".json");
    }
    nlohmann::json config_json;
    config_file >> config_json;
    config_file.close();

    // Use the new load_from_config method that handles optimizer and loss
    Sequential<T> model = load_from_config(config_json);

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
  void update_parameters() const {
    auto params = parameters();
    auto grads = gradients();
    if (optimizer_) {
      optimizer_->update(params, grads);
    } else {
      throw std::runtime_error("No optimizer set for model: " + name_); 
    }
  }

  void set_optimizer(Optimizer<T> &optimizer) {
    this->optimizer_ = optimizer.clone();
  }
  
  void set_optimizer(std::unique_ptr<Optimizer<T>> optimizer) {
    this->optimizer_ = std::move(optimizer);
    printf("Optimizer set to: %s\n", this->optimizer_->name().c_str());
  }
  
  void set_loss(Loss<T> &loss) {
    this->loss_ = loss.clone();
  }
  
  void set_loss(std::unique_ptr<Loss<T>> loss) {
    this->loss_ = std::move(loss);
  }
  
  Optimizer<T>* get_optimizer() const {
    return optimizer_.get();
  }
  
  Loss<T>* get_loss() const {
    return loss_.get();
  }

  // Split the model into multiple stages (Will optimize later for heterogeneous machines)
  std::vector<Sequential<T>> split(size_t num_stages) const {
    if (num_stages == 0 || num_stages > layers_.size()) {
      throw std::invalid_argument("Invalid number of stages for split");
    } 
    std::vector<Sequential<T>> stages(num_stages);
    size_t stage_size = layers_.size() / num_stages;
    size_t remainder = layers_.size() % num_stages;
    size_t layer_index = 0;
    for (size_t i = 0; i < num_stages; ++i) {
      size_t current_stage_size = stage_size + (i < remainder ? 1 : 0);
      for (size_t j = 0; j < current_stage_size; ++j) {
        if (layer_index >= layers_.size()) {
          throw std::out_of_range("Layer index out of range during split");
        }
        stages[i].add(layers_[layer_index]->clone());
        layer_index++;
      }
      if (this->optimizer_) {
        stages[i].set_optimizer(this->optimizer_->clone());
      }
      if (this->loss_) {
        stages[i].set_loss(this->loss_->clone());
      }
    }
    return stages;
  }

  // Get all layers
  const std::vector<std::unique_ptr<Layer<T>>> &get_layers() const {
    return layers_;
  }
  
  // Serialization methods
  nlohmann::json get_config() const {
    nlohmann::json config;
    config["name"] = name_;
    config["is_training"] = is_training_;
    
    // Serialize layers
    nlohmann::json layers_config = nlohmann::json::array();
    for (const auto& layer : layers_) {
      LayerConfig layer_config = layer->get_config();
      nlohmann::json layer_json;
      layer_json["type"] = layer->type();
      layer_json["name"] = layer_config.name;
      layer_json["parameters"] = nlohmann::json::object();
      
      // Convert std::any parameters to JSON (basic types only)
      for (const auto& [key, value] : layer_config.parameters) {
        try {
          if (auto* ptr = std::any_cast<int>(&value)) {
            layer_json["parameters"][key] = *ptr;
          } else if (auto* ptr = std::any_cast<size_t>(&value)) {
            layer_json["parameters"][key] = *ptr;
          } else if (auto* ptr = std::any_cast<float>(&value)) {
            layer_json["parameters"][key] = *ptr;
          } else if (auto* ptr = std::any_cast<double>(&value)) {
            layer_json["parameters"][key] = *ptr;
          } else if (auto* ptr = std::any_cast<bool>(&value)) {
            layer_json["parameters"][key] = *ptr;
          } else if (auto* ptr = std::any_cast<std::string>(&value)) {
            layer_json["parameters"][key] = *ptr;
          }
        } catch (const std::bad_any_cast&) {
          // Skip parameters that can't be serialized
        }
      }
      layers_config.push_back(layer_json);
    }
    config["layers"] = layers_config;
    
    // Serialize optimizer
    if (optimizer_) {
      printf("Saving optimizer configuration...\n");
      OptimizerConfig opt_config = optimizer_->get_config();
      nlohmann::json opt_json;
      opt_json["type"] = opt_config.type;
      opt_json["name"] = opt_config.name;
      opt_json["parameters"] = nlohmann::json::object();
      
      for (const auto& [key, value] : opt_config.parameters) {
        try {
          if (auto* ptr = std::any_cast<float>(&value)) {
            opt_json["parameters"][key] = *ptr;
          } else if (auto* ptr = std::any_cast<double>(&value)) {
            opt_json["parameters"][key] = *ptr;
          }
        } catch (const std::bad_any_cast&) {
          // Skip parameters that can't be serialized
        }
      }
      config["optimizer"] = opt_json;
    }
    
    // Serialize loss
    if (loss_) {
      LossConfig loss_config = loss_->get_config();
      nlohmann::json loss_json;
      loss_json["type"] = loss_config.type;
      loss_json["name"] = loss_config.name;
      loss_json["parameters"] = nlohmann::json::object();
      
      for (const auto& [key, value] : loss_config.parameters) {
        try {
          if (auto* ptr = std::any_cast<float>(&value)) {
            loss_json["parameters"][key] = *ptr;
          } else if (auto* ptr = std::any_cast<double>(&value)) {
            loss_json["parameters"][key] = *ptr;
          }
        } catch (const std::bad_any_cast&) {
          // Skip parameters that can't be serialized
        }
      }
      config["loss"] = loss_json;
    }
    
    return config;
  }
  
  void save_config(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    file << get_config().dump(2);
    file.close();
  }
  
  static Sequential<T> load_from_config(const nlohmann::json& config) {
    Sequential<T> model(config.value("name", "sequential"));
    model.is_training_ = config.value("is_training", true);
    
    // Load optimizer if present
    if (config.contains("optimizer")) {
      printf("Loading optimizer configuration...\n");
      OptimizerConfig opt_config;
      opt_config.type = config["optimizer"]["type"];
      opt_config.name = config["optimizer"]["name"];
      
      if (config["optimizer"].contains("parameters")) {
        for (const auto& [key, value] : config["optimizer"]["parameters"].items()) {
          if (value.is_number_float()) {
            opt_config.parameters[key] = value.get<float>();
          } else if (value.is_number_integer()) {
            opt_config.parameters[key] = value.get<int>();
          }
        }
      }
      
      model.set_optimizer(OptimizerFactory<T>::create_from_config(opt_config));
    }
    
    // Load loss if present
    if (config.contains("loss")) {
      printf("Loading loss configuration...\n");
      LossConfig loss_config;
      loss_config.type = config["loss"]["type"];
      loss_config.name = config["loss"]["name"];
      
      if (config["loss"].contains("parameters")) {
        for (const auto& [key, value] : config["loss"]["parameters"].items()) {
          if (value.is_number_float()) {
            loss_config.parameters[key] = value.get<float>();
          } else if (value.is_number_integer()) {
            loss_config.parameters[key] = value.get<int>();
          }
        }
      }
      
      model.set_loss(LossFactory<T>::create_from_config(loss_config));
    }
    
    // Load layers using LayerFactory
    if (config.contains("layers")) {
      auto factory = LayerFactory<T>();
      factory.register_defaults();
      
      for (const auto& layer_json : config["layers"]) {
        LayerConfig layer_config;
        layer_config.name = layer_json.value("name", "");
        
        // Convert JSON parameters back to LayerConfig parameters
        if (layer_json.contains("parameters")) {
          for (const auto& [key, value] : layer_json["parameters"].items()) {
            if (value.is_number_integer()) {
              layer_config.parameters[key] = value.get<size_t>();
            } else if (value.is_number_float()) {
              layer_config.parameters[key] = value.get<float>();
            } else if (value.is_boolean()) {
              layer_config.parameters[key] = value.get<bool>();
            } else if (value.is_string()) {
              layer_config.parameters[key] = value.get<std::string>();
            }
          }
        }
        
        std::string layer_type = layer_json.value("type", "");
        auto layer = factory.create(layer_type, layer_config);
        model.add(std::move(layer));
      }
    }
    
    return model;
  }
  
  static Sequential<T> load_from_config_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file for reading: " + filepath);
    }
    
    nlohmann::json config;
    file >> config;
    file.close();
    
    return load_from_config(config);
  }
};

// Builder pattern for easy model construction
template <typename T = float> class SequentialBuilder {
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
    auto layer = tnn::dense<T>(
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
    auto layer = tnn::conv2d<T>(
        in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, activation, use_bias, 
        name.empty() ? "conv2d_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &activation(const std::string &activation_name,
                                      const std::string &name = "") {
    auto layer = tnn::activation<T>(
        activation_name,
        name.empty() ? "activation_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &maxpool2d(size_t pool_h, size_t pool_w,
                                     size_t stride_h = 0, size_t stride_w = 0,
                                     size_t pad_h = 0, size_t pad_w = 0,
                                     const std::string &name = "") {
    auto layer = tnn::maxpool2d<T>(
        pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
        name.empty() ? "maxpool2d_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &dropout(T dropout_rate,
                                   const std::string &name = "") {
    auto layer = tnn::dropout<T>(
        dropout_rate,
        name.empty() ? "dropout_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &batchnorm(size_t num_features, T epsilon = T(1e-5),
                                       T momentum = T(0.1), bool affine = true,
                                       const std::string &name = "") {
    auto layer = tnn::batchnorm<T>(
        num_features, epsilon, momentum, affine,
        name.empty() ? "batchnorm_" + std::to_string(model_.size()) : name);
    model_.add(std::move(layer));
    return *this;
  }

  SequentialBuilder &flatten(const std::string &name = "") {
    auto layer = tnn::flatten<T>(
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
