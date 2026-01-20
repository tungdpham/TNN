/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/base_layer.hpp"
#include "profiling/event.hpp"
#include "tensor/tensor.hpp"

#include "cuda/error_handler.hpp"
#include <cstddef>
#include <fmt/core.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace tnn {
struct Partition {
  size_t start_layer;
  size_t end_layer; // exclusive

  Partition(size_t start, size_t end) : start_layer(start), end_layer(end) {}
};

class Sequential : public Layer {
private:
  std::vector<std::unique_ptr<Layer>> layers_;
  size_t max_size_ = 0;

  void compute_max_size(const std::vector<size_t> &input_shape, DType_t dtype) {
    std::vector<size_t> current_shape = input_shape;
    size_t current_max = std::accumulate(current_shape.begin(), current_shape.end(),
                                         get_dtype_size(dtype), std::multiplies<size_t>());
    for (const auto &layer : layers_) {
      current_shape = layer->compute_output_shape(current_shape);
      size_t activation_size = std::accumulate(current_shape.begin(), current_shape.end(),
                                               get_dtype_size(dtype), std::multiplies<size_t>());
      current_max = std::max(current_max, activation_size);
    }
    max_size_ = current_max;
  }

protected:
  void init_impl() override {
    for (auto &layer : layers_) {
      layer->init();
    }
  }

  void forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id = 0) override {
    if (layers_.empty()) {
      throw std::runtime_error("Cannot forward through empty sequential model");
    }
    auto start = Clock::now();
    compute_max_size(input->shape(), input->data_type());

    Tensor current_input = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
      try {
        auto start = Clock::now();
        output = this->get_buffer(layers_[i]->compute_output_shape(current_input->shape()),
                                  input->data_type());
        layers_[i]->forward(current_input, output, micro_batch_id);
        current_input = output;
        auto end = Clock::now();
        this->profiler_.add_event(
            Event{EventType::COMPUTE, start, end, layers_[i]->name() + " forward"});
      } catch (const std::exception &e) {
        throw std::runtime_error("Error while forward in layer " + std::to_string(i) + " (" +
                                 layers_[i]->name() + "): " + e.what());
      }
    }
    this->device_->getFlow("default")->synchronize();
    auto end = Clock::now();
    this->profiler_.add_event(Event{EventType::COMPUTE, start, end, "Sequential forward"});
  }

  void backward_impl(const Tensor &gradient, Tensor &grad_input,
                     size_t micro_batch_id = 0) override {
    if (layers_.empty()) {
      throw std::runtime_error("Cannot backward through empty sequential model");
    }
    auto start = Clock::now();
    const Tensor *current_gradient = &gradient;
    Tensor temp_output = this->get_buffer({max_size_}, gradient->data_type());
    Tensor temp = this->get_buffer({max_size_}, gradient->data_type());
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
      try {
        auto start = Clock::now();
        layers_[i]->backward(*current_gradient, temp, micro_batch_id);
        std::swap(temp, temp_output);
        current_gradient = &temp_output;
        auto end = Clock::now();
        this->profiler_.add_event(
            Event{EventType::COMPUTE, start, end, layers_[i]->name() + " backward"});
      } catch (const std::exception &e) {
        throw std::runtime_error("Error in backward pass of layer " + std::to_string(i) + " (" +
                                 layers_[i]->type() + "): " + e.what());
      }
    }

    grad_input->ensure((*current_gradient)->shape());
    (*current_gradient)->copy_to(grad_input);
    this->device_->getFlow("default")->synchronize();
    auto end = Clock::now();
    this->profiler_.add_event(Event{EventType::COMPUTE, start, end, "Sequential backward"});
  }

  void on_set_device(const Device &device) override {
    for (auto &layer : layers_) {
      layer->set_device(device);
    }
  }

  void on_set_training(bool training) override {
    for (auto &layer : layers_) {
      layer->set_training(training);
    }
  }

public:
  explicit Sequential(const std::string &name = "seq",
                      std::vector<std::unique_ptr<Layer>> layers = {})
      : Layer() {
    this->name_ = name;
    layers_ = std::move(layers);
  }

  Sequential(const Sequential &) = delete;
  Sequential &operator=(const Sequential &) = delete;

  Sequential(Sequential &&) = default;
  Sequential &operator=(Sequential &&) = default;

  const Device *get_device() const { return this->device_; }

  /**
   * @brief Returns a vector of pointers to all params in the model
   */
  std::vector<Tensor> parameters() override {
    std::vector<Tensor> all_params;
    for (auto &layer : layers_) {
      auto layer_params = layer->parameters();
      all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
    }
    return all_params;
  }

  /**
   * @brief Returns a vector of pointers to all gradients in the model
   */
  std::vector<Tensor> gradients() override {
    std::vector<Tensor> all_grads;
    for (auto &layer : layers_) {
      auto layer_grads = layer->gradients();
      all_grads.insert(all_grads.end(), layer_grads.begin(), layer_grads.end());
    }
    return all_grads;
  }

  /**
   * @brief Returns the output shape for given input shape
   * @param input_shape The shape of the input tensor as a vector of sizes.
   */
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override {
    if (layers_.empty()) {
      return input_shape;
    }

    std::vector<size_t> current_shape = input_shape;
    for (const auto &layer : layers_) {
      current_shape = layer->compute_output_shape(current_shape);
    }

    return current_shape;
  }

  void print_summary(const std::vector<size_t> &input_shape) const {
    if (layers_.empty()) {
      std::cout << "Empty model.\n";
      return;
    }

    auto format_shape = [](const std::vector<size_t> &shape) {
      std::string shape_str = "(";
      for (size_t j = 0; j < shape.size(); ++j) {
        if (j > 0)
          shape_str += ",";
        shape_str += std::to_string(shape[j]);
      }
      shape_str += ")";
      return shape_str;
    };

    std::cout << std::string(100, '=') << "\n";
    std::cout << "Model Summary: " << name_ << "\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << std::left << std::setw(20) << "Layer (Type)" << std::setw(20) << "Input Shape"
              << std::setw(20) << "Output Shape" << std::setw(20) << "Forward Flops"
              << std::setw(20) << "Backward Flops" << "\n";

    std::vector<size_t> current_shape = input_shape;
    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto &layer = layers_[i];
      std::cout << std::left << std::setw(20)
                << (layer->get_config().name.empty() ? layer->type() : layer->get_config().name);

      std::cout << std::setw(20) << format_shape(current_shape);

      auto output_shape = layer->compute_output_shape(current_shape);
      std::cout << std::setw(20) << format_shape(output_shape);

      std::cout << std::setw(20) << layer->forward_flops(current_shape) << std::setw(20)
                << layer->backward_flops(current_shape) << "\n";
      current_shape = layer->compute_output_shape(current_shape);
    }
    std::cout << std::string(100, '-') << "\n";
  }

  std::vector<Sequential> split(std::vector<Partition> &partitions) const {
    if (partitions.empty()) {
      throw std::invalid_argument("Partitions vector is empty");
    }
    std::vector<Sequential> stages;
    stages.reserve(partitions.size());
    for (const auto &part : partitions) {
      if (part.start_layer >= layers_.size() || part.end_layer > layers_.size() ||
          part.start_layer >= part.end_layer) {
        throw std::out_of_range("Invalid partition range");
      }

      auto layers = std::vector<std::unique_ptr<Layer>>();
      for (size_t i = part.start_layer; i < part.end_layer; ++i) {
        layers.push_back(layers_[i]->clone());
      }
      Sequential stage(name_ + "_part_" + std::to_string(stages.size()), std::move(layers));
      stages.push_back(std::move(stage));
    }
    return stages;
  }

  const std::vector<Layer *> &get_layers() const {
    static std::vector<Layer *> layer_ptrs;
    layer_ptrs.clear();
    for (const auto &layer : layers_) {
      layer_ptrs.push_back(layer.get());
    }
    return layer_ptrs;
  }

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override {
    if (layers_.empty()) {
      return 0;
    }
    uint64_t total_flops = 0;
    std::vector<size_t> current_shape = input_shape;
    for (const auto &layer : layers_) {
      total_flops += layer->forward_flops(current_shape);
      current_shape = layer->compute_output_shape(current_shape);
    }
    return total_flops;
  }

  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override {
    if (layers_.empty()) {
      return 0;
    }
    uint64_t total_flops = 0;
    std::vector<size_t> current_shape = input_shape;
    for (const auto &layer : layers_) {
      total_flops += layer->backward_flops(current_shape);
      current_shape = layer->compute_output_shape(current_shape);
    }
    return total_flops;
  }

  bool has_parameters() const override {
    for (const auto &layer : layers_) {
      if (layer->has_parameters()) {
        return true;
      }
    }
    return false;
  }

  std::string type() const override { return "sequential"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = name_;
    std::vector<nlohmann::json> layers_config;
    for (const auto &layer : layers_) {
      layers_config.push_back(layer->get_config().to_json());
    }
    config.parameters["layers"] = layers_config;
    return config;
  }

  std::unique_ptr<Layer> clone() const override {
    auto cloned_layers = std::vector<std::unique_ptr<Layer>>();
    for (const auto &layer : layers_) {
      cloned_layers.push_back(layer->clone());
    }
    auto cloned = std::make_unique<Sequential>(name_, std::move(cloned_layers));
    return cloned;
  }

  size_t cached_memory_bytes() const override {
    size_t total = 0;
    for (const auto &layer : layers_) {
      total += layer->cached_memory_bytes();
    }
    return total;
  }

  void save_to_file(const std::string &path) const {
    // Save config as JSON
    std::ofstream config_file(path + "_config.json");
    if (!config_file.is_open()) {
      throw std::runtime_error("Failed to open file for writing: " + path + "_config.json");
    }
    nlohmann::json j = get_config().to_json();
    config_file << j.dump(2);
    config_file.close();

    // Save parameters as binary
    std::ofstream params_file(path + "_params.bin", std::ios::binary);
    if (!params_file.is_open()) {
      throw std::runtime_error("Failed to open file for writing: " + path + "_params.bin");
    }
    const_cast<Sequential *>(this)->save_state(params_file);
    params_file.close();
  }

  void load_from_file(const std::string &path, const Device &device) {
    // Load parameters only (config-based reconstruction not implemented)
    std::ifstream params_file(path + "_params.bin", std::ios::binary);
    if (!params_file.is_open()) {
      throw std::runtime_error("Failed to open file for reading: " + path + "_params.bin");
    }
    set_device(device);
    load_state(params_file);
    params_file.close();
  }
};

} // namespace tnn