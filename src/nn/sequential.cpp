/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/sequential.hpp"
#include "nn/layers.hpp"
#include "profiling/event.hpp"
#include "tensor/tensor.hpp"

#include <algorithm>
#include <fmt/core.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace tnn {

void Sequential::compute_max_size(const std::vector<size_t> &input_shape, DType_t dtype) {
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

void Sequential::init_impl() {
  for (auto &layer : layers_) {
    layer->init();
  }
}

void Sequential::on_set_io_dtype(DType_t dtype) {
  for (auto &layer : layers_) {
    layer->set_io_dtype(dtype);
  }
}

void Sequential::on_set_param_dtype(DType_t dtype) {
  for (auto &layer : layers_) {
    layer->set_param_dtype(dtype);
  }
}

void Sequential::on_set_compute_dtype(DType_t dtype) {
  for (auto &layer : layers_) {
    layer->set_compute_dtype(dtype);
  }
}

void Sequential::on_set_device(const Device &device) {
  for (auto &layer : layers_) {
    layer->set_device(device);
  }
}

void Sequential::on_set_training(bool training) {
  for (auto &layer : layers_) {
    layer->set_training(training);
  }
}

void Sequential::forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  if (layers_.empty()) {
    throw std::runtime_error("Cannot forward through empty sequential model");
  }
  auto start = Clock::now();
  compute_max_size(input->shape(), input->data_type());

  Tensor current_input = input;
  Tensor current_output;
  for (size_t i = 0; i < layers_.size(); ++i) {
    try {
      auto start = Clock::now();
      current_output = this->get_buffer(layers_[i]->compute_output_shape(current_input->shape()),
                                        input->data_type());
      layers_[i]->forward(current_input, current_output, micro_batch_id);
      current_input = current_output;
      auto end = Clock::now();
      this->profiler_.add_event(
          Event{EventType::COMPUTE, start, end, layers_[i]->name() + " forward"});
    } catch (const std::exception &e) {
      throw std::runtime_error("Error while forward in layer " + std::to_string(i) + " (" +
                               layers_[i]->name() + "): " + e.what());
    }
  }
  output = current_output;
  this->device_->getFlow("default")->synchronize();
  auto end = Clock::now();
  this->profiler_.add_event(Event{EventType::COMPUTE, start, end, "Sequential forward"});
}

void Sequential::backward_impl(const Tensor &gradient, Tensor &grad_input, size_t micro_batch_id) {
  if (layers_.empty()) {
    throw std::runtime_error("Cannot backward through empty sequential model");
  }
  auto start = Clock::now();
  Tensor current_gradient = gradient;
  Tensor current_grad_input = this->get_buffer({max_size_}, gradient->data_type());
  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
    try {
      auto start = Clock::now();
      // no need to renew buffer since backward doesn't cache inputs
      layers_[i]->backward(current_gradient, current_grad_input, micro_batch_id);
      std::swap(current_gradient, current_grad_input);
      auto end = Clock::now();
      this->profiler_.add_event(
          Event{EventType::COMPUTE, start, end, layers_[i]->name() + " backward"});
    } catch (const std::exception &e) {
      throw std::runtime_error("Error in backward pass of layer " + std::to_string(i) + " (" +
                               layers_[i]->type() + "): " + e.what());
    }
  }
  grad_input = current_gradient;
  this->device_->getFlow("default")->synchronize();
  auto end = Clock::now();
  this->profiler_.add_event(Event{EventType::COMPUTE, start, end, "Sequential backward"});
}

Sequential::Sequential(const std::string &name, std::vector<std::unique_ptr<Layer>> layers)
    : Layer() {
  this->name_ = name;
  layers_ = std::move(layers);
}

Sequential::Sequential() : Layer() {}

std::vector<Tensor> Sequential::parameters() {
  std::vector<Tensor> all_params;
  for (auto &layer : layers_) {
    auto layer_params = layer->parameters();
    all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
  }
  return all_params;
}

std::vector<Tensor> Sequential::gradients() {
  std::vector<Tensor> all_grads;
  for (auto &layer : layers_) {
    auto layer_grads = layer->gradients();
    all_grads.insert(all_grads.end(), layer_grads.begin(), layer_grads.end());
  }
  return all_grads;
}

std::vector<size_t> Sequential::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (layers_.empty()) {
    return input_shape;
  }

  std::vector<size_t> current_shape = input_shape;
  for (const auto &layer : layers_) {
    current_shape = layer->compute_output_shape(current_shape);
  }

  return current_shape;
}

void Sequential::print_summary(const std::vector<size_t> &input_shape) const {
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
            << std::setw(20) << "Output Shape" << std::setw(20) << "Forward Flops" << std::setw(20)
            << "Backward Flops" << "\n";

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

std::vector<Sequential> Sequential::split(std::vector<Partition> &partitions) const {
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

const std::vector<Layer *> &Sequential::get_layers() const {
  static std::vector<Layer *> layer_ptrs;
  layer_ptrs.clear();
  for (const auto &layer : layers_) {
    layer_ptrs.push_back(layer.get());
  }
  return layer_ptrs;
}

uint64_t Sequential::forward_flops(const std::vector<size_t> &input_shape) const {
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

uint64_t Sequential::backward_flops(const std::vector<size_t> &input_shape) const {
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

bool Sequential::has_parameters() const {
  for (const auto &layer : layers_) {
    if (layer->has_parameters()) {
      return true;
    }
  }
  return false;
}

LayerConfig Sequential::get_config() const {
  LayerConfig config;
  config.name = name_;
  config.type = TYPE_NAME;
  std::vector<nlohmann::json> layers_config = nlohmann::json::array();
  for (const auto &layer : layers_) {
    auto layer_config = layer->get_config();
    layers_config.push_back(layer_config.to_json());
  }
  config.parameters["layers"] = layers_config;
  return config;
}

std::unique_ptr<Sequential> Sequential::create_from_config(const LayerConfig &config) {
  std::vector<std::unique_ptr<Layer>> layers;
  std::cout << "Creating Sequential layer from config: " << config.to_json().dump(2) << std::endl;
  // Get the layers as nlohmann::json first, then convert to vector
  nlohmann::json layers_json = config.get<nlohmann::json>("layers", nlohmann::json::array());
  LayerFactory::register_defaults();

  if (!layers_json.is_array()) {
    throw std::runtime_error("Expected 'layers' to be an array in Sequential config");
  }

  for (const auto &layer_json : layers_json) {
    LayerConfig layer_config = LayerConfig::from_json(layer_json);
    auto layer = LayerFactory::create(layer_config);
    layers.push_back(std::move(layer));
  }
  return std::make_unique<Sequential>(config.name, std::move(layers));
}

std::unique_ptr<Layer> Sequential::clone() const {
  auto cloned_layers = std::vector<std::unique_ptr<Layer>>();
  for (const auto &layer : layers_) {
    cloned_layers.push_back(layer->clone());
  }
  auto cloned = std::make_unique<Sequential>(name_, std::move(cloned_layers));
  return cloned;
}

size_t Sequential::cached_memory_bytes() const {
  size_t total = 0;
  for (const auto &layer : layers_) {
    total += layer->cached_memory_bytes();
  }
  return total;
}

} // namespace tnn
