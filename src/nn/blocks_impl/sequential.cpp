/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/sequential.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "nlohmann/json_fwd.hpp"
#include "nn/block.hpp"
#include "nn/layer.hpp"
#include "nn/layers.hpp"
#include "type/type.hpp"

namespace tnn {
void Sequential::forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                              size_t mb_id) {
  if (layers_.empty()) {
    throw std::runtime_error("Cannot forward through empty sequential model");
  }
  Vec<Vec<size_t>> input_shapes(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    input_shapes[i] = inputs[i]->shape();
  }
  Vec<Vec<Vec<size_t>>> out_shapes(layers_.size());
  out_shapes[0] = layers_[0]->output_shapes(input_shapes);
  for (size_t i = 1; i < layers_.size(); ++i) {
    out_shapes[i] = layers_[i]->output_shapes(out_shapes[i - 1]);
  }
  input_shapes_cache_[mb_id] = input_shapes;
  Vec<ConstTensor> current_inputs = inputs;
  for (size_t i = 0; i < layers_.size(); ++i) {
    Vec<Vec<size_t>> current_input_shapes = (i == 0) ? input_shapes : out_shapes[i - 1];
    Vec<Tensor> current_outputs;
    if (i == layers_.size() - 1) {
      current_outputs = outputs;
    } else {
      current_outputs.resize(out_shapes[i].size());
      for (size_t j = 0; j < out_shapes[i].size(); ++j) {
        if (layers_[i]->fwd_cache_bytes(current_input_shapes) > 0) {
          current_outputs[j] = this->get_cache_tensor();
        } else {
          current_outputs[j] = this->get_workspace({}, io_dtype_);
        }
      }
      if (!is_training_) {
        allocator_->flip();
      }
    }
    layers_[i]->forward(current_inputs, current_outputs, mb_id);
    current_inputs = Vec<ConstTensor>(current_outputs.begin(), current_outputs.end());
  }
  this->device().getFlow(this->flow_handle_)->synchronize();
}

void Sequential::backward_impl(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                               size_t mb_id) {
  if (layers_.empty()) {
    throw std::runtime_error("Cannot backward through empty sequential model");
  }
  auto it_in_shapes = input_shapes_cache_.find(mb_id);
  if (it_in_shapes == input_shapes_cache_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(mb_id));
  }
  Vec<Vec<size_t>> input_shape = it_in_shapes->second;
  Vec<Vec<Vec<size_t>>> out_shapes(layers_.size());
  out_shapes[0] = layers_[0]->output_shapes(input_shape);
  for (size_t i = 1; i < layers_.size(); ++i) {
    out_shapes[i] = layers_[i]->output_shapes(out_shapes[i - 1]);
  }
  Vec<ConstTensor> current_gradients = grad_outputs;
  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
    Vec<Tensor> current_grad_inputs;
    if (i == 0) {
      current_grad_inputs = grad_inputs;
    } else {
      current_grad_inputs.resize(out_shapes[i - 1].size());
      for (size_t j = 0; j < out_shapes[i - 1].size(); ++j) {
        current_grad_inputs[j] = this->get_workspace({}, io_dtype_);
      }
      allocator_->flip();  // algorithm 1 definitely applies
    }
    layers_[i]->backward(current_gradients, current_grad_inputs, mb_id);
    current_gradients = Vec<ConstTensor>(current_grad_inputs.begin(), current_grad_inputs.end());
  }
  this->device().getFlow(this->flow_handle_)->synchronize();
}

Sequential::Sequential(Vec<std::unique_ptr<Layer>> layers, const std::string &name)
    : Block(name) {
  layers_ = std::move(layers);
}

Vec<Vec<size_t>> Sequential::output_shapes(const Vec<Vec<size_t>> &input_shapes) const {
  if (layers_.empty()) {
    return input_shapes;
  }

  Vec<Vec<size_t>> current_shapes = input_shapes;
  for (const auto &layer : layers_) {
    current_shapes = layer->output_shapes(current_shapes);
  }

  return current_shapes;
}

size_t Sequential::fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const {
  if (layers_.empty()) return 0;
  size_t total_cache = 0;
  Vec<Vec<size_t>> current_shapes = input_shapes;

  for (auto it = layers_.begin(); it != layers_.end(); ++it) {
    const auto &layer = *it;
    total_cache += layer->fwd_cache_bytes(current_shapes);
    current_shapes = layer->output_shapes(current_shapes);
  }

  return total_cache;
}

size_t Sequential::fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (layers_.empty()) return 0;
  size_t total_ws = 0;
  Vec<Vec<size_t>> current_shapes = input_shapes;

  // total = max over all layer workspace
  for (auto it = layers_.begin(); it != layers_.end(); ++it) {
    const auto &layer = *it;
    size_t layer_ws = layer->fwd_workspace(current_shapes);
    total_ws = std::max(total_ws, layer_ws);
    current_shapes = layer->output_shapes(current_shapes);
  }

  return total_ws;
}

size_t Sequential::inf_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (layers_.empty()) return 0;

  Vec<size_t> sub_ws;
  Vec<size_t> in_bytes;
  Vec<Vec<size_t>> cur = input_shapes;
  for (const auto &layer : layers_) {
    Vec<Vec<size_t>> out = layer->output_shapes(cur);
    sub_ws.push_back(layer->inf_workspace(cur));
    in_bytes.push_back(get_shapes_bytes(cur, io_dtype_));
    cur = out;
  }

  size_t m_b = 0;
  if (layers_.size() == 1) {
    m_b = sub_ws[0];
  } else if (layers_.size() > 1) {
    m_b = sub_ws[0];
    for (size_t i = 1; i < layers_.size() - 1; ++i) {
      m_b = std::max(m_b, sub_ws[i] + in_bytes[i]);
    }
    m_b = std::max(m_b, sub_ws[layers_.size() - 1]);
  }
  return m_b;
}

size_t Sequential::bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const {
  if (layers_.empty()) return 0;

  Vec<size_t> sub_ws;
  Vec<size_t> out_bytes;
  Vec<Vec<size_t>> cur = input_shapes;
  for (const auto &layer : layers_) {
    Vec<Vec<size_t>> out = layer->output_shapes(cur);
    sub_ws.push_back(layer->bwd_workspace(cur));
    out_bytes.push_back(get_shapes_bytes(out, io_dtype_));
    cur = out;
  }

  size_t m_b = 0;
  if (layers_.size() == 1) {
    m_b = sub_ws[0];
  } else if (layers_.size() > 1) {
    m_b = sub_ws[0];
    for (size_t i = 1; i < layers_.size() - 1; ++i) {
      m_b = std::max(m_b, sub_ws[i] + out_bytes[i]);  // layer bwd ws + grad output size
    }
    m_b = std::max(m_b, sub_ws[layers_.size() - 1]);
  }
  return m_b;
}

void Sequential::print_summary(const Vec<size_t> &input_shape) const {
  if (layers_.empty()) {
    std::cout << "Empty model.\n";
    return;
  }

  auto format_shape = [](const Vec<size_t> &shape) {
    std::string shape_str = "(";
    for (size_t j = 0; j < shape.size(); ++j) {
      if (j > 0) shape_str += ",";
      shape_str += std::to_string(shape[j]);
    }
    shape_str += ")";
    return shape_str;
  };

  std::cout << std::string(100, '=') << "\n";
  std::cout << "Model Summary: " << name_ << "\n";
  std::cout << std::string(100, '=') << "\n";
  std::cout << std::left << std::setw(20) << "Layer (Type)" << std::setw(20) << "Input Shape"
            << std::setw(20) << "Output Shape" << "\n";

  Vec<size_t> current_shape = input_shape;
  for (size_t i = 0; i < layers_.size(); ++i) {
    const auto &layer = layers_[i];
    std::cout << std::left << std::setw(20)
              << (layer->get_config().name.empty() ? layer->type() : layer->get_config().name);

    std::cout << std::setw(20) << format_shape(current_shape);

    auto output_shape = layer->output_shapes({current_shape})[0];
    std::cout << std::setw(20) << format_shape(output_shape) << "\n";
    current_shape = layer->output_shapes({current_shape})[0];
  }
  std::cout << std::string(100, '-') << "\n";
}

Vec<Layer *> Sequential::get_layers() { return this->layers(); }

LayerConfig Sequential::get_config() const {
  LayerConfig config;
  config.name = name_;
  config.type = TYPE_NAME;
  nlohmann::json layers_config = nlohmann::json::array();
  for (const auto &layer : layers_) {
    auto layer_config = layer->get_config();
    layers_config.push_back(layer_config.to_json());
  }
  config.set("layers", layers_config);
  return config;
}

std::unique_ptr<Sequential> Sequential::create_from_config(const LayerConfig &config) {
  Vec<std::unique_ptr<Layer>> layers;
  nlohmann::json layers_json = config.get<nlohmann::json>("layers", nlohmann::json::array());
  if (!layers_json.is_array()) {
    throw std::runtime_error("Sequential layer config 'layers' parameter must be an array");
  }
  LayerFactory::register_defaults();
  for (const auto &layer_json : layers_json) {
    LayerConfig layer_config = LayerConfig::from_json(layer_json);
    auto layer = LayerFactory::create(layer_config);
    layers.push_back(std::move(layer));
  }
  return std::make_unique<Sequential>(std::move(layers), config.name);
}

}  // namespace tnn
