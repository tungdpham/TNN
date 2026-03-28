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
Vec<Tensor> Sequential::forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) {
  if (layers_.empty()) {
    throw std::runtime_error("Cannot forward through empty sequential model");
  }
  Vec<ConstTensor> current_inputs = inputs;
  Vec<Tensor> current_outputs;
  if (layers_.size() % 2 == 0) {
    // assuming we are on the reverse side of input, flip so output of last layer is always opposite
    // side of input.
    allocator_->flip();
  }
  for (size_t i = 0; i < layers_.size(); ++i) {
    current_outputs = layers_[i]->forward(current_inputs, mb_id);
    current_inputs = Vec<ConstTensor>(current_outputs.begin(), current_outputs.end());
    if (i != layers_.size() - 1) {
      allocator_->flip();
    }
  }
  return current_outputs;
}

Vec<Tensor> Sequential::backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) {
  if (layers_.empty()) {
    throw std::runtime_error("Cannot backward through empty sequential model");
  }
  Vec<ConstTensor> current_gradients = grad_outputs;
  Vec<Tensor> grad_inputs;
  if (layers_.size() % 2 == 0) {
    // flip so grad output of last layer is always opposite side of input.
    allocator_->flip();
  }
  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
    grad_inputs = layers_[i]->backward(current_gradients, mb_id);
    current_gradients = Vec<ConstTensor>(grad_inputs.begin(), grad_inputs.end());
    if (i != 0) {
      allocator_->flip();  // algorithm 1 definitely applies
    }
  }
  return grad_inputs;
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
      m_b = std::max(
          m_b,
          sub_ws[i] + out_bytes[i]);  // grad output size (input) + workspace (including output)
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
