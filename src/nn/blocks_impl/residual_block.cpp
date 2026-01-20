/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/residual_block.hpp"
#include "nn/activations.hpp"
#include "tensor/ops.hpp"

#include <algorithm>
#include <numeric>

namespace tnn {

ResidualBlock::ResidualBlock(std::vector<std::unique_ptr<Layer>> main_path,
                             std::vector<std::unique_ptr<Layer>> shortcut_path,
                             const std::string &final_activation, const std::string &name)
    : main_path_(std::move(main_path)), shortcut_path_(std::move(shortcut_path)),
      activation_type_(final_activation) {
  this->name_ = name;

  if (final_activation != "none" && final_activation != "linear") {
    auto factory = ActivationFactory();
    factory.register_defaults();
    final_activation_ = factory.create(final_activation);
  }
}

ResidualBlock::ResidualBlock(const ResidualBlock &other)
    : activation_type_(other.activation_type_) {
  this->name_ = other.name_;
  this->is_training_ = other.is_training_;

  for (const auto &layer : other.main_path_) {
    main_path_.push_back(layer->clone());
  }

  for (const auto &layer : other.shortcut_path_) {
    shortcut_path_.push_back(layer->clone());
  }

  if (other.final_activation_) {
    auto factory = ActivationFactory();
    factory.register_defaults();
    final_activation_ = factory.create(activation_type_);
  }
}

void ResidualBlock::init_params() {
  for (auto &layer : main_path_) {
    layer->init();
  }
  for (auto &layer : shortcut_path_) {
    layer->init();
  }
}

void ResidualBlock::forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  // Cache input shape for backward pass
  input_shape_cache_[micro_batch_id] = input->shape();

  size_t max_size = 0;
  std::vector<size_t> current_shape = input->shape();
  for (auto &layer : main_path_) {
    auto layer_shape = layer->compute_output_shape(current_shape);
    size_t layer_size =
        std::accumulate(layer_shape.begin(), layer_shape.end(), 1, std::multiplies<size_t>());
    max_size = std::max(max_size, layer_size);
    current_shape = layer_shape;
  }

  // Main path: F(x)
  const Tensor *main_path = &input;
  Tensor main_output = this->get_buffer({max_size});
  Tensor temp_output_main = this->get_buffer({max_size});
  for (auto &layer : main_path_) {
    layer->forward(*main_path, temp_output_main, micro_batch_id);
    std::swap(main_output, temp_output_main);
    main_path = &main_output;
  }

  // Shortcut path: x or projection(x)
  const Tensor *shortcut_path = &input;
  Tensor shortcut_output = this->get_buffer({max_size});
  Tensor temp_output_shortcut = this->get_buffer({max_size});
  for (auto &layer : shortcut_path_) {
    layer->forward(*shortcut_path, temp_output_shortcut, micro_batch_id);
    std::swap(shortcut_output, temp_output_shortcut);
    shortcut_path = &shortcut_output;
  }

  // Residual connection: F(x) + x
  output->ensure((*main_path)->shape(), this->device_);
  DISPATCH_ON_DTYPE_TO_METHOD(TensorOps::add, *main_path, *shortcut_path, output, output->size());

  // Cache pre-activation output for backward pass
  if (this->is_training_) {
    Tensor &pre_act = pre_activation_cache_[micro_batch_id];
    if (pre_act == nullptr) {
      pre_act = make_io_tensor(output->shape());
    }
    pre_act->ensure(output->shape(), this->device_);
    output->copy_to(pre_act);
  }

  if (final_activation_) {
    final_activation_->apply(output, output);
  }
}

void ResidualBlock::backward_impl(const Tensor &gradient, Tensor &grad_input,
                                  size_t micro_batch_id) {

  auto it_pre_act = pre_activation_cache_.find(micro_batch_id);
  if (final_activation_ && it_pre_act == pre_activation_cache_.end()) {
    throw std::runtime_error("No cached pre-activation output found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  const Tensor *grad_to_propagate = &gradient;
  Tensor grad_act_pooled;

  if (final_activation_) {
    grad_act_pooled = this->get_buffer(gradient->shape());
    final_activation_->compute_gradient(it_pre_act->second, *grad_to_propagate, grad_act_pooled);
    grad_to_propagate = &grad_act_pooled;
  }

  // Retrieve cached input shape
  auto it_input_shape = input_shape_cache_.find(micro_batch_id);
  if (it_input_shape == input_shape_cache_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }

  // Calculate maximum buffer size needed
  size_t max_size = 0;
  std::vector<size_t> current_shape = it_input_shape->second;
  for (auto &layer : main_path_) {
    auto layer_shape = layer->compute_output_shape(current_shape);
    size_t layer_size =
        std::accumulate(layer_shape.begin(), layer_shape.end(), 1, std::multiplies<size_t>());
    max_size = std::max(max_size, layer_size);
    current_shape = layer_shape;
  }

  // Backward through main path
  const Tensor *grad_main = grad_to_propagate;
  Tensor grad_output_main = this->get_buffer({max_size});
  Tensor temp_grad_main = this->get_buffer({max_size});
  for (int i = static_cast<int>(main_path_.size()) - 1; i >= 0; --i) {
    main_path_[i]->backward(*grad_main, temp_grad_main, micro_batch_id);
    std::swap(grad_output_main, temp_grad_main);
    grad_main = &grad_output_main;
  }

  // Backward through shortcut
  const Tensor *grad_shortcut = grad_to_propagate;
  Tensor grad_output_shortcut = this->get_buffer({max_size});
  Tensor temp_grad_shortcut = this->get_buffer({max_size});
  if (!shortcut_path_.empty()) {
    for (int i = static_cast<int>(shortcut_path_.size()) - 1; i >= 0; --i) {
      shortcut_path_[i]->backward(*grad_shortcut, temp_grad_shortcut, micro_batch_id);
      std::swap(grad_output_shortcut, temp_grad_shortcut);
      grad_shortcut = &grad_output_shortcut;
    }
  }

  grad_input->ensure((*grad_main)->shape(), this->device_);
  DISPATCH_ON_DTYPE_TO_METHOD(TensorOps::add, *grad_main, *grad_shortcut, grad_input,
                              grad_input->size());
}

void ResidualBlock::collect_parameters(std::vector<Tensor> &params) {
  for (auto &layer : main_path_) {
    auto layer_params = layer->parameters();
    params.insert(params.end(), layer_params.begin(), layer_params.end());
  }

  for (auto &layer : shortcut_path_) {
    auto layer_params = layer->parameters();
    params.insert(params.end(), layer_params.begin(), layer_params.end());
  }
}

void ResidualBlock::collect_gradients(std::vector<Tensor> &grads) {
  for (auto &layer : main_path_) {
    auto layer_grads = layer->gradients();
    grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
  }

  for (auto &layer : shortcut_path_) {
    auto layer_grads = layer->gradients();
    grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
  }
}

void ResidualBlock::on_set_training(bool training) {
  this->is_training_ = training;
  for (auto &layer : main_path_) {
    layer->set_training(training);
  }
  for (auto &layer : shortcut_path_) {
    layer->set_training(training);
  }
}

void ResidualBlock::on_set_device(const Device &device) {
  for (auto &layer : main_path_) {
    layer->set_device(device);
  }
  for (auto &layer : shortcut_path_) {
    layer->set_device(device);
  }
}

std::vector<size_t>
ResidualBlock::compute_output_shape(const std::vector<size_t> &input_shape) const {
  std::vector<size_t> shape = input_shape;
  for (const auto &layer : main_path_) {
    shape = layer->compute_output_shape(shape);
  }
  return shape;
}

uint64_t ResidualBlock::forward_flops(const std::vector<size_t> &input_shape) const {
  uint64_t main_complexity = 0;
  std::vector<size_t> current_shape = input_shape;
  for (const auto &layer : main_path_) {
    main_complexity += layer->forward_flops(current_shape);
    current_shape = layer->compute_output_shape(current_shape);
  }

  uint64_t shortcut_complexity = 0;
  std::vector<size_t> shortcut_shape = input_shape;
  for (const auto &layer : shortcut_path_) {
    shortcut_complexity += layer->forward_flops(shortcut_shape);
    shortcut_shape = layer->compute_output_shape(shortcut_shape);
  }

  // Add complexity for element-wise addition (use output shape after main path transformations)
  size_t add_complexity = 1;
  for (size_t dim : current_shape) {
    add_complexity *= dim;
  }

  return main_complexity + shortcut_complexity + add_complexity;
}

uint64_t ResidualBlock::backward_flops(const std::vector<size_t> &input_shape) const {
  uint64_t main_complexity = 0;
  std::vector<size_t> current_shape = input_shape;
  for (const auto &layer : main_path_) {
    main_complexity += layer->backward_flops(current_shape);
    current_shape = layer->compute_output_shape(current_shape);
  }

  uint64_t shortcut_complexity = 0;
  std::vector<size_t> shortcut_shape = input_shape;
  for (const auto &layer : shortcut_path_) {
    shortcut_complexity += layer->backward_flops(shortcut_shape);
    shortcut_shape = layer->compute_output_shape(shortcut_shape);
  }

  // Add complexity for gradient summation (use output shape after main path transformations)
  size_t add_complexity = 1;
  for (size_t dim : current_shape) {
    add_complexity *= dim;
  }

  return main_complexity + shortcut_complexity + add_complexity;
}

std::string ResidualBlock::type() const { return "residual_block"; }

LayerConfig ResidualBlock::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["activation"] = activation_type_;
  config.parameters["has_projection"] = (!shortcut_path_.empty());

  // Serialize main_path layers
  nlohmann::json main_array = nlohmann::json::array();
  for (const auto &layer : main_path_) {
    LayerConfig sub_cfg = layer->get_config();
    nlohmann::json sub_json;
    sub_json["type"] = layer->type();
    sub_json["name"] = sub_cfg.name;
    sub_json["parameters"] = nlohmann::json::object();
    for (const auto &[k, v] : sub_cfg.parameters) {
      try {
        if (auto *int_ptr = std::any_cast<int>(&v)) {
          sub_json["parameters"][k] = *int_ptr;
        } else if (auto *size_ptr = std::any_cast<size_t>(&v)) {
          sub_json["parameters"][k] = *size_ptr;
        } else if (auto *float_ptr = std::any_cast<float>(&v)) {
          sub_json["parameters"][k] = *float_ptr;
        } else if (auto *double_ptr = std::any_cast<double>(&v)) {
          sub_json["parameters"][k] = *double_ptr;
        } else if (auto *bool_ptr = std::any_cast<bool>(&v)) {
          sub_json["parameters"][k] = *bool_ptr;
        } else if (auto *string_ptr = std::any_cast<std::string>(&v)) {
          sub_json["parameters"][k] = *string_ptr;
        }
      } catch (const std::bad_any_cast &) {
      }
    }
    main_array.push_back(sub_json);
  }

  // Serialize shortcut_path layers
  nlohmann::json shortcut_array = nlohmann::json::array();
  for (const auto &layer : shortcut_path_) {
    LayerConfig sub_cfg = layer->get_config();
    nlohmann::json sub_json;
    sub_json["type"] = layer->type();
    sub_json["name"] = sub_cfg.name;
    sub_json["parameters"] = nlohmann::json::object();
    for (const auto &[k, v] : sub_cfg.parameters) {
      try {
        if (auto *int_ptr = std::any_cast<int>(&v)) {
          sub_json["parameters"][k] = *int_ptr;
        } else if (auto *size_ptr = std::any_cast<size_t>(&v)) {
          sub_json["parameters"][k] = *size_ptr;
        } else if (auto *float_ptr = std::any_cast<float>(&v)) {
          sub_json["parameters"][k] = *float_ptr;
        } else if (auto *double_ptr = std::any_cast<double>(&v)) {
          sub_json["parameters"][k] = *double_ptr;
        } else if (auto *bool_ptr = std::any_cast<bool>(&v)) {
          sub_json["parameters"][k] = *bool_ptr;
        } else if (auto *string_ptr = std::any_cast<std::string>(&v)) {
          sub_json["parameters"][k] = *string_ptr;
        }
      } catch (const std::bad_any_cast &) {
      }
    }
    shortcut_array.push_back(sub_json);
  }

  // Store serialized arrays as strings in parameters map
  config.parameters["main_path"] = main_array.dump();
  config.parameters["shortcut_path"] = shortcut_array.dump();

  return config;
}

std::unique_ptr<Layer> ResidualBlock::clone() const {
  std::vector<std::unique_ptr<Layer>> main_clone;
  for (const auto &layer : main_path_) {
    main_clone.push_back(layer->clone());
  }
  std::vector<std::unique_ptr<Layer>> shortcut_clone;
  for (const auto &layer : shortcut_path_) {
    shortcut_clone.push_back(layer->clone());
  }
  return std::make_unique<ResidualBlock>(std::move(main_clone), std::move(shortcut_clone),
                                         activation_type_, this->name_);
}

const std::vector<std::unique_ptr<Layer>> &ResidualBlock::get_main_path() const {
  return main_path_;
}

const std::vector<std::unique_ptr<Layer>> &ResidualBlock::get_shortcut_path() const {
  return shortcut_path_;
}

size_t ResidualBlock::cached_memory_bytes() const {
  size_t total_bytes = 0;
  for (const auto &layer : main_path_) {
    total_bytes += layer->cached_memory_bytes();
  }
  for (const auto &layer : shortcut_path_) {
    total_bytes += layer->cached_memory_bytes();
  }
  for (const auto &[_, tensor] : pre_activation_cache_) {
    size_t dtype_size = get_dtype_size(tensor->data_type());
    total_bytes += tensor->size() * dtype_size;
  }
  total_bytes += Layer::cached_memory_bytes();
  return total_bytes;
}

} // namespace tnn
