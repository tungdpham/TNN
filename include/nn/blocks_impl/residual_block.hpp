/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/activations.hpp"
#include "nn/layers_impl/base_layer.hpp"
#include "nn/layers_impl/parameterized_layer.hpp"
#include "ops/ops.hpp"

#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tnn {

/**
 * @brief Residual block implementing skip connections: output = F(x) + x
 *
 * Supports both identity shortcuts (when input/output dimensions match)
 * and projection shortcuts (1x1 conv when dimensions differ).
 */
template <typename T = float> class ResidualBlock : public ParameterizedLayer<T> {
private:
  std::vector<std::unique_ptr<Layer<T>>> main_path_;
  std::vector<std::unique_ptr<Layer<T>>> shortcut_path_;
  std::unique_ptr<ActivationFunction<T>> final_activation_;

  std::unordered_map<size_t, Tensor<T>> pre_activation_cache_;

  std::unordered_map<size_t, Tensor<T>> grad_after_activation_cache_;

  std::string activation_type_;

public:
  /**
   * @brief Constructs a residual block
   * @param main_path The main transformation path F(x) as a vector of layers
   * @param shortcut_path Optional projection path for dimension matching (empty for identity)
   * @param final_activation Activation applied after addition (e.g., "relu")
   * @param name Layer name
   */
  ResidualBlock(std::vector<std::unique_ptr<Layer<T>>> main_path,
                std::vector<std::unique_ptr<Layer<T>>> shortcut_path,
                const std::string &final_activation = "relu",
                const std::string &name = "residual_block")
      : main_path_(std::move(main_path)), shortcut_path_(std::move(shortcut_path)),
        activation_type_(final_activation) {
    this->name_ = name;

    if (final_activation != "none" && final_activation != "linear") {
      auto factory = ActivationFactory<T>();
      factory.register_defaults();
      final_activation_ = factory.create(final_activation);
    }
  }

  ResidualBlock(const ResidualBlock &other) : activation_type_(other.activation_type_) {
    this->name_ = other.name_;
    this->is_training_ = other.is_training_;

    for (const auto &layer : other.main_path_) {
      main_path_.push_back(layer->clone());
    }

    for (const auto &layer : other.shortcut_path_) {
      shortcut_path_.push_back(layer->clone());
    }

    if (other.final_activation_) {
      auto factory = ActivationFactory<T>();
      factory.register_defaults();
      final_activation_ = factory.create(activation_type_);
    }
  }

  void initialize_params() override {
    for (auto &layer : main_path_) {
      layer->initialize();
    }
    for (auto &layer : shortcut_path_) {
      layer->initialize();
    }
  }

  const Tensor<T> &forward(const Tensor<T> &input, size_t micro_batch_id = 0) override {
    const Tensor<T> *current_input = &input;
    Tensor<T> device_input;
    if (input.device() != this->device_) {
      device_input = input.to_device(this->get_device());
      current_input = &device_input;
    }

    // Main path: F(x)
    const Tensor<T> *main_path = current_input;
    for (auto &layer : main_path_) {
      main_path = &layer->forward(*main_path, micro_batch_id);
    }

    // Shortcut path: x or projection(x)
    const Tensor<T> *shortcut_path = current_input;
    if (!shortcut_path_.empty()) {
      for (auto &layer : shortcut_path_) {
        shortcut_path = &layer->forward(*shortcut_path, micro_batch_id);
      }
    }

    // Residual connection: F(x) + x
    Tensor<T> &output = this->get_output_buffer(micro_batch_id, main_path->shape());
    ops::add(main_path->data_ptr(), shortcut_path->data_ptr(), output.data_ptr(), output.size());

    // Cache pre-activation output for backward pass
    auto it_pre_act = pre_activation_cache_.find(micro_batch_id);
    if (it_pre_act == pre_activation_cache_.end()) {
      pre_activation_cache_[micro_batch_id] = output.clone();
    } else {
      it_pre_act->second.resize(output.shape());
      ops::copy(output.data_ptr(), it_pre_act->second.data_ptr(), output.size());
    }

    if (final_activation_) {
      final_activation_->apply(output);
    }

    return output;
  }

  const Tensor<T> &backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) override {
    const Tensor<T> *current_gradient = &gradient;
    Tensor<T> device_gradient;
    if (gradient.device() != this->device_) {
      device_gradient = gradient.to_device(this->device_);
      current_gradient = &device_gradient;
    }

    auto it_pre_act = pre_activation_cache_.find(micro_batch_id);
    if (final_activation_ && it_pre_act == pre_activation_cache_.end()) {
      throw std::runtime_error("No cached pre-activation output found for micro-batch ID: " +
                               std::to_string(micro_batch_id));
    }

    // Gradient through final activation
    auto it_grad_act = grad_after_activation_cache_.find(micro_batch_id);
    if (it_grad_act == grad_after_activation_cache_.end()) {
      grad_after_activation_cache_[micro_batch_id] = current_gradient->clone();
      it_grad_act = grad_after_activation_cache_.find(micro_batch_id);
    } else {
      it_grad_act->second.resize(current_gradient->shape());
      ops::copy(current_gradient->data_ptr(), it_grad_act->second.data_ptr(),
                current_gradient->size());
    }
    if (final_activation_) {
      final_activation_->compute_gradient_inplace(it_pre_act->second, it_grad_act->second);
    }

    const Tensor<T> *grad_to_propagate =
        final_activation_ ? &it_grad_act->second : current_gradient;
    // Backward through main path
    const Tensor<T> *grad_main = grad_to_propagate;
    for (int i = static_cast<int>(main_path_.size()) - 1; i >= 0; --i) {
      grad_main = &main_path_[i]->backward(*grad_main, micro_batch_id);
    }

    // Backward through shortcut
    const Tensor<T> *grad_shortcut = grad_to_propagate;
    if (!shortcut_path_.empty()) {
      for (int i = static_cast<int>(shortcut_path_.size()) - 1; i >= 0; --i) {
        grad_shortcut = &shortcut_path_[i]->backward(*grad_shortcut, micro_batch_id);
      }
    }

    // Sum gradients from both paths
    Tensor<T> &grad_input = this->get_gradient_buffer(micro_batch_id, grad_main->shape());
    ops::add(grad_main->data_ptr(), grad_shortcut->data_ptr(), grad_input.data_ptr(),
             grad_input.size());
    return grad_input;
  }

  void collect_parameters(std::vector<Tensor<T> *> &params) override {
    for (auto &layer : main_path_) {
      auto layer_params = layer->parameters();
      params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    for (auto &layer : shortcut_path_) {
      auto layer_params = layer->parameters();
      params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
  }

  void collect_gradients(std::vector<Tensor<T> *> &grads) override {
    for (auto &layer : main_path_) {
      auto layer_grads = layer->gradients();
      grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }

    for (auto &layer : shortcut_path_) {
      auto layer_grads = layer->gradients();
      grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }
  }

  void clear_gradients() override {
    for (auto &layer : main_path_) {
      layer->clear_gradients();
    }
    for (auto &layer : shortcut_path_) {
      layer->clear_gradients();
    }
  }

  bool has_parameters() const override {
    for (const auto &layer : main_path_) {
      if (layer->has_parameters())
        return true;
    }
    for (const auto &layer : shortcut_path_) {
      if (layer->has_parameters())
        return true;
    }
    return false;
  }

  void set_training(bool training) override {
    this->is_training_ = training;
    for (auto &layer : main_path_) {
      layer->set_training(training);
    }
    for (auto &layer : shortcut_path_) {
      layer->set_training(training);
    }
  }

  void set_device(const Device *device) override {
    Layer<T>::set_device(device);
    for (auto &layer : main_path_) {
      layer->set_device(device);
    }
    for (auto &layer : shortcut_path_) {
      layer->set_device(device);
    }
  }

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override {
    std::vector<size_t> shape = input_shape;
    for (const auto &layer : main_path_) {
      shape = layer->compute_output_shape(shape);
    }
    return shape;
  }

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override {
    uint64_t main_complexity = 0;
    std::vector<size_t> current_shape = input_shape;
    for (const auto &layer : main_path_) {
      main_complexity += layer->forward_complexity(current_shape);
      current_shape = layer->compute_output_shape(current_shape);
    }

    uint64_t shortcut_complexity = 0;
    std::vector<size_t> shortcut_shape = input_shape;
    for (const auto &layer : shortcut_path_) {
      shortcut_complexity += layer->forward_complexity(shortcut_shape);
      shortcut_shape = layer->compute_output_shape(shortcut_shape);
    }

    // Add complexity for element-wise addition (use output shape after main path transformations)
    size_t add_complexity = 1;
    for (size_t dim : current_shape) {
      add_complexity *= dim;
    }

    return main_complexity + shortcut_complexity + add_complexity;
  }

  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override {
    uint64_t main_complexity = 0;
    std::vector<size_t> current_shape = input_shape;
    for (const auto &layer : main_path_) {
      main_complexity += layer->backward_complexity(current_shape);
      current_shape = layer->compute_output_shape(current_shape);
    }

    uint64_t shortcut_complexity = 0;
    std::vector<size_t> shortcut_shape = input_shape;
    for (const auto &layer : shortcut_path_) {
      shortcut_complexity += layer->backward_complexity(shortcut_shape);
      shortcut_shape = layer->compute_output_shape(shortcut_shape);
    }

    // Add complexity for gradient summation (use output shape after main path transformations)
    size_t add_complexity = 1;
    for (size_t dim : current_shape) {
      add_complexity *= dim;
    }

    return main_complexity + shortcut_complexity + add_complexity;
  }

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override {
    return forward_complexity(input_shape);
  }

  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override {
    return backward_complexity(input_shape);
  }

  std::string type() const override { return "residual_block"; }

  LayerConfig get_config() const override {
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

  std::unique_ptr<Layer<T>> clone() const override {
    std::vector<std::unique_ptr<Layer<T>>> main_clone;
    for (const auto &layer : main_path_) {
      main_clone.push_back(layer->clone());
    }
    std::vector<std::unique_ptr<Layer<T>>> shortcut_clone;
    for (const auto &layer : shortcut_path_) {
      shortcut_clone.push_back(layer->clone());
    }
    return std::make_unique<ResidualBlock<T>>(std::move(main_clone), std::move(shortcut_clone),
                                              activation_type_, this->name_);
  }

  const std::vector<std::unique_ptr<Layer<T>>> &get_main_path() const { return main_path_; }
  const std::vector<std::unique_ptr<Layer<T>>> &get_shortcut_path() const { return shortcut_path_; }
};
} // namespace tnn
