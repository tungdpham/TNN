/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/activations.hpp"
#include "nn/layers_impl/base_layer.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tnn {

/**
 * @brief Residual block implementing skip connections: output = F(x) + x
 *
 * Supports both identity shortcuts (when input/output dimensions match)
 * and projection shortcuts (1x1 conv when dimensions differ).
 */
template <typename T = float> class ResidualBlock : public Layer<T> {
private:
  std::vector<std::unique_ptr<Layer<T>>> main_path_;
  std::unique_ptr<Layer<T>> shortcut_;
  std::unique_ptr<ActivationFunction<T>> final_activation_;

  Tensor<T> main_output_cache_;
  Tensor<T> shortcut_output_cache_;
  Tensor<T> pre_activation_cache_;

  std::string activation_type_;

public:
  /**
   * @brief Constructs a residual block
   * @param main_path The main transformation path F(x) as a vector of layers
   * @param shortcut Optional projection layer for dimension matching (nullptr for identity)
   * @param final_activation Activation applied after addition (e.g., "relu")
   * @param name Layer name
   */
  ResidualBlock(std::vector<std::unique_ptr<Layer<T>>> main_path,
                std::unique_ptr<Layer<T>> shortcut = nullptr,
                const std::string &final_activation = "relu",
                const std::string &name = "residual_block")
      : main_path_(std::move(main_path)), shortcut_(std::move(shortcut)),
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

    if (other.shortcut_) {
      shortcut_ = other.shortcut_->clone();
    }

    if (other.final_activation_) {
      auto factory = ActivationFactory<T>();
      factory.register_defaults();
      final_activation_ = factory.create(activation_type_);
    }
  }

  void initialize() override {
    for (auto &layer : main_path_) {
      layer->initialize();
    }
    if (shortcut_) {
      shortcut_->initialize();
    }
  }

  Tensor<T> forward(const Tensor<T> &input, size_t micro_batch_id = 0) override {
    const Tensor<T> &current_input =
        input.device() == this->device_ ? input : input.to_device(this->get_device());

    // Main path: F(x)
    Tensor<T> current = current_input.clone();
    for (auto &layer : main_path_) {
      layer->forward_inplace(current, micro_batch_id);
    }
    main_output_cache_ = std::move(current);

    // Shortcut path: x or projection(x)
    if (shortcut_) {
      shortcut_output_cache_ = shortcut_->forward(current_input, micro_batch_id);
    } else {
      shortcut_output_cache_ = current_input.clone();
    }

    // Residual connection: F(x) + x
    pre_activation_cache_ = main_output_cache_ + shortcut_output_cache_;

    Tensor<T> output = pre_activation_cache_.clone();

    if (final_activation_) {
      final_activation_->apply(output);
    }

    return output;
  }

  Tensor<T> backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) override {
    Tensor<T> grad = gradient.clone();

    // Gradient through final activation
    if (final_activation_) {
      final_activation_->compute_gradient_inplace(pre_activation_cache_, grad);
    }

    // Backward through main path
    Tensor<T> grad_main = grad.clone();
    for (int i = static_cast<int>(main_path_.size()) - 1; i >= 0; --i) {
      main_path_[i]->backward_inplace(grad_main, micro_batch_id);
    }

    // Backward through shortcut
    Tensor<T> grad_shortcut;
    if (shortcut_) {
      grad_shortcut = shortcut_->backward(grad, micro_batch_id);
    } else {
      grad_shortcut = grad.clone();
    }

    // Sum gradients from both paths
    return grad_main + grad_shortcut;
  }

  std::vector<Tensor<T> *> parameters() override {
    std::vector<Tensor<T> *> params;
    for (auto &layer : main_path_) {
      auto layer_params = layer->parameters();
      params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    if (shortcut_) {
      auto shortcut_params = shortcut_->parameters();
      params.insert(params.end(), shortcut_params.begin(), shortcut_params.end());
    }

    return params;
  }

  std::vector<Tensor<T> *> gradients() override {
    std::vector<Tensor<T> *> grads;
    for (auto &layer : main_path_) {
      auto layer_grads = layer->gradients();
      grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }

    if (shortcut_) {
      auto shortcut_grads = shortcut_->gradients();
      grads.insert(grads.end(), shortcut_grads.begin(), shortcut_grads.end());
    }

    return grads;
  }

  bool has_parameters() const override {
    for (const auto &layer : main_path_) {
      if (layer->has_parameters())
        return true;
    }
    return (shortcut_ && shortcut_->has_parameters());
  }

  void set_training(bool training) override {
    this->is_training_ = training;
    for (auto &layer : main_path_) {
      layer->set_training(training);
    }
    if (shortcut_) {
      shortcut_->set_training(training);
    }
  }

  void set_device(const Device *device) override {
    Layer<T>::set_device(device);
    for (auto &layer : main_path_) {
      layer->set_device(device);
    }
    if (shortcut_) {
      shortcut_->set_device(device);
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
    if (shortcut_) {
      shortcut_complexity = shortcut_->forward_complexity(input_shape);
    }

    // Add complexity for element-wise addition (approximately input_size operations)
    size_t add_complexity = 1;
    for (size_t dim : input_shape) {
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
    if (shortcut_) {
      shortcut_complexity = shortcut_->backward_complexity(input_shape);
    }

    size_t add_complexity = 1;
    for (size_t dim : input_shape) {
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

  std::string type() const override { return "ResidualBlock"; }

  LayerConfig get_config() const override {
    LayerConfig config;
    config.name = this->name_;
    config.parameters["activation"] = activation_type_;
    config.parameters["has_projection"] = (shortcut_ != nullptr);
    return config;
  }

  std::unique_ptr<Layer<T>> clone() const override {
    std::vector<std::unique_ptr<Layer<T>>> main_clone;
    for (const auto &layer : main_path_) {
      main_clone.push_back(layer->clone());
    }
    auto shortcut_clone = shortcut_ ? shortcut_->clone() : nullptr;
    return std::make_unique<ResidualBlock<T>>(std::move(main_clone), std::move(shortcut_clone),
                                              activation_type_, this->name_);
  }

  const std::vector<std::unique_ptr<Layer<T>>> &get_main_path() const { return main_path_; }
  const Layer<T> *get_shortcut() const { return shortcut_.get(); }
};
} // namespace tnn
