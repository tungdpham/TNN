/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/residual_block.hpp"

#include <algorithm>
#include <numeric>

#include "nn/activations.hpp"
#include "nn/layers.hpp"
#include "ops/ops.hpp"

namespace tnn {

ResidualBlock::ResidualBlock(std::vector<std::unique_ptr<Layer>> main_path,
                             std::vector<std::unique_ptr<Layer>> shortcut_path,
                             const std::string &final_activation, const std::string &name)
    : main_path_(std::move(main_path)),
      shortcut_path_(std::move(shortcut_path)),
      activation_type_(final_activation) {
  if (main_path_.empty()) {
    throw std::runtime_error("Main path of ResidualBlock cannot be empty.");
  }
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

void ResidualBlock::on_set_io_dtype(DType_t dtype) {
  for (auto &layer : main_path_) {
    layer->set_io_dtype(dtype);
  }
  for (auto &layer : shortcut_path_) {
    layer->set_io_dtype(dtype);
  }
}

void ResidualBlock::on_set_param_dtype(DType_t dtype) {
  for (auto &layer : main_path_) {
    layer->set_param_dtype(dtype);
  }
  for (auto &layer : shortcut_path_) {
    layer->set_param_dtype(dtype);
  }
}

void ResidualBlock::on_set_compute_dtype(DType_t dtype) {
  for (auto &layer : main_path_) {
    layer->set_compute_dtype(dtype);
  }
  for (auto &layer : shortcut_path_) {
    layer->set_compute_dtype(dtype);
  }
}

void ResidualBlock::forward_impl(const ConstTensor &input, Tensor &output, size_t mb_id) {
  input_shape_cache_[mb_id] = input->shape();

  size_t max_size = 0;
  std::vector<size_t> current_shape = input->shape();
  max_size = std::max(max_size, input->size());
  for (auto &layer : main_path_) {
    auto layer_shape = layer->compute_output_shape(current_shape);
    size_t layer_size =
        std::accumulate(layer_shape.begin(), layer_shape.end(), 1, std::multiplies<size_t>());
    max_size = std::max(max_size, layer_size);
    current_shape = layer_shape;
  }

  ConstTensor main_output = input;  // main output = f exist ? input : f(input)
  for (auto &layer : main_path_) {
    Tensor temp_output = this->get_buffer(layer->compute_output_shape(main_output->shape()),
                                          main_output->data_type());
    layer->forward(main_output, temp_output, mb_id);
    main_output = temp_output;
  }

  ConstTensor shortcut_output = input;  // shortcut output = g exist ? input : g(input)
  for (auto &layer : shortcut_path_) {
    Tensor temp_output = this->get_buffer(layer->compute_output_shape(shortcut_output->shape()),
                                          shortcut_output->data_type());
    layer->forward(shortcut_output, temp_output, mb_id);
    shortcut_output = temp_output;
  }

  if (final_activation_) {
    Tensor &pre_act = this->get_mutable_tensor(mb_id, "pre_activation");
    if (!pre_act)
      pre_act = this->get_buffer(main_output->shape(), main_output->data_type());
    else
      pre_act->ensure(main_output->shape());
    DISPATCH_ON_DTYPE_TO_METHOD(ops::add, main_output->data_ptr(), shortcut_output->data_ptr(),
                                pre_act->data_ptr(), pre_act->size());

    output->ensure(main_output->shape());
    final_activation_->apply(pre_act, output);
  } else {
    output->ensure(main_output->shape());
    DISPATCH_ON_DTYPE_TO_METHOD(ops::add, main_output->data_ptr(), shortcut_output->data_ptr(),
                                output->data_ptr(), output->size());
  }
}

void ResidualBlock::backward_impl(const ConstTensor &gradient, Tensor &grad_input, size_t mb_id) {
  Tensor &pre_act = this->get_mutable_tensor(mb_id, "pre_activation");
  if (final_activation_ && !pre_act) {
    throw std::runtime_error("No cached pre-activation output found for micro-batch ID: " +
                             std::to_string(mb_id));
  }

  ConstTensor grad_to_propagate = gradient;

  if (final_activation_) {
    Tensor dpre_act = this->get_buffer(pre_act->shape(), pre_act->data_type());
    final_activation_->compute_gradient(pre_act, gradient, dpre_act);
    grad_to_propagate = dpre_act;
  }

  auto it_input_shape = input_shape_cache_.find(mb_id);
  if (it_input_shape == input_shape_cache_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(mb_id));
  }

  size_t max_size = 0;
  std::vector<size_t> current_shape = it_input_shape->second;
  size_t input_size =
      std::accumulate(current_shape.begin(), current_shape.end(), 1, std::multiplies<size_t>());
  max_size = std::max(max_size, input_size);
  for (auto &layer : main_path_) {
    auto layer_shape = layer->compute_output_shape(current_shape);
    size_t layer_size =
        std::accumulate(layer_shape.begin(), layer_shape.end(), 1, std::multiplies<size_t>());
    max_size = std::max(max_size, layer_size);
    current_shape = layer_shape;
  }

  // little trick to avoid const correctness issue
  ConstTensor main_grad = grad_to_propagate;
  for (int i = static_cast<int>(main_path_.size()) - 1; i >= 0; --i) {
    Tensor temp_grad = this->get_buffer({max_size}, gradient->data_type());
    main_path_[i]->backward(main_grad, temp_grad, mb_id);
    main_grad = temp_grad;
  }

  ConstTensor shortcut_grad = grad_to_propagate;  // same here
  if (!shortcut_path_.empty()) {
    for (int i = static_cast<int>(shortcut_path_.size()) - 1; i >= 0; --i) {
      Tensor temp_grad = this->get_buffer({max_size}, gradient->data_type());
      shortcut_path_[i]->backward(shortcut_grad, temp_grad, mb_id);
      shortcut_grad = temp_grad;
    }
  }

  grad_input->ensure(main_grad->shape());
  DISPATCH_ON_DTYPE_TO_METHOD(ops::add, main_grad->data_ptr(), shortcut_grad->data_ptr(),
                              grad_input->data_ptr(), grad_input->size());
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

std::vector<size_t> ResidualBlock::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
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

  size_t add_complexity = 1;
  for (size_t dim : current_shape) {
    add_complexity *= dim;
  }

  return main_complexity + shortcut_complexity + add_complexity;
}

LayerConfig ResidualBlock::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["activation"] = activation_type_;

  nlohmann::json main_array = nlohmann::json::array();
  for (const auto &layer : main_path_) {
    LayerConfig sub_cfg = layer->get_config();
    nlohmann::json sub_json = sub_cfg.to_json();
    main_array.push_back(sub_json);
  }

  nlohmann::json shortcut_array = nlohmann::json::array();
  for (const auto &layer : shortcut_path_) {
    LayerConfig sub_cfg = layer->get_config();
    nlohmann::json sub_json = sub_cfg.to_json();
    shortcut_array.push_back(sub_json);
  }

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
  return total_bytes;
}

std::unique_ptr<ResidualBlock> ResidualBlock::create_from_config(const LayerConfig &config) {
  std::vector<std::unique_ptr<Layer>> main_path;
  std::vector<std::unique_ptr<Layer>> shortcut_path;
  nlohmann::json main_json = nlohmann::json::parse(config.get<std::string>("main_path"));
  LayerFactory::register_defaults();
  for (const auto &layer_json : main_json) {
    LayerConfig layer_config = LayerConfig::from_json(layer_json);
    auto layer = LayerFactory::create(layer_config);
    main_path.push_back(std::move(layer));
  }
  nlohmann::json shortcut_json = nlohmann::json::parse(config.get<std::string>("shortcut_path"));
  for (const auto &layer_json : shortcut_json) {
    LayerConfig layer_config = LayerConfig::from_json(layer_json);
    auto layer = LayerFactory::create(layer_config);
    shortcut_path.push_back(std::move(layer));
  }

  std::string activation = config.get<std::string>("activation", "relu");
  return std::make_unique<ResidualBlock>(std::move(main_path), std::move(shortcut_path), activation,
                                         config.name);
}

}  // namespace tnn
