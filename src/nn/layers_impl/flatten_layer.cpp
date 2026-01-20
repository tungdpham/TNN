/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/flatten_layer.hpp"
#include <stdexcept>

namespace tnn {

FlattenLayer::FlattenLayer(int start_dim, const std::string &name)
    : StatelessLayer(name), start_dim_(start_dim) {}

void FlattenLayer::forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id) {
  micro_batch_original_shapes_[micro_batch_id] = input->shape();

  std::vector<size_t> output_shape = compute_output_shape(input->shape());
  output->ensure(output_shape);

  input->copy_to(output);
}

void FlattenLayer::backward_impl(const Tensor &gradient, Tensor &grad_input,
                                 size_t micro_batch_id) {
  auto it = micro_batch_original_shapes_.find(micro_batch_id);
  if (it == micro_batch_original_shapes_.end()) {
    throw std::runtime_error("No cached shape found for micro-batch ID in FlattenLayer: " +
                             std::to_string(micro_batch_id));
  }
  const std::vector<size_t> &original_shape = it->second;

  grad_input->ensure(original_shape);
  gradient->copy_to(grad_input);
}

std::string FlattenLayer::type() const { return "flatten"; }

LayerConfig FlattenLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["start_dim"] = start_dim_;
  return config;
}

std::unique_ptr<Layer> FlattenLayer::clone() const {
  return std::make_unique<FlattenLayer>(this->start_dim_, this->name_);
}

std::vector<size_t>
FlattenLayer::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::invalid_argument("FlattenLayer expects non-empty input shape");
  }

  std::vector<size_t> output_shape;

  output_shape.push_back(input_shape[0]);

  size_t flat_dim = 1;
  int start = std::max(1, start_dim_);

  for (int i = 1; i < start && i < static_cast<int>(input_shape.size()); ++i) {
    output_shape.push_back(input_shape[i]);
  }

  for (size_t i = static_cast<size_t>(start); i < input_shape.size(); ++i) {
    flat_dim *= input_shape[i];
  }

  output_shape.push_back(flat_dim);

  return output_shape;
}

std::unique_ptr<Layer> FlattenLayer::create_from_config(const LayerConfig &config) {
  int start_dim = config.get<int>("start_dim", 1);
  return std::make_unique<FlattenLayer>(start_dim, config.name);
}

uint64_t FlattenLayer::forward_flops(const std::vector<size_t> &input_shape) const { return 0; }

uint64_t FlattenLayer::backward_flops(const std::vector<size_t> &input_shape) const { return 0; }

} // namespace tnn
