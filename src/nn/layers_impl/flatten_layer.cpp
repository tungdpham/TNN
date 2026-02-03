/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/flatten_layer.hpp"

#include <stdexcept>

namespace tnn {

FlattenLayer::FlattenLayer(int start_dim, int end_dim, const std::string &name)
    : StatelessLayer(name), start_dim_(start_dim), end_dim_(end_dim) {}

void FlattenLayer::forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id) {
  micro_batch_original_shapes_[mb_id] = input->shape();

  std::vector<size_t> output_shape = compute_output_shape(input->shape());
  output->ensure(output_shape);

  input->copy_to(output);
}

void FlattenLayer::backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                                 size_t mb_id) {
  auto it = micro_batch_original_shapes_.find(mb_id);
  if (it == micro_batch_original_shapes_.end()) {
    throw std::runtime_error("No cached shape found for micro-batch ID in FlattenLayer: " +
                             std::to_string(mb_id));
  }
  const std::vector<size_t> &original_shape = it->second;
  size_t expected_size =
      std::accumulate(original_shape.begin(), original_shape.end(), 1, std::multiplies<size_t>());
  if (gradient->size() != expected_size) {
    throw std::runtime_error("Gradient size does not match original input size in FlattenLayer");
  }
  grad_input->ensure(original_shape);
  gradient->copy_to(grad_input);
}

LayerConfig FlattenLayer::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.parameters["start_dim"] = start_dim_;
  config.parameters["end_dim"] = end_dim_;
  return config;
}

std::unique_ptr<Layer> FlattenLayer::clone() const {
  return std::make_unique<FlattenLayer>(this->start_dim_, this->end_dim_, this->name_);
}

std::vector<size_t> FlattenLayer::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::invalid_argument("FlattenLayer expects non-empty input shape");
  }

  std::vector<size_t> output_shape;

  output_shape.push_back(input_shape[0]);

  int start = std::max(1, start_dim_);
  int end = (end_dim_ < 0) ? static_cast<int>(input_shape.size())
                           : std::min(end_dim_ + 1, static_cast<int>(input_shape.size()));

  // Add dimensions before start_dim
  for (int i = 1; i < start && i < static_cast<int>(input_shape.size()); ++i) {
    output_shape.push_back(input_shape[i]);
  }

  // Flatten dimensions from start_dim to end_dim
  size_t flat_dim = 1;
  for (int i = start; i < end; ++i) {
    flat_dim *= input_shape[i];
  }
  output_shape.push_back(flat_dim);

  // Add dimensions after end_dim
  for (int i = end; i < static_cast<int>(input_shape.size()); ++i) {
    output_shape.push_back(input_shape[i]);
  }

  return output_shape;
}

std::unique_ptr<FlattenLayer> FlattenLayer::create_from_config(const LayerConfig &config) {
  int start_dim = config.get<int>("start_dim", 1);
  int end_dim = config.get<int>("end_dim", -1);
  return std::make_unique<FlattenLayer>(start_dim, end_dim, config.name);
}

uint64_t FlattenLayer::forward_flops(const std::vector<size_t> &input_shape) const { return 0; }

uint64_t FlattenLayer::backward_flops(const std::vector<size_t> &input_shape) const { return 0; }

}  // namespace tnn
