/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/flatten_layer.hpp"

#include <stdexcept>

namespace synet {

FlattenLayerImpl::FlattenLayerImpl(int start_dim, int end_dim, const std::string &name)
    : SISOLayerImpl(name),
      start_dim_(start_dim),
      end_dim_(end_dim) {}

Tensor FlattenLayerImpl::forward_impl(const Tensor &input, Residuals &residuals) {
  Tensor shape_tensor = Tensor({input.shape().size()});
  std::copy(input.shape().begin(), input.shape().end(), shape_tensor.data_as<size_t>());
  residuals["original_shape"] = shape_tensor;

  Vec<size_t> output_shape = compute_output_shape(input.shape());
  Tensor output = get_tensor(output_shape, io_dtype_);

  input.copy_to(output);
  return output;
}

Tensor FlattenLayerImpl::backward_impl(const Tensor &grad_output, Residuals &residuals) {
  // const Vec<size_t> &original_shape = residuals["original_shape"];
  const Tensor &shape_tensor = residuals["original_shape"];
  if (!shape_tensor) {
    throw std::runtime_error(
        "No cached original shape found for backward pass in FlattenLayerImpl");
  }
  Vec<size_t> original_shape(shape_tensor.size());
  std::copy(shape_tensor.data_as<size_t>(), shape_tensor.data_as<size_t>() + shape_tensor.size(),
            original_shape.begin());

  size_t expected_size =
      std::accumulate(original_shape.begin(), original_shape.end(), 1, std::multiplies<size_t>());
  if (grad_output.size() != expected_size) {
    throw std::runtime_error(
        "Gradient size does not match original input size in FlattenLayerImpl");
  }
  Tensor grad_input = get_tensor(original_shape, io_dtype_);
  grad_output.copy_to(grad_input);
  return grad_input;
}

LayerConfig FlattenLayerImpl::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.type = this->type();
  config.set("start_dim", start_dim_);
  config.set("end_dim", end_dim_);
  return config;
}

Vec<size_t> FlattenLayerImpl::compute_output_shape(const Vec<size_t> &input_shape) const {
  if (input_shape.empty()) {
    throw std::invalid_argument("FlattenLayerImpl expects non-empty input shape");
  }

  Vec<size_t> output_shape;

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

std::shared_ptr<FlattenLayerImpl> FlattenLayerImpl::create_from_config(const LayerConfig &config) {
  int start_dim = config.get<int>("start_dim", 1);
  int end_dim = config.get<int>("end_dim", -1);
  return std::make_shared<FlattenLayerImpl>(start_dim, end_dim, config.name);
}

}  // namespace synet
