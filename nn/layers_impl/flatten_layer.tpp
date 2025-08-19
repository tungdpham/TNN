#include "flatten_layer.hpp"

#include <stdexcept>

namespace tnn {

// Constructor
template <typename T>
FlattenLayer<T>::FlattenLayer(const std::string &name)
    : StatelessLayer<T>(name) {}

// Forward pass
template <typename T>
Tensor<T> FlattenLayer<T>::forward(const Tensor<T> &input, int micro_batch_id) {
  micro_batch_original_shapes_[micro_batch_id] = input.shape();

  size_t batch_size = input.batch_size();
  size_t features = input.channels() * input.height() * input.width();

  Tensor<T> output = input.reshape({batch_size, features, 1, 1});

  return output;
}

// Backward pass
template <typename T>
Tensor<T> FlattenLayer<T>::backward(const Tensor<T> &grad_output,
                                    int micro_batch_id) {
  auto it = micro_batch_original_shapes_.find(micro_batch_id);
  if (it == micro_batch_original_shapes_.end()) {
    throw std::runtime_error(
        "No cached shape found for micro-batch ID in FlattenLayer: " +
        std::to_string(micro_batch_id));
  }
  const std::vector<size_t> &original_shape = it->second;

  // Reshape back to original shape
  Tensor<T> grad_input = grad_output.reshape(original_shape);

  return grad_input;
}

// Virtual method implementations
template <typename T>
std::string FlattenLayer<T>::type() const {
  return "flatten";
}

template <typename T>
LayerConfig FlattenLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  return config;
}

template <typename T>
std::unique_ptr<Layer<T>> FlattenLayer<T>::clone() const {
  return std::make_unique<FlattenLayer<T>>(this->name_);
}

template <typename T>
std::vector<size_t> FlattenLayer<T>::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("FlattenLayer expects 4D input");
  }

  size_t features = input_shape[1] * input_shape[2] * input_shape[3];
  return {input_shape[0], features, 1, 1};
}

template <typename T>
std::unique_ptr<Layer<T>>
FlattenLayer<T>::create_from_config(const LayerConfig &config) {
  return std::make_unique<FlattenLayer<T>>(config.name);
}

} // namespace tnn
