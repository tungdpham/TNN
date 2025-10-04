/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/dropout_layer.hpp"

#include <stdexcept>

#include "utils/parallel_for.hpp"

namespace tnn {

template <typename T>
DropoutLayer<T>::DropoutLayer(T dropout_rate, const std::string &name)
    : StatelessLayer<T>(name), dropout_rate_(dropout_rate), generator_(std::random_device{}()) {
  if (dropout_rate < T(0) || dropout_rate >= T(1)) {
    throw std::invalid_argument("Dropout rate must be in [0, 1)");
  }
}

template <typename T>
Tensor<T> DropoutLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  if (!this->is_training_) {
    return input;
  }

  Tensor<T> mask(input.shape());
  Tensor<T> output = input;

  std::uniform_real_distribution<T> distribution(T(0), T(1));

  T scale = T(1) / (T(1) - dropout_rate_);

  utils::parallel_for_2d(input.batch_size(), input.channels(), [&](size_t n, size_t c) {
    thread_local std::mt19937 local_generator(std::random_device{}());
    thread_local std::uniform_real_distribution<T> local_distribution(T(0), T(1));
    for (size_t h = 0; h < input.height(); ++h) {
      for (size_t w = 0; w < input.width(); ++w) {
        if (local_distribution(local_generator) < dropout_rate_) {
          mask(n, c, h, w) = T(0);
          output(n, c, h, w) = T(0);
        } else {
          mask(n, c, h, w) = scale;
          output(n, c, h, w) *= scale;
        }
      }
    }
  });

  micro_batch_masks_[micro_batch_id] = mask.clone();
  return output;
}

template <typename T>
Tensor<T> DropoutLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  if (!this->is_training_) {
    return gradient;
  }

  auto it_mask = micro_batch_masks_.find(micro_batch_id);
  if (it_mask == micro_batch_masks_.end()) {
    throw std::runtime_error("No cached mask found for micro-batch ID in DropoutLayer: " +
                             std::to_string(micro_batch_id));
  }
  const Tensor<T> &mask = it_mask->second;

  Tensor<T> grad_input = gradient;

  utils::parallel_for_2d(gradient.batch_size(), gradient.channels(), [&](size_t n, size_t c) {
    for (size_t h = 0; h < gradient.height(); ++h) {
      for (size_t w = 0; w < gradient.width(); ++w) {
        grad_input(n, c, h, w) *= mask(n, c, h, w);
      }
    }
  });

  return grad_input;
}

template <typename T> std::string DropoutLayer<T>::type() const { return "dropout"; }

template <typename T> LayerConfig DropoutLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["dropout_rate"] = dropout_rate_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> DropoutLayer<T>::clone() const {
  return std::make_unique<DropoutLayer<T>>(dropout_rate_, this->name_);
}

template <typename T>
std::vector<size_t>
DropoutLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

template <typename T>
std::unique_ptr<Layer<T>> DropoutLayer<T>::create_from_config(const LayerConfig &config) {
  T dropout_rate = config.get<T>("dropout_rate");
  return std::make_unique<DropoutLayer<T>>(dropout_rate, config.name);
}

template <typename T>
uint64_t DropoutLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  if (!this->is_training_) {
    return 0;
  }

  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  uint64_t rng_flops = num_elements;
  uint64_t mask_flops = num_elements;
  uint64_t scale_flops = static_cast<uint64_t>((1.0 - dropout_rate_) * num_elements);

  return rng_flops + mask_flops + scale_flops;
}

template <typename T>
uint64_t DropoutLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  if (!this->is_training_) {
    return 0;
  }

  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  return num_elements;
}

template <typename T>
uint64_t DropoutLayer<T>::forward_memory_traffic(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t bytes_per_element = sizeof(T);

  uint64_t input_bytes = num_elements * bytes_per_element;

  uint64_t output_bytes = num_elements * bytes_per_element;

  if (!this->is_training_) {

    return input_bytes + output_bytes;
  }

  uint64_t mask_bytes = num_elements * bytes_per_element;

  return input_bytes + output_bytes + mask_bytes;
}

template <typename T>
uint64_t DropoutLayer<T>::backward_memory_traffic(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t bytes_per_element = sizeof(T);

  if (!this->is_training_) {

    return 2 * num_elements * bytes_per_element;
  }

  uint64_t input_grad_bytes = num_elements * bytes_per_element;

  uint64_t mask_bytes = num_elements * bytes_per_element;

  uint64_t output_grad_bytes = num_elements * bytes_per_element;

  return input_grad_bytes + mask_bytes + output_grad_bytes;
}

template <typename T>
double DropoutLayer<T>::forward_arithmetic_intensity(const std::vector<size_t> &input_shape) const {
  uint64_t flops = forward_flops(input_shape);
  uint64_t memory_bytes = forward_memory_traffic(input_shape);

  if (memory_bytes == 0)
    return 0.0;
  return static_cast<double>(flops) / static_cast<double>(memory_bytes);
}

template <typename T>
double
DropoutLayer<T>::backward_arithmetic_intensity(const std::vector<size_t> &input_shape) const {
  uint64_t flops = backward_flops(input_shape);
  uint64_t memory_bytes = backward_memory_traffic(input_shape);

  if (memory_bytes == 0)
    return 0.0;
  return static_cast<double>(flops) / static_cast<double>(memory_bytes);
}

template <typename T>
uint64_t DropoutLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) {

  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t DropoutLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) {
  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template class DropoutLayer<float>;
template class DropoutLayer<double>;

} // namespace tnn
