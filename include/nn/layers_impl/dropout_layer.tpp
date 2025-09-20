/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "dropout_layer.hpp"

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
uint32_t DropoutLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) {
  // random number generation: O(N) where N is number of elements
  // mask application: O(N) where N is number of elements
  return std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>()) * 2;
}

template <typename T>
uint32_t DropoutLayer<T>::backward_complexity(const std::vector<size_t> &gradient_shape) {
  return std::accumulate(gradient_shape.begin(), gradient_shape.end(), 1,
                         std::multiplies<size_t>());
}

} // namespace tnn
