/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../tensor/tensor.hpp"
#include "activations.hpp"
#include "optimizers.hpp"
#include "layers_impl/base_layer.hpp"
namespace tnn {

template <typename T>
std::unique_ptr<ActivationFunction<T>>
create_activation(const std::string &name) {
  ActivationFactory<T>::register_defaults();
  return ActivationFactory<T>::create(name);
}

// Forward declarations for specific layers
template <typename T> class DenseLayer;
template <typename T> class ActivationLayer;
template <typename T> class Conv2DLayer;
template <typename T> class MaxPool2DLayer;
template <typename T> class DropoutLayer;
template <typename T> class FlattenLayer;
template <typename T> class LayerFactory;
template <typename T> class BatchNormLayer;

} // namespace tnn

// Layer implementations
#include "layers_impl/base_layer.hpp"
#include "layers_impl/stateless_layer.hpp"
#include "layers_impl/parameterized_layer.hpp"
#include "layers_impl/dense_layer.hpp"
#include "layers_impl/activation_layer.hpp"
#include "layers_impl/conv2d_layer.hpp"
#include "layers_impl/maxpool2d_layer.hpp"
#include "layers_impl/dropout_layer.hpp"
#include "layers_impl/flatten_layer.hpp"
#include "layers_impl/layer_factory.hpp"
#include "layers_impl/batchnorm_layer.hpp"

namespace tnn {

template <typename T = float>
std::unique_ptr<Layer<T>>
dense(size_t input_features, size_t output_features,
      const std::string &activation = "none", bool use_bias = true,
      const std::string &name = "dense") {
  std::unique_ptr<ActivationFunction<T>> act = nullptr;
  if (activation != "none") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    act = factory.create(activation);
  }

  return std::make_unique<DenseLayer<T>>(input_features, output_features,
                                              std::move(act), use_bias, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>>
conv2d(size_t in_channels, size_t out_channels, size_t kernel_h,
       size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
       size_t pad_h = 0, size_t pad_w = 0,
       const std::string &activation = "none", bool use_bias = true,
       const std::string &name = "conv2d") {
  std::unique_ptr<ActivationFunction<T>> act = nullptr;
  if (activation != "none") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    act = factory.create(activation);
  }
  return std::make_unique<Conv2DLayer<T>>(
      in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h,
      pad_w, use_bias, std::move(act), name);
}

template <typename T = float>
std::unique_ptr<Layer<T>>
activation(const std::string &activation_name,
           const std::string &name = "activation") {
  auto factory = ActivationFactory<T>();
  factory.register_defaults();
  auto act = factory.create(activation_name);
  return std::make_unique<ActivationLayer<T>>(std::move(act), name);
}

template <typename T = float>
std::unique_ptr<Layer<T>>
maxpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 0,
          size_t stride_w = 0, size_t pad_h = 0, size_t pad_w = 0,
          const std::string &name = "maxpool2d") {
  return std::make_unique<MaxPool2DLayer<T>>(pool_h, pool_w, stride_h,
                                                  stride_w, pad_h, pad_w, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>> dropout(T dropout_rate,
                                       const std::string &name = "dropout") {
  return std::make_unique<DropoutLayer<T>>(dropout_rate, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>>
batchnorm(size_t num_features, T epsilon = T(1e-5), T momentum = T(0.1),
          bool affine = true, const std::string &name = "batchnorm") {
  return std::make_unique<BatchNormLayer<T>>(num_features, epsilon,
                                                  momentum, affine, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>> flatten(const std::string &name = "flatten") {
  return std::make_unique<FlattenLayer<T>>(name);
}

} // namespace tnn