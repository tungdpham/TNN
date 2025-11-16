/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "activations.hpp"
#include "layers_impl/base_layer.hpp"

namespace tnn {

template <typename T>
std::unique_ptr<ActivationFunction<T>> create_activation(const std::string &name) {
  ActivationFactory<T>::register_defaults();
  return ActivationFactory<T>::create(name);
}

template <typename T> class DenseLayer;
template <typename T> class ActivationLayer;
template <typename T> class Conv2DLayer;
template <typename T> class MaxPool2DLayer;
template <typename T> class DropoutLayer;
template <typename T> class FlattenLayer;
template <typename T> class BatchNormLayer;

} // namespace tnn

// Wrapper to include all layer implementations
#include "layers_impl/activation_layer.hpp"
#include "layers_impl/base_layer.hpp"
#include "layers_impl/batchnorm_layer.hpp"
#include "layers_impl/conv2d_layer.hpp"
#include "layers_impl/dense_layer.hpp"
#include "layers_impl/dropout_layer.hpp"
#include "layers_impl/flatten_layer.hpp"
#include "layers_impl/maxpool2d_layer.hpp"

namespace tnn {
template <typename T = float>
std::unique_ptr<Layer<T>> dense_layer(size_t input_features, size_t output_features,
                                      const std::string &activation = "none", bool use_bias = true,
                                      const std::string &name = "dense") {
  std::unique_ptr<ActivationFunction<T>> act = nullptr;
  if (activation != "none" && activation != "linear") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    act = factory.create(activation);
  }

  return std::make_unique<DenseLayer<T>>(input_features, output_features, std::move(act), use_bias,
                                         name);
}

template <typename T = float>
std::unique_ptr<Layer<T>> conv2d_layer(size_t in_channels, size_t out_channels, size_t kernel_h,
                                       size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
                                       size_t pad_h = 0, size_t pad_w = 0, bool use_bias = true,
                                       const std::string &name = "conv2d") {
  return std::make_unique<Conv2DLayer<T>>(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                          stride_w, pad_h, pad_w, use_bias, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>> activation_layer(const std::string &activation_name,
                                           const std::string &name = "activation") {
  auto factory = ActivationFactory<T>();
  factory.register_defaults();
  auto act = factory.create(activation_name);
  return std::make_unique<ActivationLayer<T>>(std::move(act), name);
}

template <typename T = float>
std::unique_ptr<Layer<T>> maxpool2d_layer(size_t pool_h, size_t pool_w, size_t stride_h = 0,
                                          size_t stride_w = 0, size_t pad_h = 0, size_t pad_w = 0,
                                          const std::string &name = "maxpool2d") {
  return std::make_unique<MaxPool2DLayer<T>>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                             name);
}

template <typename T = float>
std::unique_ptr<Layer<T>> dropout_layer(T dropout_rate, const std::string &name = "dropout") {
  return std::make_unique<DropoutLayer<T>>(dropout_rate, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>> batchnorm_layer(size_t num_features, T epsilon = T(1e-5),
                                          T momentum = T(0.1), bool affine = true,
                                          const std::string &name = "batchnorm") {
  return std::make_unique<BatchNormLayer<T>>(num_features, epsilon, momentum, affine, name);
}

template <typename T = float>
std::unique_ptr<Layer<T>> flatten_layer(const std::string &name = "flatten") {
  return std::make_unique<FlattenLayer<T>>(name);
}

template <typename T = float> class LayerFactory {
private:
  static std::unordered_map<std::string,
                            std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)>>
      creators_;

public:
  static void
  register_layer(const std::string &type,
                 std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)> creator) {
    creators_[type] = creator;
  }

  static std::unique_ptr<Layer<T>> create(const std::string &type, const LayerConfig &config) {
    auto it = creators_.find(type);
    if (it != creators_.end()) {
      return it->second(config);
    }
    throw std::invalid_argument("Unknown layer type: " + type);
  }

  static std::unique_ptr<Layer<T>> create(const LayerConfig &config) {
    return create(config.get<std::string>("type"), config);
  }

  static void register_defaults() {
    register_layer("dense", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t input_features = config.get<size_t>("input_features");
      size_t output_features = config.get<size_t>("output_features");
      bool use_bias = config.get<bool>("use_bias", true);
      std::string activation_name = config.get<std::string>("activation", "none");

      std::unique_ptr<ActivationFunction<T>> activation = nullptr;
      if (activation_name != "none") {
        auto factory = ActivationFactory<T>();
        factory.register_defaults();
        activation = factory.create(activation_name);
      }

      return std::make_unique<DenseLayer<T>>(input_features, output_features, std::move(activation),
                                             use_bias, config.name);
    });

    register_layer("conv2d", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t in_channels = config.get<size_t>("in_channels");
      size_t out_channels = config.get<size_t>("out_channels");
      size_t kernel_h = config.get<size_t>("kernel_h");
      size_t kernel_w = config.get<size_t>("kernel_w");
      size_t stride_h = config.get<size_t>("stride_h", 1);
      size_t stride_w = config.get<size_t>("stride_w", 1);
      size_t pad_h = config.get<size_t>("pad_h", 0);
      size_t pad_w = config.get<size_t>("pad_w", 0);
      bool use_bias = config.get<bool>("use_bias", true);

      return std::make_unique<Conv2DLayer<T>>(in_channels, out_channels, kernel_h, kernel_w,
                                              stride_h, stride_w, pad_h, pad_w, use_bias,
                                              config.name);
    });

    register_layer("activation", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      std::string activation_name = config.get<std::string>("activation");
      auto factory = ActivationFactory<T>();
      factory.register_defaults();
      auto activation = factory.create(activation_name);
      if (!activation) {
        throw std::invalid_argument("Failed to create activation: " + activation_name);
      }
      return std::make_unique<ActivationLayer<T>>(std::move(activation), config.name);
    });

    register_layer("maxpool2d", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t pool_h = config.get<size_t>("pool_h");
      size_t pool_w = config.get<size_t>("pool_w");
      size_t stride_h = config.get<size_t>("stride_h", 0);
      size_t stride_w = config.get<size_t>("stride_w", 0);
      size_t pad_h = config.get<size_t>("pad_h", 0);
      size_t pad_w = config.get<size_t>("pad_w", 0);

      return std::make_unique<MaxPool2DLayer<T>>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                                 config.name);
    });

    register_layer("dropout", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      T dropout_rate = config.get<T>("dropout_rate");
      return std::make_unique<DropoutLayer<T>>(dropout_rate, config.name);
    });

    register_layer("batchnorm", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t num_features = config.get<size_t>("num_features");
      T epsilon = config.get<T>("epsilon", T(1e-5));
      T momentum = config.get<T>("momentum", T(0.1));
      bool affine = config.get<bool>("affine", true);
      return std::make_unique<BatchNormLayer<T>>(num_features, epsilon, momentum, affine,
                                                 config.name);
    });

    register_layer("flatten", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      return std::make_unique<FlattenLayer<T>>(config.name);
    });
  }

  static std::vector<std::string> available_types() {
    std::vector<std::string> types;
    for (const auto &pair : creators_) {
      types.push_back(pair.first);
    }
    return types;
  }
};

template <typename T = float> class LayerBuilder {
private:
  std::vector<std::unique_ptr<Layer<T>>> layers_;
  std::vector<size_t> input_shape_;
  bool input_shape_set_ = false;

public:
  explicit LayerBuilder(const std::string &name = "Block") {}

  std::vector<size_t> get_current_shape() const {
    if (!input_shape_set_) {
      throw std::runtime_error("Input shape must be set before adding layers. "
                               "Use .input() method first.");
    }

    std::vector<size_t> shape_with_batch = {1};
    shape_with_batch.insert(shape_with_batch.end(), input_shape_.begin(), input_shape_.end());

    // Compute output shape by passing through all layers
    std::vector<size_t> current_shape = shape_with_batch;
    for (const auto &layer : layers_) {
      current_shape = layer->compute_output_shape(current_shape);
    }
    return current_shape;
  }

  size_t get_feature_count() const {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.empty()) {
      throw std::runtime_error("Cannot compute feature count from empty shape");
    }

    size_t feature_count = 1;
    for (size_t i = 1; i < current_shape.size(); ++i) {
      feature_count *= current_shape[i];
    }

    return feature_count;
  }

  LayerBuilder &input(const std::vector<size_t> &shape) {
    input_shape_ = shape;
    input_shape_set_ = true;
    return *this;
  }

  LayerBuilder &dense(size_t output_features, const std::string &activation = "none",
                      bool use_bias = true, const std::string &name = "") {

    size_t input_features = get_feature_count();

    auto layer = dense_layer<T>(input_features, output_features, activation, use_bias,
                                name.empty() ? "dense_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &conv2d(size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h = 1,
                       size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                       bool use_bias = true, const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 4) {
      throw std::runtime_error("Conv2D requires 4D input (batch, channels, "
                               "height, width). Current shape has " +
                               std::to_string(current_shape.size()) + " dimensions.");
    }

    size_t in_channels = current_shape[1];

    auto layer = conv2d_layer<T>(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
                                 pad_h, pad_w, use_bias,
                                 name.empty() ? "conv2d_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &batchnorm(T epsilon = T(1e-5), T momentum = T(0.1), bool affine = true,
                          const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 2) {
      throw std::runtime_error("BatchNorm requires at least 2D input (batch, features)");
    }

    size_t num_features;
    if (current_shape.size() == 2) {

      num_features = current_shape[1];
    } else if (current_shape.size() >= 4) {

      num_features = current_shape[1];
    } else {

      num_features = current_shape[1];
    }

    auto layer =
        batchnorm_layer<T>(num_features, epsilon, momentum, affine,
                           name.empty() ? "batchnorm_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &activation(const std::string &activation_name, const std::string &name = "") {
    auto layer = activation_layer<T>(
        activation_name, name.empty() ? "activation_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &maxpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 0, size_t stride_w = 0,
                          size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "") {
    auto layer =
        maxpool2d_layer<T>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                           name.empty() ? "maxpool2d_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &dropout(T dropout_rate, const std::string &name = "") {
    auto layer = dropout_layer<T>(
        dropout_rate, name.empty() ? "dropout_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &flatten(const std::string &name = "") {
    auto layer =
        flatten_layer<T>(name.empty() ? "flatten_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &add_layer(std::unique_ptr<Layer<T>> layer) {
    layers_.push_back(std::move(layer));
    return *this;
  }

  std::vector<std::unique_ptr<Layer<T>>> build() {
    if (!input_shape_set_) {
      throw std::runtime_error("Input shape must be set before building block. "
                               "Use .input() method.");
    }
    try {
      std::vector<size_t> shape_with_batch = {1};
      shape_with_batch.insert(shape_with_batch.end(), input_shape_.begin(), input_shape_.end());

      std::vector<size_t> output_shape = get_current_shape();
      std::cout << "Block built successfully. Input shape (without batch): (";
      for (size_t i = 0; i < input_shape_.size(); ++i) {
        if (i > 0)
          std::cout << ", ";
        std::cout << input_shape_[i];
      }
      std::cout << ") -> Output shape (without batch): (";

      for (size_t i = 1; i < output_shape.size(); ++i) {
        if (i > 1)
          std::cout << ", ";
        std::cout << output_shape[i];
      }
      std::cout << ")" << std::endl;
    } catch (const std::exception &e) {
      throw std::runtime_error("Shape inference failed during build: " + std::string(e.what()));
    }

    return std::move(layers_);
  }

  const std::vector<size_t> &get_input_shape() const { return input_shape_; }
  bool is_input_shape_set() const { return input_shape_set_; }
};

template <typename T>
std::unordered_map<std::string, std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)>>
    LayerFactory<T>::creators_;

} // namespace tnn