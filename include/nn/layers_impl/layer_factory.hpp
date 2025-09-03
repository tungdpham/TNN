/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "../layers.hpp"

namespace tnn {

template <typename T = float> class LayerFactory {
private:
  static std::unordered_map<
      std::string,
      std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)>>
      creators_;

public:
  static void register_layer(
      const std::string &type,
      std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)> creator) {
    creators_[type] = creator;
  }

  static std::unique_ptr<Layer<T>> create(const std::string &type,
                                          const LayerConfig &config) {
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
    register_layer(
        "dense", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
          size_t input_features = config.get<size_t>("input_features");
          size_t output_features = config.get<size_t>("output_features");
          bool use_bias = config.get<bool>("use_bias", true);
          std::string activation_name =
              config.get<std::string>("activation", "none");

          std::unique_ptr<ActivationFunction<T>> activation = nullptr;
          if (activation_name != "none") {
            auto factory = ActivationFactory<T>();
            factory.register_defaults();
            activation = factory.create(activation_name);
          }

          return std::make_unique<DenseLayer<T>>(
              input_features, output_features, std::move(activation), use_bias,
              config.name);
        });

    register_layer(
        "conv2d", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
          size_t in_channels = config.get<size_t>("in_channels");
          size_t out_channels = config.get<size_t>("out_channels");
          size_t kernel_h = config.get<size_t>("kernel_h");
          size_t kernel_w = config.get<size_t>("kernel_w");
          size_t stride_h = config.get<size_t>("stride_h", 1);
          size_t stride_w = config.get<size_t>("stride_w", 1);
          size_t pad_h = config.get<size_t>("pad_h", 0);
          size_t pad_w = config.get<size_t>("pad_w", 0);
          bool use_bias = config.get<bool>("use_bias", true);
          std::string activation_name =
              config.get<std::string>("activation", "none");

          std::unique_ptr<ActivationFunction<T>> activation = nullptr;
          if (activation_name != "none") {
            auto factory = ActivationFactory<T>();
            factory.register_defaults();
            activation = factory.create(activation_name);
          }

          return std::make_unique<Conv2DLayer<T>>(
              in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
              pad_h, pad_w, use_bias, std::move(activation), config.name);
        });

    register_layer("activation",
                   [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
                     std::string activation_name =
                         config.get<std::string>("activation");
                     auto factory = ActivationFactory<T>();
                     factory.register_defaults();
                     auto activation = factory.create(activation_name);
                     if (!activation) {
                       throw std::invalid_argument(
                           "Failed to create activation: " + activation_name);
                     }
                     return std::make_unique<ActivationLayer<T>>(
                         std::move(activation), config.name);
                   });

    register_layer("maxpool2d",
                   [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
                     size_t pool_h = config.get<size_t>("pool_h");
                     size_t pool_w = config.get<size_t>("pool_w");
                     size_t stride_h = config.get<size_t>("stride_h", 0);
                     size_t stride_w = config.get<size_t>("stride_w", 0);
                     size_t pad_h = config.get<size_t>("pad_h", 0);
                     size_t pad_w = config.get<size_t>("pad_w", 0);

                     return std::make_unique<MaxPool2DLayer<T>>(
                         pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                         config.name);
                   });

    register_layer(
        "dropout", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
          T dropout_rate = config.get<T>("dropout_rate");
          return std::make_unique<DropoutLayer<T>>(dropout_rate, config.name);
        });

    register_layer("batchnorm",
                   [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
                     size_t num_features = config.get<size_t>("num_features");
                     T epsilon = config.get<T>("epsilon", T(1e-5));
                     T momentum = config.get<T>("momentum", T(0.1));
                     bool affine = config.get<bool>("affine", true);
                     return std::make_unique<BatchNormLayer<T>>(
                         num_features, epsilon, momentum, affine, config.name);
                   });

    register_layer("flatten",
                   [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
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

template <typename T>
std::unordered_map<
    std::string, std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)>>
    LayerFactory<T>::creators_;

} // namespace tnn
