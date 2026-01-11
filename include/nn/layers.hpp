/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "activations.hpp"
#include "layers_impl/base_layer.hpp"

namespace tnn {

template <typename T>
std::unique_ptr<EWActivationFunction<T>> create_activation(const std::string &name) {
  ActivationFactory<T>::register_defaults();
  return ActivationFactory<T>::create(name);
}

template <typename T> class DenseLayer;
template <typename T> class ActivationLayer;
template <typename T> class Conv1DLayer;
template <typename T> class Conv2DLayer;
template <typename T> class MaxPool2DLayer;
template <typename T> class AvgPool2DLayer;
template <typename T> class DropoutLayer;
template <typename T> class FlattenLayer;
template <typename T> class BatchNormLayer;
template <typename T> class GroupNormLayer;
template <typename T> class LayerNormLayer;
template <typename T> class ClassTokenLayer;
template <typename T> class PositionalEmbeddingLayer;
template <typename T> class EmbeddingLayer;
template <typename T> class CausalAttentionBlock;
template <typename T> class FullAttentionBlock;
template <typename T> class FlashAttentionBlock;
template <typename T> class ResidualBlock;
template <typename T> class SliceLayer;
template <typename T> class TransposeLayer;

} // namespace tnn

// Wrapper to include all layer implementations
#include "blocks_impl/causal_attention_block.hpp"
#include "blocks_impl/flash_attention_block.hpp"
#include "blocks_impl/full_attention_block.hpp"
#include "blocks_impl/residual_block.hpp"
#include "layers_impl/activation_layer.hpp"
#include "layers_impl/avgpool2d_layer.hpp"
#include "layers_impl/base_layer.hpp"
#include "layers_impl/batchnorm_layer.hpp"
#include "layers_impl/class_token_layer.hpp"
#include "layers_impl/conv1d_layer.hpp"
#include "layers_impl/conv2d_layer.hpp"
#include "layers_impl/dense_layer.hpp"
#include "layers_impl/dropout_layer.hpp"
#include "layers_impl/embedding_layer.hpp"
#include "layers_impl/flatten_layer.hpp"
#include "layers_impl/groupnorm_layer.hpp"
#include "layers_impl/layer_norm_layer.hpp"
#include "layers_impl/maxpool2d_layer.hpp"
#include "layers_impl/positional_embedding_layer.hpp"
#include "layers_impl/slice_layer.hpp"
#include "layers_impl/transpose_layer.hpp"

namespace tnn {
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

      return std::make_unique<DenseLayer<T>>(input_features, output_features, use_bias,
                                             config.name);
    });

    register_layer("conv1d", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t in_channels = config.get<size_t>("in_channels");
      size_t out_channels = config.get<size_t>("out_channels");
      size_t kernel_size = config.get<size_t>("kernel_size");
      size_t stride = config.get<size_t>("stride", 1);
      size_t padding = config.get<size_t>("padding", 0);
      bool use_bias = config.get<bool>("use_bias", true);

      return std::make_unique<Conv1DLayer<T>>(in_channels, out_channels, kernel_size, stride,
                                              padding, use_bias, config.name);
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

    register_layer("avgpool2d", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t pool_h = config.get<size_t>("pool_h");
      size_t pool_w = config.get<size_t>("pool_w");
      size_t stride_h = config.get<size_t>("stride_h", 0);
      size_t stride_w = config.get<size_t>("stride_w", 0);
      size_t pad_h = config.get<size_t>("pad_h", 0);
      size_t pad_w = config.get<size_t>("pad_w", 0);

      return std::make_unique<AvgPool2DLayer<T>>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
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

    register_layer("groupnorm", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t num_groups = config.get<size_t>("num_groups");
      size_t num_channels = config.get<size_t>("num_channels");
      T epsilon = config.get<T>("epsilon", T(1e-5));
      bool affine = config.get<bool>("affine", true);
      return std::make_unique<GroupNormLayer<T>>(num_groups, num_channels, epsilon, affine,
                                                 config.name);
    });

    register_layer("layer_norm", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t normalized_shape = config.get<size_t>("normalized_shape");
      T epsilon = config.get<T>("epsilon", T(1e-5));
      bool affine = config.get<bool>("affine", true);
      return std::make_unique<LayerNormLayer<T>>(normalized_shape, epsilon, affine, config.name);
    });

    register_layer("flatten", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      return FlattenLayer<T>::create_from_config(config);
    });

    register_layer("class_token", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t embed_dim = config.get<size_t>("embed_dim");
      return std::make_unique<ClassTokenLayer<T>>(embed_dim, config.name);
    });

    register_layer("pos_embedding", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t embed_dim = config.get<size_t>("embed_dim");
      size_t seq_len = config.get<size_t>("seq_len");
      return std::make_unique<PositionalEmbeddingLayer<T>>(embed_dim, seq_len, config.name);
    });

    register_layer("slice", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      return SliceLayer<T>::create_from_config(config);
    });

    register_layer("full_attention", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t embed_dim = config.get<size_t>("embed_dim");
      size_t num_heads = config.get<size_t>("num_heads");
      return std::make_unique<FullAttentionBlock<T>>(embed_dim, num_heads, config.name);
    });

    register_layer("flash_attention", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t embed_dim = config.get<size_t>("embed_dim");
      size_t num_heads = config.get<size_t>("num_heads");
      return std::make_unique<FlashAttentionBlock<T>>(embed_dim, num_heads, config.name);
    });

    register_layer("embedding", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t vocab_size = config.get<size_t>("vocab_size");
      size_t embed_dim = config.get<size_t>("embed_dim");
      size_t padding_idx = config.get<size_t>("padding_idx", static_cast<size_t>(-1));
      return std::make_unique<EmbeddingLayer<T>>(vocab_size, embed_dim, config.name, padding_idx);
    });

    register_layer("causal_attention", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      size_t embed_dim = config.get<size_t>("embed_dim");
      size_t num_heads = config.get<size_t>("num_heads");
      return std::make_unique<CausalAttentionBlock<T>>(embed_dim, num_heads, config.name);
    });

    register_layer("residual_block", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      std::string activation = config.get<std::string>("activation", "relu");
      bool has_projection = config.get<bool>("has_projection", false);
      std::string main_path_str = config.get<std::string>("main_path", std::string("[]"));
      std::string shortcut_path_str = config.get<std::string>("shortcut_path", std::string("[]"));

      nlohmann::json main_json = nlohmann::json::parse(main_path_str);
      nlohmann::json shortcut_json = nlohmann::json::parse(shortcut_path_str);

      LayerFactory<T> sub_factory;
      sub_factory.register_defaults();

      std::vector<std::unique_ptr<Layer<T>>> main_layers;
      for (const auto &sub : main_json) {
        LayerConfig sub_cfg;
        sub_cfg.name = sub.value("name", "");
        if (sub.contains("parameters")) {
          for (const auto &[k, v] : sub["parameters"].items()) {
            if (v.is_number_integer()) {
              sub_cfg.parameters[k] = v.template get<size_t>();
            } else if (v.is_number_float()) {
              sub_cfg.parameters[k] = v.template get<float>();
            } else if (v.is_boolean()) {
              sub_cfg.parameters[k] = v.template get<bool>();
            } else if (v.is_string()) {
              sub_cfg.parameters[k] = v.template get<std::string>();
            }
          }
        }
        std::string sub_type = sub.value("type", "");
        main_layers.push_back(sub_factory.create(sub_type, sub_cfg));
      }

      std::vector<std::unique_ptr<Layer<T>>> shortcut_layers;
      if (has_projection) {
        for (const auto &sub : shortcut_json) {
          LayerConfig sub_cfg;
          sub_cfg.name = sub.value("name", "");
          if (sub.contains("parameters")) {
            for (const auto &[k, v] : sub["parameters"].items()) {
              if (v.is_number_integer()) {
                sub_cfg.parameters[k] = v.template get<size_t>();
              } else if (v.is_number_float()) {
                sub_cfg.parameters[k] = v.template get<float>();
              } else if (v.is_boolean()) {
                sub_cfg.parameters[k] = v.template get<bool>();
              } else if (v.is_string()) {
                sub_cfg.parameters[k] = v.template get<std::string>();
              }
            }
          }
          std::string sub_type = sub.value("type", "");
          shortcut_layers.push_back(sub_factory.create(sub_type, sub_cfg));
        }
      }

      return std::make_unique<ResidualBlock<T>>(std::move(main_layers), std::move(shortcut_layers),
                                                activation, config.name);
    });

    register_layer("transpose", [](const LayerConfig &config) -> std::unique_ptr<Layer<T>> {
      return std::make_unique<TransposeLayer<T>>(config.name);
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

    // Compute output shape by passing through all layers
    std::vector<size_t> current_shape = input_shape_;

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

  LayerBuilder &input(const std::vector<size_t> &batchless_shape) {
    input_shape_ = {1};
    input_shape_.insert(input_shape_.end(), batchless_shape.begin(), batchless_shape.end());

    input_shape_set_ = true;
    return *this;
  }

  LayerBuilder &dense(size_t output_features, bool use_bias = true, const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();
    size_t input_features = current_shape.back();

    auto layer = std::make_unique<DenseLayer<T>>(
        input_features, output_features, use_bias,
        name.empty() ? "dense_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &conv1d(size_t out_channels, size_t kernel_size, size_t stride = 1,
                       size_t padding = 0, bool use_bias = true, const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 3) {
      throw std::runtime_error("Conv1D requires at least 3D input (batch, channels, length). "
                               "Current shape has " +
                               std::to_string(current_shape.size()) + " dimensions.");
    }

    size_t in_channels = current_shape[1];

    auto layer = std::make_unique<Conv1DLayer<T>>(
        in_channels, out_channels, kernel_size, stride, padding, use_bias,
        name.empty() ? "conv1d_" + std::to_string(layers_.size()) : name);
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

    auto layer = std::make_unique<Conv2DLayer<T>>(
        in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, use_bias,
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

    auto layer = std::make_unique<BatchNormLayer<T>>(
        num_features, epsilon, momentum, affine,
        name.empty() ? "batchnorm_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &groupnorm(T num_groups, T epsilon = T(1e-5), bool affine = true,
                          const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 2) {
      throw std::runtime_error("GroupNorm requires at least 2D input (batch, features)");
    }

    size_t num_channels;
    if (current_shape.size() == 2) {

      num_channels = current_shape[1];
    } else if (current_shape.size() >= 4) {

      num_channels = current_shape[1];
    } else {

      num_channels = current_shape[1];
    }

    auto layer = std::make_unique<GroupNormLayer<T>>(
        num_groups, num_channels, epsilon, affine,
        name.empty() ? "groupnorm_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &layernorm(T epsilon = T(1e-5), bool affine = true, const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 2) {
      throw std::runtime_error("LayerNorm requires at least 2D input (batch, features)");
    }

    size_t num_features = current_shape.back();

    auto layer = std::make_unique<LayerNormLayer<T>>(
        num_features, epsilon, affine,
        name.empty() ? "layernorm_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &activation(const std::string &activation_name, const std::string &name = "") {
    auto factory = ActivationFactory<T>();
    factory.register_defaults();
    auto act = factory.create(activation_name);
    auto layer = std::make_unique<ActivationLayer<T>>(
        std::move(act), name.empty() ? "activation_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &maxpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 0, size_t stride_w = 0,
                          size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "") {
    auto layer = std::make_unique<MaxPool2DLayer<T>>(
        pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
        name.empty() ? "maxpool2d_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &avgpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 1, size_t stride_w = 1,
                          size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "") {
    auto layer = std::make_unique<AvgPool2DLayer<T>>(
        pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
        name.empty() ? "avgpool2d_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &dropout(T dropout_rate, const std::string &name = "") {
    auto layer = std::make_unique<DropoutLayer<T>>(
        dropout_rate, name.empty() ? "dropout_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &flatten(int start_dim = 1, const std::string &name = "") {
    auto layer = std::make_unique<FlattenLayer<T>>(
        start_dim, name.empty() ? "flatten_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &class_token(size_t embed_dim, const std::string &name = "") {
    auto layer = std::make_unique<ClassTokenLayer<T>>(
        embed_dim, name.empty() ? "class_token_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &positional_embedding(size_t embed_dim, size_t seq_len,
                                     const std::string &name = "") {
    auto layer = std::make_unique<PositionalEmbeddingLayer<T>>(
        embed_dim, seq_len,
        name.empty() ? "pos_embedding_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &slice(size_t axis, size_t start, size_t length, const std::string &name = "") {
    auto layer = std::make_unique<SliceLayer<T>>(
        axis, start, length, name.empty() ? "slice_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &full_attention(size_t embed_dim, size_t num_heads, const std::string &name = "") {
    auto layer = std::make_unique<FullAttentionBlock<T>>(
        embed_dim, num_heads,
        name.empty() ? "full_attention_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &flash_attention(size_t embed_dim, size_t num_heads, const std::string &name = "") {
    auto layer = std::make_unique<FlashAttentionBlock<T>>(
        embed_dim, num_heads,
        name.empty() ? "flash_attention_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &embedding(size_t vocab_size, size_t embed_dim, const std::string &name = "",
                          size_t padding_idx = static_cast<size_t>(-1)) {
    auto layer = std::make_unique<EmbeddingLayer<T>>(
        vocab_size, embed_dim, name.empty() ? "embedding_" + std::to_string(layers_.size()) : name,
        padding_idx);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &causal_attention(size_t embed_dim, size_t num_heads, const std::string &name = "") {
    auto layer = std::make_unique<CausalAttentionBlock<T>>(
        embed_dim, num_heads,
        name.empty() ? "causal_attention_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &transpose(const std::string &name = "") {
    auto layer = std::make_unique<TransposeLayer<T>>(
        name.empty() ? "transpose_" + std::to_string(layers_.size()) : name);
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
    return std::move(layers_);
  }

  const std::vector<size_t> &get_input_shape() const { return input_shape_; }
  bool is_input_shape_set() const { return input_shape_set_; }
};

template <typename T>
std::unordered_map<std::string, std::function<std::unique_ptr<Layer<T>>(const LayerConfig &)>>
    LayerFactory<T>::creators_;

} // namespace tnn