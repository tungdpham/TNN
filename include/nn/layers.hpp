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
#include "type/type.hpp"

namespace tnn {
inline std::unique_ptr<ActivationFunction> create_activation(const std::string &name) {
  ActivationFactory::register_defaults();
  return ActivationFactory::create(name);
}

class DenseLayer;
class ActivationLayer;
class LegacyConv2DLayer;
class MaxPool2DLayer;
class AvgPool2DLayer;
class DropoutLayer;
class FlattenLayer;
class BatchNormLayer;
class GroupNormLayer;
class LayerNormLayer;
class ClassTokenLayer;
class PositionalEmbeddingLayer;
class EmbeddingLayer;
class AttentionBlock;
class FusedAttentionBlock;
class ResidualBlock;
class SliceLayer;
class TransposeLayer;

} // namespace tnn

// Wrapper to include all layer implementations
#include "blocks_impl/attention_block.hpp"
#include "blocks_impl/residual_block.hpp"
#include "layers_impl/activation_layer.hpp"
#include "layers_impl/avgpool2d_layer.hpp"
#include "layers_impl/base_layer.hpp"
#include "layers_impl/batchnorm_layer.hpp"
#include "layers_impl/class_token_layer.hpp"
#include "layers_impl/conv2d_layer.hpp"
#include "layers_impl/dense_layer.hpp"
#include "layers_impl/dropout_layer.hpp"
#include "layers_impl/embedding_layer.hpp"
#include "layers_impl/flatten_layer.hpp"
#include "layers_impl/groupnorm_layer.hpp"
#include "layers_impl/layer_norm_layer.hpp"
#include "layers_impl/legacy_avgpool2d_layer.hpp"
#include "layers_impl/legacy_batchnorm_layer.hpp"
#include "layers_impl/legacy_conv2d_layer.hpp"
#include "layers_impl/legacy_maxpool2d_layer.hpp"
#include "layers_impl/maxpool2d_layer.hpp"
#include "layers_impl/positional_embedding_layer.hpp"
#include "layers_impl/slice_layer.hpp"
#include "layers_impl/transpose_layer.hpp"

namespace tnn {
class LayerFactory {
private:
  static std::unordered_map<std::string, std::function<std::unique_ptr<Layer>(const LayerConfig &)>>
      creators_;

public:
  static void register_layer(const std::string &type,
                             std::function<std::unique_ptr<Layer>(const LayerConfig &)> creator) {
    creators_[type] = creator;
  }

  static std::unique_ptr<Layer> create(const std::string &type, const LayerConfig &config) {
    auto it = creators_.find(type);
    if (it != creators_.end()) {
      return it->second(config);
    }
    throw std::invalid_argument("Unknown layer type: " + type);
  }

  static std::unique_ptr<Layer> create(const LayerConfig &config) {
    return create(config.get<std::string>("type"), config);
  }

  static void register_defaults() {
    register_layer("dense", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t input_features = config.get<size_t>("input_features");
      size_t output_features = config.get<size_t>("output_features");
      bool use_bias = config.get<bool>("use_bias", true);

      return std::make_unique<DenseLayer>(input_features, output_features, use_bias, config.name);
    });

    register_layer("activation", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      std::string activation_name = config.get<std::string>("activation");
      auto factory = ActivationFactory();
      factory.register_defaults();
      auto activation = factory.create(activation_name);
      if (!activation) {
        throw std::invalid_argument("Failed to create activation: " + activation_name);
      }
      return std::make_unique<ActivationLayer>(std::move(activation), config.name);
    });

    register_layer("conv2d", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t in_channels = config.get<size_t>("in_channels");
      size_t out_channels = config.get<size_t>("out_channels");
      size_t kernel_h = config.get<size_t>("kernel_h");
      size_t kernel_w = config.get<size_t>("kernel_w");
      size_t stride_h = config.get<size_t>("stride_h", 1);
      size_t stride_w = config.get<size_t>("stride_w", 1);
      size_t pad_h = config.get<size_t>("pad_h", 0);
      size_t pad_w = config.get<size_t>("pad_w", 0);
      bool use_bias = config.get<bool>("use_bias", true);

      return std::make_unique<Conv2DLayer>(in_channels, out_channels, kernel_h, kernel_w, stride_h,
                                           stride_w, pad_h, pad_w, use_bias, config.name);
    });

    register_layer("maxpool2d", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t pool_h = config.get<size_t>("pool_h");
      size_t pool_w = config.get<size_t>("pool_w");
      size_t stride_h = config.get<size_t>("stride_h", 0);
      size_t stride_w = config.get<size_t>("stride_w", 0);
      size_t pad_h = config.get<size_t>("pad_h", 0);
      size_t pad_w = config.get<size_t>("pad_w", 0);

      return std::make_unique<MaxPool2DLayer>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                              config.name);
    });

    register_layer("avgpool2d", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t pool_h = config.get<size_t>("pool_h");
      size_t pool_w = config.get<size_t>("pool_w");
      size_t stride_h = config.get<size_t>("stride_h", 0);
      size_t stride_w = config.get<size_t>("stride_w", 0);
      size_t pad_h = config.get<size_t>("pad_h", 0);
      size_t pad_w = config.get<size_t>("pad_w", 0);

      return std::make_unique<AvgPool2DLayer>(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
                                              config.name);
    });

    register_layer("batchnorm", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t num_features = config.get<size_t>("num_features");
      float epsilon = config.get<float>("epsilon", 1e-5f);
      float momentum = config.get<float>("momentum", 0.1f);
      bool affine = config.get<bool>("affine", true);
      return std::make_unique<BatchNormLayer>(num_features, epsilon, momentum, affine, config.name);
    });

    register_layer("legacy_conv2d", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t in_channels = config.get<size_t>("in_channels");
      size_t out_channels = config.get<size_t>("out_channels");
      size_t kernel_h = config.get<size_t>("kernel_h");
      size_t kernel_w = config.get<size_t>("kernel_w");
      size_t stride_h = config.get<size_t>("stride_h", 1);
      size_t stride_w = config.get<size_t>("stride_w", 1);
      size_t pad_h = config.get<size_t>("pad_h", 0);
      size_t pad_w = config.get<size_t>("pad_w", 0);
      bool use_bias = config.get<bool>("use_bias", true);

      return std::make_unique<LegacyConv2DLayer>(in_channels, out_channels, kernel_h, kernel_w,
                                                 stride_h, stride_w, pad_h, pad_w, use_bias,
                                                 config.name);
    });

    register_layer("legacy_maxpool2d", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t pool_h = config.get<size_t>("pool_h");
      size_t pool_w = config.get<size_t>("pool_w");
      size_t stride_h = config.get<size_t>("stride_h", 0);
      size_t stride_w = config.get<size_t>("stride_w", 0);
      size_t pad_h = config.get<size_t>("pad_h", 0);
      size_t pad_w = config.get<size_t>("pad_w", 0);

      return std::make_unique<LegacyMaxPool2DLayer>(pool_h, pool_w, stride_h, stride_w, pad_h,
                                                    pad_w, config.name);
    });

    register_layer("legacy_avgpool2d", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t pool_h = config.get<size_t>("pool_h");
      size_t pool_w = config.get<size_t>("pool_w");
      size_t stride_h = config.get<size_t>("stride_h", 0);
      size_t stride_w = config.get<size_t>("stride_w", 0);
      size_t pad_h = config.get<size_t>("pad_h", 0);
      size_t pad_w = config.get<size_t>("pad_w", 0);

      return std::make_unique<LegacyAvgPool2DLayer>(pool_h, pool_w, stride_h, stride_w, pad_h,
                                                    pad_w, config.name);
    });

    register_layer("legacy_batchnorm", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t num_features = config.get<size_t>("num_features");
      float epsilon = config.get<float>("epsilon", 1e-5f);
      float momentum = config.get<float>("momentum", 0.1f);
      bool affine = config.get<bool>("affine", true);
      return std::make_unique<LegacyBatchNormLayer>(num_features, epsilon, momentum, affine,
                                                    config.name);
    });

    register_layer("dropout", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      float dropout_rate = config.get<float>("dropout_rate");
      return std::make_unique<DropoutLayer>(dropout_rate, config.name);
    });

    register_layer("groupnorm", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t num_groups = config.get<size_t>("num_groups");
      size_t num_channels = config.get<size_t>("num_channels");
      float epsilon = config.get<float>("epsilon", 1e-5f);
      bool affine = config.get<bool>("affine", true);
      return std::make_unique<GroupNormLayer>(num_groups, num_channels, epsilon, affine,
                                              config.name);
    });

    register_layer("layer_norm", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t normalized_shape = config.get<size_t>("normalized_shape");
      float epsilon = config.get<float>("epsilon", 1e-5f);
      bool affine = config.get<bool>("affine", true);
      return std::make_unique<LayerNormLayer>(normalized_shape, epsilon, affine, config.name);
    });

    register_layer("flatten", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      return FlattenLayer::create_from_config(config);
    });

    register_layer("class_token", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t embed_dim = config.get<size_t>("embed_dim");
      return std::make_unique<ClassTokenLayer>(embed_dim, config.name);
    });

    register_layer("pos_embedding", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t embed_dim = config.get<size_t>("embed_dim");
      size_t seq_len = config.get<size_t>("seq_len");
      return std::make_unique<PositionalEmbeddingLayer>(embed_dim, seq_len, config.name);
    });

    register_layer("slice", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      return SliceLayer::create_from_config(config);
    });

    register_layer("embedding", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t vocab_size = config.get<size_t>("vocab_size");
      size_t embed_dim = config.get<size_t>("embed_dim");
      size_t padding_idx = config.get<size_t>("padding_idx", static_cast<size_t>(-1));
      return std::make_unique<EmbeddingLayer>(vocab_size, embed_dim, config.name, padding_idx);
    });

    register_layer("residual_block", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      std::string activation = config.get<std::string>("activation", "relu");
      bool has_projection = config.get<bool>("has_projection", false);
      std::string main_path_str = config.get<std::string>("main_path", std::string("[]"));
      std::string shortcut_path_str = config.get<std::string>("shortcut_path", std::string("[]"));

      nlohmann::json main_json = nlohmann::json::parse(main_path_str);
      nlohmann::json shortcut_json = nlohmann::json::parse(shortcut_path_str);

      LayerFactory sub_factory;
      sub_factory.register_defaults();

      std::vector<std::unique_ptr<Layer>> main_layers;
      for (const auto &sub : main_json) {
        LayerConfig sub_cfg = LayerConfig::from_json(sub);
        std::string sub_type = sub.value("type", "");
        main_layers.push_back(sub_factory.create(sub_type, sub_cfg));
      }

      std::vector<std::unique_ptr<Layer>> shortcut_layers;
      if (has_projection) {
        for (const auto &sub : shortcut_json) {
          LayerConfig sub_cfg = LayerConfig::from_json(sub);
          std::string sub_type = sub.value("type", "");
          shortcut_layers.push_back(sub_factory.create(sub_type, sub_cfg));
        }
      }

      return std::make_unique<ResidualBlock>(std::move(main_layers), std::move(shortcut_layers),
                                             activation, config.name);
    });

    register_layer("attention_block", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      size_t embed_dim = config.get<size_t>("embed_dim");
      size_t num_heads = config.get<size_t>("num_heads");
      bool is_causal = config.get<bool>("is_causal", false);
      return std::make_unique<AttentionBlock>(embed_dim, num_heads, is_causal, config.name);
    });

    register_layer("transpose", [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      return std::make_unique<TransposeLayer>(config.name);
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

class LayerBuilder {
private:
  std::vector<std::unique_ptr<Layer>> layers_;
  std::vector<size_t> input_shape_;
  DType_t io_dtype_ = DType_t::FP32;
  bool input_shape_set_ = false;

public:
  explicit LayerBuilder(const std::string &name = "Block") {}

  std::vector<size_t> get_current_shape() const {
    if (!input_shape_set_) {
      throw std::runtime_error("Input shape must be set before adding layers. "
                               "Use .input() method first.");
    }
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

  LayerBuilder &dtype(DType_t dtype) {
    io_dtype_ = dtype;
    return *this;
  }

  LayerBuilder &dense(size_t output_features, bool use_bias = true, const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();
    size_t input_features = current_shape.back();

    auto layer = std::make_unique<DenseLayer>(
        input_features, output_features, use_bias,
        name.empty() ? "dense_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &conv2d(size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h = 1,
                       size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                       bool use_bias = true, const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() != 4) {
      throw std::runtime_error("Conv2D requires 4D input (batch, channels, "
                               "height, width). Current shape has " +
                               std::to_string(current_shape.size()) + " dimensions.");
    }

    size_t in_channels = current_shape.back();

    auto layer = std::make_unique<Conv2DLayer>(
        in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, use_bias,
        name.empty() ? "conv2d_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &batchnorm(float epsilon = 1e-5f, float momentum = 0.1f, bool affine = true,
                          const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 2) {
      throw std::runtime_error("BatchNorm requires at least 2D input (batch, features)");
    }

    size_t num_features = current_shape.back();

    auto layer = std::make_unique<BatchNormLayer>(
        num_features, epsilon, momentum, affine,
        name.empty() ? "batchnorm_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &maxpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 0, size_t stride_w = 0,
                          size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "") {
    auto layer = std::make_unique<MaxPool2DLayer>(
        pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
        name.empty() ? "maxpool2d_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &avgpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 1, size_t stride_w = 1,
                          size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "") {
    auto layer = std::make_unique<AvgPool2DLayer>(
        pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
        name.empty() ? "avgpool2d_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &legacy_conv2d(size_t out_channels, size_t kernel_h, size_t kernel_w,
                              size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0,
                              size_t pad_w = 0, bool use_bias = true,
                              const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 4) {
      throw std::runtime_error("Conv2D requires 4D input (batch, channels, "
                               "height, width). Current shape has " +
                               std::to_string(current_shape.size()) + " dimensions.");
    }

    size_t in_channels = current_shape[1];

    auto layer = std::make_unique<LegacyConv2DLayer>(
        in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, use_bias,
        name.empty() ? "conv2d_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &legacy_batchnorm(float epsilon = 1e-5f, float momentum = 0.1f, bool affine = true,
                                 const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 2) {
      throw std::runtime_error("BatchNorm requires at least 2D input (batch, features)");
    }

    size_t num_features = current_shape[1];

    auto layer = std::make_unique<BatchNormLayer>(
        num_features, epsilon, momentum, affine,
        name.empty() ? "batchnorm_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &legacy_maxpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 0,
                                 size_t stride_w = 0, size_t pad_h = 0, size_t pad_w = 0,
                                 const std::string &name = "") {
    auto layer = std::make_unique<LegacyMaxPool2DLayer>(
        pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
        name.empty() ? "maxpool2d_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &legacy_avgpool2d(size_t pool_h, size_t pool_w, size_t stride_h = 1,
                                 size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                                 const std::string &name = "") {
    auto layer = std::make_unique<LegacyAvgPool2DLayer>(
        pool_h, pool_w, stride_h, stride_w, pad_h, pad_w,
        name.empty() ? "avgpool2d_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &groupnorm(float num_groups, float epsilon = 1e-5f, bool affine = true,
                          const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 2) {
      throw std::runtime_error("GroupNorm requires at least 2D input (batch, features)");
    }

    size_t num_channels = current_shape.back();

    auto layer = std::make_unique<GroupNormLayer>(
        num_groups, num_channels, epsilon, affine,
        name.empty() ? "groupnorm_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &layernorm(float epsilon = 1e-5f, bool affine = true, const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 2) {
      throw std::runtime_error("LayerNorm requires at least 2D input (batch, features)");
    }

    size_t num_features = current_shape.back();

    auto layer = std::make_unique<LayerNormLayer>(
        num_features, epsilon, affine,
        name.empty() ? "layernorm_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &activation(const std::string &activation_name, const std::string &name = "") {
    auto factory = ActivationFactory();
    factory.register_defaults();
    auto act = factory.create(activation_name);
    auto layer = std::make_unique<ActivationLayer>(
        std::move(act), name.empty() ? "activation_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::unique_ptr<Layer>(std::move(layer)));
    return *this;
  }

  LayerBuilder &dropout(float dropout_rate, const std::string &name = "") {
    auto layer = std::make_unique<DropoutLayer>(
        dropout_rate, name.empty() ? "dropout_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &flatten(int start_dim = 1, const std::string &name = "") {
    auto layer = std::make_unique<FlattenLayer>(
        start_dim, name.empty() ? "flatten_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &class_token(size_t embed_dim, const std::string &name = "") {
    auto layer = std::make_unique<ClassTokenLayer>(
        embed_dim, name.empty() ? "class_token_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &positional_embedding(size_t embed_dim, size_t seq_len,
                                     const std::string &name = "") {
    auto layer = std::make_unique<PositionalEmbeddingLayer>(
        embed_dim, seq_len,
        name.empty() ? "pos_embedding_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &slice(size_t axis, size_t start, size_t length, const std::string &name = "") {
    auto layer = std::make_unique<SliceLayer>(
        axis, start, length, name.empty() ? "slice_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &attention(size_t embed_dim, size_t num_heads, bool is_causal = false,
                          const std::string &name = "") {
    auto layer = std::make_unique<AttentionBlock>(
        embed_dim, num_heads, is_causal,
        name.empty() ? "attention_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &embedding(size_t vocab_size, size_t embed_dim, const std::string &name = "",
                          size_t padding_idx = static_cast<size_t>(-1)) {
    auto layer = std::make_unique<EmbeddingLayer>(
        vocab_size, embed_dim, name.empty() ? "embedding_" + std::to_string(layers_.size()) : name,
        padding_idx);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &transpose(const std::string &name = "") {
    auto layer = std::make_unique<TransposeLayer>(
        name.empty() ? "transpose_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &residual_block(std::vector<std::unique_ptr<Layer>> main_path,
                               std::vector<std::unique_ptr<Layer>> shortcut_path,
                               const std::string &activation, const std::string &name = "") {
    auto layer = std::make_unique<ResidualBlock>(
        std::move(main_path), std::move(shortcut_path), activation,
        name.empty() ? "residual_block_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  /**
   * Two 3x3 convolutions with batch normalization
   */
  LayerBuilder &basic_residual_block(size_t in_channels, size_t out_channels, size_t stride = 1,
                                     const std::string &name = "basic_residual_block") {
    std::vector<size_t> current_shape = get_current_shape();
    std::vector<size_t> input_shape =
        std::vector<size_t>{current_shape[1], current_shape[2], current_shape[3]};
    auto main_path = LayerBuilder()
                         .input(input_shape)
                         .conv2d(out_channels, 3, 3, stride, stride, 1, 1, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                         .activation("relu")
                         .conv2d(out_channels, 3, 3, 1, 1, 1, 1, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                         .build();

    std::vector<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder()
                     .input(input_shape)
                     .conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
                     .batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                     .build();
    }

    auto res_block = std::make_unique<ResidualBlock>(
        std::move(main_path), std::move(shortcut), "relu",
        name.empty() ? "basic_residual_block_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(res_block));
    return *this;
  }

  /**
   * Two 3x3 convolutions with batch normalization and optional dropout
   * Uses pre-activation (BN-ReLU-Conv) ordering as in the original WRN paper
   */
  LayerBuilder &wide_residual_block(size_t in_channels, size_t out_channels, size_t stride = 1,
                                    float dropout_rate = 0.0f,
                                    const std::string &name = "wide_residual_block") {
    auto current_shape = get_current_shape();
    auto input_shape = std::vector<size_t>{current_shape[1], current_shape[2], current_shape[3]};

    // Build main path with pre-activation (BN-ReLU-Conv) ordering
    LayerBuilder main_builder;
    main_builder.input(input_shape)
        .batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
        .activation("relu")
        .conv2d(out_channels, 3, 3, stride, stride, 1, 1, true)
        .batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn2")
        .activation("relu");

    if (dropout_rate > 0.0f) {
      main_builder.dropout(dropout_rate);
    }

    main_builder.conv2d(out_channels, 3, 3, 1, 1, 1, 1, true);

    auto main_path = main_builder.build();

    std::vector<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder()
                     .input(input_shape)
                     .conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
                     .build();
    }

    // Note: WRN uses identity activation after addition (no ReLU)
    auto res_block = std::make_unique<ResidualBlock>(
        std::move(main_path), std::move(shortcut), "linear",
        name.empty() ? "wide_residual_block_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(res_block));
    return *this;
  }

  /**
   * 1x1 conv, 3x3 conv, 1x1 conv, bn
   */
  LayerBuilder &bottleneck_residual_block(size_t in_channels, size_t mid_channels,
                                          size_t out_channels, size_t stride = 1,
                                          const std::string &name = "bottleneck_residual_block") {
    auto current_shape = get_current_shape();
    auto input_shape = std::vector<size_t>{current_shape[1], current_shape[2], current_shape[3]};
    auto main_path = LayerBuilder()
                         .input(input_shape)
                         .conv2d(mid_channels, 1, 1, 1, 1, 0, 0, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                         .activation("relu")
                         .conv2d(mid_channels, 3, 3, stride, stride, 1, 1, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
                         .activation("relu")
                         .conv2d(out_channels, 1, 1, 1, 1, 0, 0, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn2")
                         .build();

    std::vector<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder()
                     .input(input_shape)
                     .conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
                     .batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn3")
                     .build();
    }

    auto res_block =
        std::make_unique<ResidualBlock>(std::move(main_path), std::move(shortcut), "relu", name);
    layers_.push_back(std::move(res_block));
    return *this;
  }

  /**
   * @brief Helper function to create a GPT-style Transformer block
   * x = x + Dropout(CausalAttention(LayerNorm(x)))
   * x = x + Dropout(Projection(Activation(Expansion(LayerNorm(x)))))
   */
  LayerBuilder &gpt_block(size_t embed_dim, size_t num_heads, size_t ffn_dim,
                          float dropout_rate = 0.1f, bool is_causal = false,
                          const std::string &activation_fn = "gelu", const std::string &name = "") {
    std::string valid_name = name.empty() ? "gpt_block_" + std::to_string(layers_.size()) : name;
    std::vector<size_t> current_shape = get_current_shape();

    std::vector<size_t> batchless_shape(current_shape.begin() + 1, current_shape.end());

    // 1. Attention Sub-block (Residual)
    auto attn_main = LayerBuilder()
                         .input(batchless_shape)
                         .layernorm(dtype_eps(io_dtype_), true, "ln_1")
                         .attention(embed_dim, num_heads, is_causal, "attn")
                         .dropout(dropout_rate)
                         .build();

    auto attn_res =
        std::make_unique<ResidualBlock>(std::move(attn_main), std::vector<std::unique_ptr<Layer>>(),
                                        "linear", valid_name + "_attn");
    layers_.push_back(std::move(attn_res));

    // 2. Feed-Forward Sub-block (Residual)
    auto ffn_main = LayerBuilder()
                        .input(batchless_shape) // Input shape matches (residual preserves shape)
                        .layernorm(dtype_eps(io_dtype_), true, "ln_2")
                        .dense(ffn_dim, true, "mlp_fc1")
                        .activation(activation_fn)
                        .dense(embed_dim, true, "mlp_fc2")
                        .dropout(dropout_rate)
                        .build();

    auto ffn_res = std::make_unique<ResidualBlock>(
        std::move(ffn_main), std::vector<std::unique_ptr<Layer>>(), "linear", valid_name + "_ffn");
    layers_.push_back(std::move(ffn_res));

    return *this;
  }

  LayerBuilder &add_layer(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
    return *this;
  }

  std::vector<std::unique_ptr<Layer>> build() {
    if (!input_shape_set_) {
      throw std::runtime_error("Input shape must be set before building block. "
                               "Use .input() method.");
    }
    return std::move(layers_);
  }

  const std::vector<size_t> &get_input_shape() const { return input_shape_; }
  bool is_input_shape_set() const { return input_shape_set_; }
};

inline std::unordered_map<std::string, std::function<std::unique_ptr<Layer>(const LayerConfig &)>>
    LayerFactory::creators_;

} // namespace tnn