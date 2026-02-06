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
#include "nn/blocks_impl/flash_attention_block.hpp"
#include "nn/layer.hpp"
#include "nn/layers_impl/legacy_dense_layer.hpp"
#include "type/type.hpp"

namespace tnn {
inline std::unique_ptr<ActivationFunction> create_activation(const std::string &name) {
  ActivationFactory::register_defaults();
  return ActivationFactory::create(name);
}

class LegacyDenseLayer;
class LegacyConv2DLayer;
class LegacyMaxPool2DLayer;
class LegacyAvgPool2DLayer;
class LegacyBatchNormLayer;

class DenseLayer;
class ActivationLayer;
class Conv2DLayer;
class MaxPool2DLayer;
class AvgPool2DLayer;
class BatchNormLayer;
class DropoutLayer;
class FlattenLayer;
class GroupNormLayer;
class LayerNormLayer;
class ClassTokenLayer;
class PositionalEmbeddingLayer;
class EmbeddingLayer;
class AttentionBlock;
class FlashAttentionBlock;
class ResidualBlock;
class SliceLayer;
class TransposeLayer;
class Sequential;

}  // namespace tnn

// Wrapper to include all layer implementations
#include "blocks_impl/attention_block.hpp"
#include "blocks_impl/flash_attention_block.hpp"
#include "blocks_impl/residual_block.hpp"
#include "layer.hpp"
#include "layers_impl/activation_layer.hpp"
#include "layers_impl/avgpool2d_layer.hpp"
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
#include "layers_impl/legacy_dense_layer.hpp"
#include "layers_impl/legacy_maxpool2d_layer.hpp"
#include "layers_impl/maxpool2d_layer.hpp"
#include "layers_impl/positional_embedding_layer.hpp"
#include "layers_impl/slice_layer.hpp"
#include "layers_impl/transpose_layer.hpp"
#include "nn/sequential.hpp"

namespace tnn {

// Concept to ensure LayerType has TYPE_NAME and create_from_config
template <typename T>
concept HasLayerTypeName = requires {
  { T::TYPE_NAME } -> std::convertible_to<const char *>;
  {
    T::create_from_config(std::declval<const LayerConfig &>())
  } -> std::convertible_to<std::unique_ptr<Layer>>;
};

class LayerFactory {
private:
  static std::unordered_map<std::string, std::function<std::unique_ptr<Layer>(const LayerConfig &)>>
      creators_;

public:
  static void register_layer(const std::string &type,
                             std::function<std::unique_ptr<Layer>(const LayerConfig &)> creator) {
    creators_[type] = creator;
  }

  template <HasLayerTypeName LayerType>
  static void register_layer_type() {
    register_layer(LayerType::TYPE_NAME, [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      return LayerType::create_from_config(config);
    });
  }

  static std::unique_ptr<Layer> create(const std::string &type, const LayerConfig &config) {
    auto it = creators_.find(type);
    if (it != creators_.end()) {
      return it->second(config);
    }
    throw std::invalid_argument("Unknown layer type: " + type);
  }

  static std::unique_ptr<Layer> create(const LayerConfig &config) {
    return create(config.type, config);
  }

  static void register_defaults() {
    register_layer_type<DenseLayer>();
    register_layer_type<ActivationLayer>();
    register_layer_type<Conv2DLayer>();
    register_layer_type<MaxPool2DLayer>();
    register_layer_type<AvgPool2DLayer>();
    register_layer_type<BatchNormLayer>();
    register_layer_type<LegacyConv2DLayer>();
    register_layer_type<LegacyMaxPool2DLayer>();
    register_layer_type<LegacyAvgPool2DLayer>();
    register_layer_type<LegacyBatchNormLayer>();
    register_layer_type<DropoutLayer>();
    register_layer_type<GroupNormLayer>();
    register_layer_type<LayerNormLayer>();
    register_layer_type<LegacyDenseLayer>();
    register_layer_type<FlattenLayer>();
    register_layer_type<ClassTokenLayer>();
    register_layer_type<PositionalEmbeddingLayer>();
    register_layer_type<SliceLayer>();
    register_layer_type<EmbeddingLayer>();
    register_layer_type<ResidualBlock>();
    register_layer_type<AttentionBlock>();
    register_layer_type<FlashAttentionBlock>();
    register_layer_type<TransposeLayer>();
    register_layer_type<ResidualBlock>();
    register_layer_type<AttentionBlock>();
    register_layer_type<FlashAttentionBlock>();
    register_layer_type<Sequential>();
  }

  static std::vector<std::string> available_types() {
    std::vector<std::string> types;
    for (const auto &pair : creators_) {
      types.push_back(pair.first);
    }
    return types;
  }
};

template <typename LayerType>
static std::unique_ptr<LayerType> load_state(std::ifstream &file, const Device &device) {
  size_t j_size;
  file.read(reinterpret_cast<char *>(&j_size), sizeof(size_t));
  std::string j_str(j_size, '\0');
  file.read(&j_str[0], j_size);
  nlohmann::json j = nlohmann::json::parse(j_str);
  LayerConfig config = LayerConfig::from_json(j);
  LayerFactory::register_defaults();
  std::unique_ptr<Layer> base_layer = LayerFactory::create(config);
  LayerType *raw_ptr = dynamic_cast<LayerType *>(base_layer.release());
  if (!raw_ptr) {
    throw std::runtime_error("Failed to cast layer to requested type");
  }
  std::unique_ptr<LayerType> layer(raw_ptr);
  layer->set_device(device);
  layer->init();
  std::vector<Tensor> params = layer->parameters();
  if (!params.empty()) {
    layer->set_param_dtype(params[0]->data_type());
  }
  for (auto &param : params) {
    load_into(file, param);
  }
  return layer;
}

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
      throw std::runtime_error(
          "Input shape must be set before adding layers. "
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
      throw std::runtime_error(
          "Conv2D requires 4D input (batch, channels, "
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
                          SBool use_relu = SBool::FALSE, const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 2) {
      throw std::runtime_error("BatchNorm requires at least 2D input (batch, features)");
    }

    size_t num_features = current_shape.back();

    auto layer = std::make_unique<BatchNormLayer>(
        num_features, epsilon, momentum, affine, use_relu == SBool::TRUE,
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

  LayerBuilder &legacy_dense(size_t output_features, bool use_bias = true,
                             const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();
    size_t input_features = current_shape.back();

    auto layer = std::make_unique<LegacyDenseLayer>(
        input_features, output_features, use_bias,
        name.empty() ? "dense_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &legacy_conv2d(size_t out_channels, size_t kernel_h, size_t kernel_w,
                              size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0,
                              size_t pad_w = 0, bool use_bias = true,
                              const std::string &name = "") {
    std::vector<size_t> current_shape = get_current_shape();

    if (current_shape.size() < 4) {
      throw std::runtime_error(
          "Conv2D requires 4D input (batch, channels, "
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

    auto layer = std::make_unique<LegacyBatchNormLayer>(
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

  LayerBuilder &flatten(int start_dim = 1, int end_dim = -1, const std::string &name = "") {
    auto layer = std::make_unique<FlattenLayer>(
        start_dim, end_dim, name.empty() ? "flatten_" + std::to_string(layers_.size()) : name);
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

  LayerBuilder &flash_attention(size_t embed_dim, size_t num_heads, bool is_causal = false,
                                const std::string &name = "") {
    auto layer = std::make_unique<FlashAttentionBlock>(
        embed_dim, num_heads, is_causal,
        name.empty() ? "flash_attention_" + std::to_string(layers_.size()) : name);
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
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn0")
                         .conv2d(out_channels, 3, 3, 1, 1, 1, 1, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::FALSE, "bn0")
                         .build();

    std::vector<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder()
                     .input(input_shape)
                     .conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
                     .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::FALSE, "bn0")
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
        .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn1")
        .conv2d(out_channels, 3, 3, stride, stride, 1, 1, true)
        .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn2");

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
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn0")
                         .conv2d(mid_channels, 3, 3, stride, stride, 1, 1, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn1")
                         .conv2d(out_channels, 1, 1, 1, 1, 0, 0, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn2")
                         .build();

    std::vector<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder()
                     .input(input_shape)
                     .conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
                     .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::FALSE, "bn3")
                     .build();
    }

    auto res_block =
        std::make_unique<ResidualBlock>(std::move(main_path), std::move(shortcut), "relu", name);
    layers_.push_back(std::move(res_block));
    return *this;
  }

  /**
   * Two 3x3 convolutions with batch normalization
   */
  LayerBuilder &legacy_basic_residual_block(size_t in_channels, size_t out_channels,
                                            size_t stride = 1,
                                            const std::string &name = "basic_residual_block") {
    std::vector<size_t> current_shape = get_current_shape();
    std::vector<size_t> input_shape =
        std::vector<size_t>{current_shape[1], current_shape[2], current_shape[3]};
    auto main_path = LayerBuilder()
                         .input(input_shape)
                         .legacy_conv2d(out_channels, 3, 3, stride, stride, 1, 1, false)
                         .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                         .legacy_conv2d(out_channels, 3, 3, 1, 1, 1, 1, false)
                         .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                         .build();

    std::vector<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder()
                     .input(input_shape)
                     .legacy_conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
                     .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
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
  LayerBuilder &legacy_wide_residual_block(size_t in_channels, size_t out_channels,
                                           size_t stride = 1, float dropout_rate = 0.0f,
                                           const std::string &name = "wide_residual_block") {
    auto current_shape = get_current_shape();
    auto input_shape = std::vector<size_t>{current_shape[1], current_shape[2], current_shape[3]};

    // Build main path with pre-activation (BN-ReLU-Conv) ordering
    LayerBuilder main_builder;
    main_builder.input(input_shape)
        .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
        .legacy_conv2d(out_channels, 3, 3, stride, stride, 1, 1, true)
        .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn2");

    if (dropout_rate > 0.0f) {
      main_builder.dropout(dropout_rate);
    }

    main_builder.legacy_conv2d(out_channels, 3, 3, 1, 1, 1, 1, true);

    auto main_path = main_builder.build();

    std::vector<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder()
                     .input(input_shape)
                     .legacy_conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
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
  LayerBuilder &legacy_bottleneck_residual_block(
      size_t in_channels, size_t mid_channels, size_t out_channels, size_t stride = 1,
      const std::string &name = "bottleneck_residual_block") {
    auto current_shape = get_current_shape();
    auto input_shape = std::vector<size_t>{current_shape[1], current_shape[2], current_shape[3]};
    auto main_path = LayerBuilder()
                         .input(input_shape)
                         .legacy_conv2d(mid_channels, 1, 1, 1, 1, 0, 0, false)
                         .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                         .legacy_conv2d(mid_channels, 3, 3, stride, stride, 1, 1, false)
                         .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
                         .legacy_conv2d(out_channels, 1, 1, 1, 1, 0, 0, false)
                         .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn2")
                         .build();

    std::vector<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder()
                     .input(input_shape)
                     .legacy_conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
                     .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn3")
                     .build();
    }

    auto res_block =
        std::make_unique<ResidualBlock>(std::move(main_path), std::move(shortcut), "relu", name);
    layers_.push_back(std::move(res_block));
    return *this;
  }

  /**
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
                        .input(batchless_shape)  // Input shape matches (residual preserves shape)
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

  /**
   * x = x + Dropout(CausalAttention(LayerNorm(x)))
   * x = x + Dropout(Projection(Activation(Expansion(LayerNorm(x)))))
   */
  LayerBuilder &flash_gpt_block(size_t embed_dim, size_t num_heads, size_t ffn_dim,
                                float dropout_rate = 0.1f, bool is_causal = false,
                                const std::string &activation_fn = "gelu",
                                const std::string &name = "") {
    std::string valid_name = name.empty() ? "gpt_block_" + std::to_string(layers_.size()) : name;
    std::vector<size_t> current_shape = get_current_shape();

    std::vector<size_t> batchless_shape(current_shape.begin() + 1, current_shape.end());

    // 1. Attention Sub-block (Residual)
    auto attn_main = LayerBuilder()
                         .input(batchless_shape)
                         .layernorm(dtype_eps(io_dtype_), true, "ln_1")
                         .flash_attention(embed_dim, num_heads, is_causal, "attn")
                         .dropout(dropout_rate)
                         .build();

    auto attn_res =
        std::make_unique<ResidualBlock>(std::move(attn_main), std::vector<std::unique_ptr<Layer>>(),
                                        "linear", valid_name + "_attn");
    layers_.push_back(std::move(attn_res));

    // 2. Feed-Forward Sub-block (Residual)
    auto ffn_main = LayerBuilder()
                        .input(batchless_shape)  // Input shape matches (residual preserves shape)
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
      throw std::runtime_error(
          "Input shape must be set before building block. "
          "Use .input() method.");
    }
    return std::move(layers_);
  }

  const std::vector<size_t> &get_input_shape() const { return input_shape_; }
  bool is_input_shape_set() const { return input_shape_set_; }
};

inline std::unordered_map<std::string, std::function<std::unique_ptr<Layer>(const LayerConfig &)>>
    LayerFactory::creators_;

}  // namespace tnn