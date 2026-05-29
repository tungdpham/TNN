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
#include "nn/layer.hpp"

namespace tnn {
inline std::unique_ptr<ActivationFunction> create_activation(const std::string &name) {
  ActivationFactory::register_defaults();
  return ActivationFactory::create(name);
}

class LegacyDenseLayerImpl;
class LegacyConv2DLayerImpl;
class LegacyMaxPool2DLayerImpl;
class LegacyAvgPool2DLayerImpl;
class LegacyBatchNormLayerImpl;

class IdentityLayerImpl;
class DenseLayerImpl;
class ActivationLayerImpl;
class Conv2DLayerImpl;
class MaxPool2DLayerImpl;
class AvgPool2DLayerImpl;
class BatchNormLayerImpl;
class DropoutLayerImpl;
class FlattenLayerImpl;
class GroupNormLayerImpl;
class LayerNormLayerImpl;
class ClassTokenLayerImpl;
class PositionalEmbeddingLayerImpl;
class EmbeddingLayerImpl;
class AttentionBlock;
class FlashAttentionBlock;
class ResidualBlock;
class SliceLayerImpl;
class TransposeLayerImpl;
class Sequential;
class MSequential;

class AddLayerImpl;
class SubLayerImpl;
class MulLayerImpl;
class DivLayerImpl;

}  // namespace tnn

// Wrapper to include all layer implementations
#include "blocks_impl/attention_block.hpp"
#include "blocks_impl/flash_attention_block.hpp"
#include "blocks_impl/residual_block.hpp"
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
#include "nn/blocks_impl/flash_attention_block.hpp"
#include "nn/blocks_impl/msequential.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/layers_impl/add_layer.hpp"
#include "nn/layers_impl/div_layer.hpp"
#include "nn/layers_impl/identity_layer.hpp"
#include "nn/layers_impl/legacy_conv2d_layer.hpp"
#include "nn/layers_impl/legacy_dense_layer.hpp"
#include "nn/layers_impl/mul_layer.hpp"
#include "nn/layers_impl/sub_layer.hpp"

namespace tnn {

// Concept to ensure LayerType has TYPE_NAME and create_from_config
template <typename T>
concept HasLayerTypeName = requires {
  { T::TYPE_NAME } -> std::convertible_to<const char *>;
  {
    T::create_from_config(std::declval<const LayerConfig &>())
  } -> std::convertible_to<std::unique_ptr<LayerImpl>>;
};

class LayerFactory {
private:
  static std::unordered_map<std::string,
                            std::function<std::unique_ptr<LayerImpl>(const LayerConfig &)>>
      creators_;

public:
  static void register_layer(
      const std::string &type,
      std::function<std::unique_ptr<LayerImpl>(const LayerConfig &)> creator) {
    creators_[type] = creator;
  }

  template <HasLayerTypeName LayerType>
  static void register_layer_type() {
    register_layer(LayerType::TYPE_NAME,
                   [](const LayerConfig &config) -> std::unique_ptr<LayerImpl> {
                     return LayerType::create_from_config(config);
                   });
  }

  static std::unique_ptr<LayerImpl> create(const std::string &type, const LayerConfig &config) {
    auto it = creators_.find(type);
    if (it != creators_.end()) {
      return it->second(config);
    }
    throw std::invalid_argument("Unknown layer type: " + type);
  }

  static std::unique_ptr<LayerImpl> create(const LayerConfig &config) {
    return create(config.type, config);
  }

  static void register_defaults() {
    register_layer_type<IdentityLayerImpl>();
    register_layer_type<DenseLayerImpl>();
    register_layer_type<ActivationLayerImpl>();
    register_layer_type<Conv2DLayerImpl>();
    register_layer_type<MaxPool2DLayerImpl>();
    register_layer_type<AvgPool2DLayerImpl>();
    register_layer_type<BatchNormLayerImpl>();
    register_layer_type<LegacyConv2DLayerImpl>();
    register_layer_type<LegacyMaxPool2DLayerImpl>();
    register_layer_type<LegacyAvgPool2DLayerImpl>();
    register_layer_type<LegacyBatchNormLayerImpl>();
    register_layer_type<DropoutLayerImpl>();
    register_layer_type<GroupNormLayerImpl>();
    register_layer_type<LayerNormLayerImpl>();
    register_layer_type<LegacyDenseLayerImpl>();
    register_layer_type<FlattenLayerImpl>();
    register_layer_type<ClassTokenLayerImpl>();
    register_layer_type<PositionalEmbeddingLayerImpl>();
    register_layer_type<SliceLayerImpl>();
    register_layer_type<EmbeddingLayerImpl>();
    register_layer_type<ResidualBlock>();
    register_layer_type<AttentionBlock>();
    register_layer_type<FlashAttentionBlock>();
    register_layer_type<TransposeLayerImpl>();
    register_layer_type<ResidualBlock>();
    register_layer_type<AttentionBlock>();
    register_layer_type<FlashAttentionBlock>();
    register_layer_type<Sequential>();
    register_layer_type<MSequential>();
    register_layer_type<AddLayerImpl>();
    register_layer_type<SubLayerImpl>();
    register_layer_type<MulLayerImpl>();
    register_layer_type<DivLayerImpl>();
  }

  static Vec<std::string> available_types() {
    Vec<std::string> types;
    for (const auto &pair : creators_) {
      types.push_back(pair.first);
    }
    return types;
  }
};

template <typename LayerType>
std::unique_ptr<LayerType> load_config(std::ifstream &file) {
  size_t j_size;
  file.read(reinterpret_cast<char *>(&j_size), sizeof(size_t));
  std::string j_str(j_size, '\0');
  file.read(&j_str[0], j_size);
  nlohmann::json j = nlohmann::json::parse(j_str);
  LayerConfig config = LayerConfig::from_json(j);
  LayerFactory::register_defaults();
  std::unique_ptr<LayerImpl> base_layer = LayerFactory::create(config);
  LayerType *raw_ptr = dynamic_cast<LayerType *>(base_layer.release());
  if (!raw_ptr) {
    throw std::runtime_error("Failed to cast layer to requested type");
  }
  std::unique_ptr<LayerType> layer(raw_ptr);
  return layer;
}

inline void load_params(std::ifstream &file, LayerImpl &layer) {
  Vec<Tensor> params = layer.parameters();
  for (auto &param : params) {
    load_into(file, param);
  }
}

inline std::unordered_map<std::string,
                          std::function<std::unique_ptr<LayerImpl>(const LayerConfig &)>>
    LayerFactory::creators_;

}  // namespace tnn

#include "nn/layer_builder.hpp"  // IWYU pragma: export
