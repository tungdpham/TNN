#pragma once

#include <memory>
#include <string>

#include "nn/layers.hpp"
#include "type/type.hpp"

namespace tnn {

class LayerBuilder {
private:
  Vec<std::unique_ptr<Layer>> layers_;
  Vec<Vec<size_t>> input_shapes_;
  DType_t io_dtype_ = DType_t::FP32;

public:
  explicit LayerBuilder(const Vec<Vec<size_t>> &batchless_shape) {
    for (const auto &shape : batchless_shape) {
      Vec<size_t> batch_shape = shape;
      batch_shape.insert(batch_shape.begin(), 1);  // Prepend batch dimension
      input_shapes_.push_back(batch_shape);
    }
  }

  Vec<Vec<size_t>> get_current_shape() const {
    Vec<Vec<size_t>> current_shapes = input_shapes_;
    for (const auto &layer : layers_) {
      current_shapes = layer->output_shapes(current_shapes);
    }
    return current_shapes;
  }

  Vec<Vec<size_t>> get_batchless_current_shape() const {
    Vec<Vec<size_t>> current_shapes = get_current_shape();
    for (auto &shape : current_shapes) {
      if (!shape.empty()) {
        shape.erase(shape.begin());  // Remove batch dimension
      }
    }
    return current_shapes;
  }

  LayerBuilder &dtype(DType_t dtype) {
    io_dtype_ = dtype;
    return *this;
  }

  LayerBuilder &dense(size_t output_features, bool use_bias = true, const std::string &name = "") {
    Vec<Vec<size_t>> current_shapes = get_current_shape();
    size_t input_features = current_shapes.back().back();

    auto layer = std::make_unique<DenseLayer>(
        input_features, output_features, use_bias,
        name.empty() ? "dense_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &conv2d(size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h = 1,
                       size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                       bool use_bias = true, const std::string &name = "") {
    Vec<Vec<size_t>> current_shapes = get_current_shape();
    Vec<size_t> current_shape = current_shapes.back();

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
    Vec<Vec<size_t>> current_shapes = get_current_shape();
    Vec<size_t> current_shape = current_shapes.back();

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
    Vec<Vec<size_t>> current_shapes = get_current_shape();
    Vec<size_t> current_shape = current_shapes.back();
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
    Vec<Vec<size_t>> current_shapes = get_current_shape();
    Vec<size_t> current_shape = current_shapes.back();

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
    Vec<Vec<size_t>> current_shapes = get_current_shape();
    Vec<size_t> current_shape = current_shapes.back();

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
    Vec<Vec<size_t>> current_shapes = get_current_shape();
    Vec<size_t> current_shape = current_shapes.back();

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
    Vec<Vec<size_t>> current_shapes = get_current_shape();
    Vec<size_t> current_shape = current_shapes.back();

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
    layers_.push_back(std::move(layer));
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

  LayerBuilder &broadcast_m_seq(Vec<std::unique_ptr<Sequential>> paths,
                                std::unique_ptr<Layer> join_layer, const std::string &name = "") {
    auto broadcast_layer = std::make_unique<MBroadcastLayer>(paths.size());
    auto m_seq_layer = std::make_unique<MSequential>(
        std::move(paths), std::move(join_layer),
        name.empty() ? "mseq_" + std::to_string(layers_.size()) : name);

    Vec<std::unique_ptr<Layer>> wrapper_layers;
    wrapper_layers.push_back(std::move(broadcast_layer));
    wrapper_layers.push_back(std::move(m_seq_layer));

    auto layer = std::make_unique<Sequential>(
        std::move(wrapper_layers),
        name.empty() ? "mseq_wrapper_" + std::to_string(layers_.size()) : name);
    layers_.push_back(std::move(layer));
    return *this;
  }

  LayerBuilder &residual_block(Vec<std::unique_ptr<Layer>> main_path,
                               Vec<std::unique_ptr<Layer>> shortcut_path,
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
    auto main_path = LayerBuilder(get_batchless_current_shape())
                         .conv2d(out_channels, 3, 3, stride, stride, 1, 1, false, name + "_conv1")
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, name + "_bn0")
                         .conv2d(out_channels, 3, 3, 1, 1, 1, 1, false, name + "_conv2")
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::FALSE, name + "_bn1")
                         .build();

    Vec<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder(get_batchless_current_shape())
                     .conv2d(out_channels, 1, 1, stride, stride, 0, 0, false, name + "_conv0")
                     .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::FALSE, name + "_bn0")
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
    // Build main path with pre-activation (BN-ReLU-Conv) ordering
    LayerBuilder main_builder(get_batchless_current_shape());
    main_builder.batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn1")
        .conv2d(out_channels, 3, 3, stride, stride, 1, 1, true)
        .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn2");

    if (dropout_rate > 0.0f) {
      main_builder.dropout(dropout_rate);
    }

    main_builder.conv2d(out_channels, 3, 3, 1, 1, 1, 1, true);

    auto main_path = main_builder.build();
    auto main_seq = std::make_unique<Sequential>(
        std::move(main_path),
        name.empty() ? "main_path_" + std::to_string(layers_.size()) : name + "_main_path");

    // Build shortcut path if needed
    std::unique_ptr<Sequential> shortcut_seq = nullptr;
    if (stride != 1 || in_channels != out_channels) {
      auto shortcut_path = LayerBuilder(get_batchless_current_shape())
                               .conv2d(out_channels, 1, 1, stride, stride, 0, 0, false)
                               .build();
      shortcut_seq = std::make_unique<Sequential>(
          std::move(shortcut_path), name.empty() ? "shortcut_path_" + std::to_string(layers_.size())
                                                 : name + "_shortcut_path");
    }

    auto res_block = std::make_unique<ResidualBlock>(
        std::move(main_seq), std::move(shortcut_seq), "linear",
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
    auto main_path = LayerBuilder(get_batchless_current_shape())
                         .conv2d(mid_channels, 1, 1, 1, 1, 0, 0, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn0")
                         .conv2d(mid_channels, 3, 3, stride, stride, 1, 1, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn1")
                         .conv2d(out_channels, 1, 1, 1, 1, 0, 0, false)
                         .batchnorm(dtype_eps(io_dtype_), 0.1f, true, SBool::TRUE, "bn2")
                         .build();

    Vec<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder(get_batchless_current_shape())
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
    auto main_path = LayerBuilder(get_batchless_current_shape())
                         .legacy_conv2d(out_channels, 3, 3, stride, stride, 1, 1, false)
                         .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                         .legacy_conv2d(out_channels, 3, 3, 1, 1, 1, 1, false)
                         .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                         .build();

    Vec<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder(get_batchless_current_shape())
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
    // Build main path with pre-activation (BN-ReLU-Conv) ordering
    LayerBuilder main_builder(get_batchless_current_shape());
    main_builder.legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
        .legacy_conv2d(out_channels, 3, 3, stride, stride, 1, 1, true)
        .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn2");

    if (dropout_rate > 0.0f) {
      main_builder.dropout(dropout_rate);
    }

    main_builder.legacy_conv2d(out_channels, 3, 3, 1, 1, 1, 1, true);

    auto main_path = main_builder.build();

    Vec<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder(get_batchless_current_shape())
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
    auto main_path = LayerBuilder(get_batchless_current_shape())
                         .legacy_conv2d(mid_channels, 1, 1, 1, 1, 0, 0, false)
                         .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                         .legacy_conv2d(mid_channels, 3, 3, stride, stride, 1, 1, false)
                         .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
                         .legacy_conv2d(out_channels, 1, 1, 1, 1, 0, 0, false)
                         .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn2")
                         .build();

    Vec<std::unique_ptr<Layer>> shortcut;
    if (stride != 1 || in_channels != out_channels) {
      shortcut = LayerBuilder(get_batchless_current_shape())
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
    Vec<Vec<size_t>> current_shape = get_current_shape();

    Vec<Vec<size_t>> batchless_shape;
    for (const auto &shape : current_shape) {
      Vec<size_t> batch_shape = shape;
      batch_shape.erase(batch_shape.begin());  // Remove batch dimension
      batchless_shape.push_back(batch_shape);
    }

    // 1. Attention Sub-block (Residual)
    auto attn_main = LayerBuilder(batchless_shape)
                         .layernorm(dtype_eps(io_dtype_), true, "ln_1")
                         .attention(embed_dim, num_heads, is_causal, "attn")
                         .dropout(dropout_rate)
                         .build();

    auto attn_res = std::make_unique<ResidualBlock>(
        std::move(attn_main), Vec<std::unique_ptr<Layer>>(), "linear", valid_name + "_attn");
    layers_.push_back(std::move(attn_res));

    // 2. Feed-Forward Sub-block (Residual)
    auto ffn_main = LayerBuilder(batchless_shape)
                        .layernorm(dtype_eps(io_dtype_), true, "ln_2")
                        .dense(ffn_dim, true, "mlp_fc1")
                        .activation(activation_fn)
                        .dense(embed_dim, true, "mlp_fc2")
                        .dropout(dropout_rate)
                        .build();

    auto ffn_res = std::make_unique<ResidualBlock>(
        std::move(ffn_main), Vec<std::unique_ptr<Layer>>(), "linear", valid_name + "_ffn");
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
    Vec<Vec<size_t>> current_shape = get_current_shape();

    Vec<Vec<size_t>> batchless_shape;
    for (const auto &shape : current_shape) {
      Vec<size_t> batch_shape = shape;
      batch_shape.erase(batch_shape.begin());  // Remove batch dimension
      batchless_shape.push_back(batch_shape);
    }

    // 1. Attention Sub-block (Residual)
    auto attn_main = LayerBuilder(batchless_shape)
                         .layernorm(dtype_eps(io_dtype_), true, "ln_1")
                         .flash_attention(embed_dim, num_heads, is_causal, "attn")
                         .dropout(dropout_rate)
                         .build();

    auto attn_res = std::make_unique<ResidualBlock>(
        std::move(attn_main), Vec<std::unique_ptr<Layer>>(), "linear", valid_name + "_attn");
    layers_.push_back(std::move(attn_res));

    // 2. Feed-Forward Sub-block (Residual)
    auto ffn_main = LayerBuilder(batchless_shape)
                        .layernorm(dtype_eps(io_dtype_), true, "ln_2")
                        .dense(ffn_dim, true, "mlp_fc1")
                        .activation(activation_fn)
                        .dense(embed_dim, true, "mlp_fc2")
                        .dropout(dropout_rate)
                        .build();

    auto ffn_res = std::make_unique<ResidualBlock>(
        std::move(ffn_main), Vec<std::unique_ptr<Layer>>(), "linear", valid_name + "_ffn");
    layers_.push_back(std::move(ffn_res));

    return *this;
  }

  LayerBuilder &add_layer(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
    return *this;
  }

  Vec<std::unique_ptr<Layer>> build() { return std::move(layers_); }

  const Vec<Vec<size_t>> &get_input_shape() const { return input_shapes_; }
};
}  // namespace tnn