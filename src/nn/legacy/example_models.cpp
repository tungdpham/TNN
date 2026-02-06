/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/legacy/example_models.hpp"

#include <fcntl.h>

#include "nn/layers.hpp"
#include "nn/sequential.hpp"
#include "type/type.hpp"

namespace tnn {
namespace legacy {
// Static member definitions
std::unordered_map<std::string, std::function<Sequential(DType_t)>> ExampleModels::creators_;

Sequential create_mnist_cnn(DType_t io_dtype_ = DType_t::FP32) {
  auto layers = LayerBuilder()
                    .input({1, 28, 28})
                    .dtype(io_dtype_)
                    .legacy_conv2d(8, 5, 5, 1, 1, 0, 0, false, "conv1")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
                    .legacy_maxpool2d(3, 3, 3, 3, 0, 0, "pool1")
                    .legacy_conv2d(16, 1, 1, 1, 1, 0, 0, false, "conv2_1x1")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn2_1x1")
                    .activation("relu", "relu2")
                    .legacy_conv2d(48, 5, 5, 1, 1, 0, 0, false, "conv3")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn3")
                    .legacy_maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                    .flatten(1, -1, "flatten")
                    .legacy_dense(10, false, "output")
                    .build();
  return Sequential("mnist_cnn", std::move(layers));
}

Sequential create_cifar10_vgg(DType_t io_dtype_ = DType_t::FP32) {
  auto layers = LayerBuilder()
                    .input({3, 32, 32})
                    .dtype(io_dtype_)
                    .legacy_conv2d(64, 3, 3, 1, 1, 1, 1, false, "conv0")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn0")
                    .legacy_conv2d(64, 3, 3, 1, 1, 1, 1, false, "conv1")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
                    .legacy_maxpool2d(2, 2, 2, 2, 0, 0, "pool0")
                    .legacy_conv2d(128, 3, 3, 1, 1, 1, 1, false, "conv2")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn2")
                    .legacy_conv2d(128, 3, 3, 1, 1, 1, 1, false, "conv3")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn3")
                    .legacy_maxpool2d(2, 2, 2, 2, 0, 0, "pool1")
                    .legacy_conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv4")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn4")
                    .legacy_conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv5")
                    .activation("relu", "relu5")
                    .legacy_conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv6")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn6")
                    .legacy_maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                    .legacy_conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv7")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn7")
                    .legacy_conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv8")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn8")
                    .legacy_conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv9")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn9")
                    .legacy_maxpool2d(2, 2, 2, 2, 0, 0, "pool3")
                    .flatten(1, -1, "flatten")
                    .legacy_dense(512, true, "fc0")
                    .activation("relu", "relu10")
                    .legacy_dense(10, true, "fc1")
                    .build();
  return Sequential("cifar10_vgg", std::move(layers));
}

Sequential create_cifar10_resnet9(DType_t io_dtype_ = DType_t::FP32) {
  auto layers = LayerBuilder()
                    .input({3, 32, 32})
                    .dtype(io_dtype_)
                    // Layer 1: 3 -> 64 -> 128 channels, 32x32 -> 16x16
                    .legacy_conv2d(64, 3, 3, 1, 1, 1, 1, false, "conv1")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
                    .legacy_conv2d(128, 3, 3, 1, 1, 1, 1, false, "conv2")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn2")
                    .legacy_maxpool2d(2, 2, 2, 2, 0, 0, "pool1")  // 32x32 -> 16x16
                    .legacy_basic_residual_block(128, 128, 1, "res_block1")
                    .legacy_basic_residual_block(128, 128, 1, "res_block2")
                    // Layer 2: 128 -> 256 channels, 16x16 -> 8x8
                    .legacy_conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv3")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn3")
                    .legacy_maxpool2d(2, 2, 2, 2, 0, 0, "pool2")  // 16x16 -> 8x8
                    .legacy_basic_residual_block(256, 256, 1, "res_block3")
                    .legacy_basic_residual_block(256, 256, 1, "res_block4")
                    // Layer 3: 256 -> 512 channels, 8x8 -> 4x4
                    .legacy_conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv4")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn4")
                    .legacy_maxpool2d(2, 2, 2, 2, 0, 0, "pool3")  // 8x8 -> 4x4
                    .legacy_basic_residual_block(512, 512, 1, "res_block5")
                    // Classification head
                    .legacy_avgpool2d(4, 4, 1, 1, 0, 0, "avgpool")  // 4x4 -> 1x1
                    .flatten(1, -1, "flatten")
                    // .legacy_dense(10, true, "output")
                    .legacy_dense(10, true, "output")
                    .build();
  return Sequential("cifar10_resnet9", std::move(layers));
}

Sequential create_cifar100_resnet18(DType_t io_dtype_ = DType_t::FP32) {
  auto layers = LayerBuilder()
                    .input({3, 32, 32})
                    .dtype(io_dtype_)
                    .legacy_conv2d(32, 3, 3, 1, 1, 1, 1, false, "conv1")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
                    .legacy_maxpool2d(2, 2, 2, 2, 0, 0, "maxpool")
                    // Layer 1: 64 channels
                    .legacy_basic_residual_block(32, 64, 1, "layer1_block1")
                    .legacy_basic_residual_block(64, 64, 1, "layer1_block2")
                    // Layer 2: 128 channels with stride 2
                    .legacy_basic_residual_block(64, 128, 2, "layer2_block1")
                    .legacy_basic_residual_block(128, 128, 1, "layer2_block2")
                    // Layer 3: 128 channels with stride 2
                    .legacy_basic_residual_block(128, 256, 2, "layer3_block1")
                    .legacy_basic_residual_block(256, 256, 1, "layer3_block2")
                    // Layer 4: 256 channels with stride 2
                    .legacy_basic_residual_block(256, 512, 2, "layer4_block1")
                    .legacy_basic_residual_block(512, 512, 1, "layer4_block2")
                    // Global average pooling and classifier
                    .legacy_avgpool2d(2, 2, 1, 1, 0, 0, "avgpool")
                    .flatten(1, -1, "flatten")
                    .legacy_dense(100, true, "fc")
                    .build();
  return Sequential("cifar100_resnet18", std::move(layers));
}

Sequential create_cifar100_wrn16_8(DType_t io_dtype_ = DType_t::FP32) {
  constexpr size_t width_factor = 8;
  constexpr float dropout_rate = 0.3f;

  constexpr size_t c1 = 16 * width_factor;  // 128
  constexpr size_t c2 = 32 * width_factor;  // 256
  constexpr size_t c3 = 64 * width_factor;  // 512

  auto layers = LayerBuilder()
                    .input({3, 32, 32})
                    .dtype(io_dtype_)
                    .legacy_conv2d(16, 3, 3, 1, 1, 1, 1, true, "conv1")
                    // Group 1: 16 -> 128 channels (2 blocks, stride 1)
                    .legacy_wide_residual_block(16, c1, 1, dropout_rate, "group1_block1")
                    .legacy_wide_residual_block(c1, c1, 1, dropout_rate, "group1_block2")
                    // Group 2: 128 -> 256 channels (2 blocks, stride 2 for downsampling)
                    .legacy_wide_residual_block(c1, c2, 2, dropout_rate, "group2_block1")
                    .legacy_wide_residual_block(c2, c2, 1, dropout_rate, "group2_block2")
                    // Group 3: 256 -> 512 channels (2 blocks, stride 2 for downsampling)
                    .legacy_wide_residual_block(c2, c3, 2, dropout_rate, "group3_block1")
                    .legacy_wide_residual_block(c3, c3, 1, dropout_rate, "group3_block2")
                    // Final BN-ReLU before pooling
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn_final")
                    // Global average pooling: 8x8 -> 1x1
                    .legacy_avgpool2d(8, 8, 1, 1, 0, 0, "avgpool")
                    .flatten(1, -1, "flatten")
                    .legacy_dense(100, true, "fc")
                    .build();

  return Sequential("cifar100_wrn16_8", std::move(layers));
}

Sequential create_tiny_imagenet_resnet18(DType_t io_dtype_ = DType_t::FP32) {
  auto layers = LayerBuilder()
                    .input({3, 64, 64})
                    .dtype(io_dtype_)
                    .legacy_conv2d(32, 3, 3, 1, 1, 1, 1, false, "conv1")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
                    .legacy_maxpool2d(2, 2, 2, 2, 0, 0, "maxpool")
                    // Layer 1: 64 channels
                    .legacy_basic_residual_block(32, 64, 1, "layer1_block1")
                    .legacy_basic_residual_block(64, 64, 1, "layer1_block2")
                    // Layer 2: 128 channels with stride 2
                    .legacy_basic_residual_block(64, 128, 2, "layer2_block1")
                    .legacy_basic_residual_block(128, 128, 1, "layer2_block2")
                    // Layer 3: 128 channels with stride 2
                    .legacy_basic_residual_block(128, 256, 2, "layer3_block1")
                    .legacy_basic_residual_block(256, 256, 1, "layer3_block2")
                    // Layer 4: 256 channels with stride 2
                    .legacy_basic_residual_block(256, 512, 2, "layer4_block1")
                    .legacy_basic_residual_block(512, 512, 1, "layer4_block2")
                    // Global average pooling and classifier
                    .legacy_avgpool2d(4, 4, 1, 1, 0, 0, "avgpool")
                    .flatten(1, -1, "flatten")
                    .legacy_dense(200, true, "fc")
                    .build();
  return Sequential("tiny_imagenet_resnet18", std::move(layers));
}

Sequential create_tiny_imagenet_wrn16_8(DType_t io_dtype_ = DType_t::FP32) {
  constexpr size_t width_factor = 8;
  constexpr float dropout_rate = 0.3f;

  constexpr size_t c1 = 16 * width_factor;  // 128
  constexpr size_t c2 = 32 * width_factor;  // 256
  constexpr size_t c3 = 64 * width_factor;  // 512

  auto layers = LayerBuilder()
                    .input({3, 64, 64})
                    .dtype(io_dtype_)
                    .legacy_conv2d(16, 3, 3, 1, 1, 1, 1, true, "conv1")
                    // Group 1: 16 -> 128 channels (2 blocks, stride 1)
                    .legacy_wide_residual_block(16, c1, 1, dropout_rate, "group1_block1")
                    .legacy_wide_residual_block(c1, c1, 1, dropout_rate, "group1_block2")
                    // Group 2: 128 -> 256 channels (2 blocks, stride 2 for downsampling)
                    .legacy_wide_residual_block(c1, c2, 2, dropout_rate, "group2_block1")
                    .legacy_wide_residual_block(c2, c2, 1, dropout_rate, "group2_block2")
                    // Group 3: 256 -> 512 channels (2 blocks, stride 2 for downsampling)
                    .legacy_wide_residual_block(c2, c3, 2, dropout_rate, "group3_block1")
                    .legacy_wide_residual_block(c3, c3, 1, dropout_rate, "group3_block2")
                    // Final BN-ReLU before pooling
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn_final")
                    // Global average pooling: 8x8 -> 1x1
                    .legacy_avgpool2d(8, 8, 1, 1, 0, 0, "avgpool")
                    .flatten(1, -1, "flatten")
                    .legacy_dense(200, true, "fc")
                    .build();

  return Sequential("tiny_imagenet_wrn16_8", std::move(layers));
}

Sequential create_tiny_imagenet_resnet50(DType_t io_dtype_ = DType_t::FP32) {
  auto layers = LayerBuilder()
                    .input({3, 64, 64})
                    .dtype(io_dtype_)
                    .legacy_conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv1")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
                    .legacy_maxpool2d(3, 3, 2, 2, 1, 1, "maxpool")
                    // Layer 1: 64 channels, 3 bottleneck blocks
                    .legacy_bottleneck_residual_block(64, 64, 256, 1, "layer1_block1")
                    .legacy_bottleneck_residual_block(256, 64, 256, 1, "layer1_block2")
                    .legacy_bottleneck_residual_block(256, 64, 256, 1, "layer1_block3")
                    // Layer 2: 128 channels, 4 bottleneck blocks with stride 2
                    .legacy_bottleneck_residual_block(256, 128, 512, 2, "layer2_block1")
                    .legacy_bottleneck_residual_block(512, 128, 512, 1, "layer2_block2")
                    .legacy_bottleneck_residual_block(512, 128, 512, 1, "layer2_block3")
                    .legacy_bottleneck_residual_block(512, 128, 512, 1, "layer2_block4")
                    // Layer 3: 256 channels, 6 bottleneck blocks with stride 2
                    .legacy_bottleneck_residual_block(512, 256, 1024, 2, "layer3_block1")
                    .legacy_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block2")
                    .legacy_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block3")
                    .legacy_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block4")
                    .legacy_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block5")
                    .legacy_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block6")
                    // Layer 4: 512 channels, 3 bottleneck blocks with stride 2
                    .legacy_bottleneck_residual_block(1024, 512, 2048, 2, "layer4_block1")
                    .legacy_bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block2")
                    .legacy_bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block3")
                    // Global average pooling and classifier
                    .legacy_avgpool2d(4, 4, 1, 1, 0, 0, "avgpool")
                    .flatten(1, -1, "flatten")
                    .legacy_dense(200, true, "fc")
                    .build();
  return Sequential("tiny_imagenet_resnet50", std::move(layers));
}

Sequential create_resnet50_imagenet(DType_t io_dtype_ = DType_t::FP32) {
  auto layers = LayerBuilder()
                    .input({3, 224, 224})
                    .dtype(io_dtype_)
                    .legacy_conv2d(64, 7, 7, 2, 2, 3, 3, true, "conv1")
                    .legacy_batchnorm(dtype_eps(io_dtype_), 0.1f, true, "bn1")
                    .legacy_maxpool2d(3, 3, 2, 2, 1, 1, "maxpool")
                    // Layer 1: 64 channels, 3 bottleneck blocks
                    .legacy_bottleneck_residual_block(64, 64, 256, 1, "layer1_block1")
                    .legacy_bottleneck_residual_block(256, 64, 256, 1, "layer1_block2")
                    .legacy_bottleneck_residual_block(256, 64, 256, 1, "layer1_block3")
                    // Layer 2: 128 channels, 4 bottleneck blocks with stride 2
                    .legacy_bottleneck_residual_block(256, 128, 512, 2, "layer2_block1")
                    .legacy_bottleneck_residual_block(512, 128, 512, 1, "layer2_block2")
                    .legacy_bottleneck_residual_block(512, 128, 512, 1, "layer2_block3")
                    .legacy_bottleneck_residual_block(512, 128, 512, 1, "layer2_block4")
                    // Layer 3: 256 channels, 6 bottleneck blocks with stride 2
                    .legacy_bottleneck_residual_block(512, 256, 1024, 2, "layer3_block1")
                    .legacy_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block2")
                    .legacy_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block3")
                    .legacy_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block4")
                    .legacy_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block5")
                    .legacy_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block6")
                    // Layer 4: 512 channels, 3 bottleneck blocks with stride 2
                    .legacy_bottleneck_residual_block(1024, 512, 2048, 2, "layer4_block1")
                    .legacy_bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block2")
                    .legacy_bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block3")
                    // Global average pooling and classifier
                    .legacy_avgpool2d(7, 7, 1, 1, 0, 0, "avgpool")
                    .flatten(1, -1, "flatten")
                    .legacy_dense(1000, true, "fc")
                    .build();
  return Sequential("imagenet_resnet50", std::move(layers));
}

Sequential create_tiny_imagenet_vit(DType_t io_dtype_ = DType_t::FP32) {
  constexpr size_t patch_size = 4;
  constexpr size_t embed_dim = 256;
  constexpr size_t num_heads = 4;
  constexpr size_t mlp_ratio = 4;
  constexpr size_t depth = 4;
  constexpr size_t num_classes = 200;
  constexpr size_t num_patches = (64 / patch_size) * (64 / patch_size);
  constexpr size_t seq_len = num_patches + 1;

  LayerBuilder builder;
  builder.input({3, 64, 64})
      .dtype(io_dtype_)
      .legacy_conv2d(embed_dim, patch_size, patch_size, patch_size, patch_size, 0, 0, true,
                     "patch_embed")
      .flatten(1, 2, "flatten_patches")  // Flatten dims 1-2 (H, W), keep dim 3 (C)
      .class_token(embed_dim)
      .positional_embedding(embed_dim, seq_len)
      .dropout(0.1f);

  for (size_t i = 0; i < depth; ++i) {
    builder.residual_block(LayerBuilder()
                               .input({seq_len, embed_dim})
                               .dtype(io_dtype_)
                               .layernorm(dtype_eps(io_dtype_), true, "ln_attn")
                               .attention(embed_dim, num_heads, false, "attn")
                               .dropout(0.1f)
                               .build(),
                           {}, "linear", "encoder_" + std::to_string(i) + "_attn");

    builder.residual_block(LayerBuilder()
                               .input({seq_len, embed_dim})
                               .dtype(io_dtype_)
                               .layernorm(dtype_eps(io_dtype_), true, "ln_mlp")
                               .legacy_dense(embed_dim * mlp_ratio, false, "fc1")
                               .activation("gelu")
                               .dropout(0.1f)
                               .legacy_dense(embed_dim, false, "fc2")
                               .dropout(0.1f)
                               .build(),
                           {}, "linear", "encoder_" + std::to_string(i) + "_mlp");
  }

  builder
      .layernorm(dtype_eps(io_dtype_), true, "ln_final")
      // .slice(1, 0, 1, "extract_cls")
      .flatten(1, -1, "flatten_seq")
      .legacy_dense(num_classes, true, "head");

  auto layers = builder.build();

  return Sequential("tiny_imagenet_vit", std::move(layers));
}

Sequential create_gpt2_small(DType_t io_dtype_ = DType_t::FP32) {
  constexpr size_t seq_len = 1024;
  constexpr size_t vocab_size = 50257;
  constexpr size_t embed_dim = 768;
  constexpr size_t num_heads = 12;
  constexpr size_t num_layers = 12;
  constexpr float dropout = 0.1f;

  LayerBuilder builder;
  builder.input({seq_len})
      .dtype(io_dtype_)
      .embedding(vocab_size, embed_dim, "token_embed")
      .positional_embedding(embed_dim, seq_len, "pos_embed")
      .dropout(dropout);

  for (size_t i = 0; i < num_layers; ++i) {
    builder.gpt_block(embed_dim, num_heads, embed_dim * 4, dropout, true, "gelu");
  }

  builder.layernorm(dtype_eps(io_dtype_), true, "ln_f").legacy_dense(vocab_size, true, "head");

  auto layers = builder.build();
  return Sequential("gpt2_small", std::move(layers));
}

Sequential create_flash_gpt2_small(DType_t io_dtype_ = DType_t::FP32) {
  constexpr size_t seq_len = 1024;
  constexpr size_t vocab_size = 50257;
  constexpr size_t embed_dim = 768;
  constexpr size_t num_heads = 12;
  constexpr size_t num_layers = 12;
  constexpr float dropout = 0.1f;

  LayerBuilder builder;
  builder.input({seq_len})
      .dtype(io_dtype_)
      .embedding(vocab_size, embed_dim, "token_embed")
      .positional_embedding(embed_dim, seq_len, "pos_embed")
      .dropout(dropout);

  for (size_t i = 0; i < num_layers; ++i) {
    builder.flash_gpt_block(embed_dim, num_heads, embed_dim * 4, dropout, true, "gelu");
  }

  builder.layernorm(dtype_eps(io_dtype_), true, "ln_f").legacy_dense(vocab_size, true, "head");

  auto layers = builder.build();
  return Sequential("flash_gpt2_small", std::move(layers));
}

Sequential create_gpt2_medium(DType_t io_dtype_ = DType_t::FP32) {
  constexpr size_t seq_len = 1024;
  constexpr size_t vocab_size = 50257;
  constexpr size_t embed_dim = 1024;
  constexpr size_t num_heads = 16;
  constexpr size_t num_layers = 24;
  constexpr float dropout = 0.1f;

  LayerBuilder builder;
  builder.input({seq_len})
      .dtype(io_dtype_)
      .embedding(vocab_size, embed_dim, "token_embed")
      .positional_embedding(embed_dim, seq_len, "pos_embed")
      .dropout(dropout);

  for (size_t i = 0; i < num_layers; ++i) {
    builder.gpt_block(embed_dim, num_heads, embed_dim * 4, dropout, true, "gelu");
  }

  builder.layernorm(dtype_eps(io_dtype_), true, "ln_f").legacy_dense(vocab_size, true, "head");

  auto layers = builder.build();
  return Sequential("gpt2_medium", std::move(layers));
}

Sequential create_flash_gpt2_medium(DType_t io_dtype_ = DType_t::FP32) {
  constexpr size_t seq_len = 1024;
  constexpr size_t vocab_size = 50257;
  constexpr size_t embed_dim = 1024;
  constexpr size_t num_heads = 16;
  constexpr size_t num_layers = 24;
  constexpr float dropout = 0.1f;

  LayerBuilder builder;
  builder.input({seq_len})
      .dtype(io_dtype_)
      .embedding(vocab_size, embed_dim, "token_embed")
      .positional_embedding(embed_dim, seq_len, "pos_embed")
      .dropout(dropout);

  for (size_t i = 0; i < num_layers; ++i) {
    builder.flash_gpt_block(embed_dim, num_heads, embed_dim * 4, dropout, true, "gelu");
  }

  builder.layernorm(dtype_eps(io_dtype_), true, "ln_f").legacy_dense(vocab_size, true, "head");

  auto layers = builder.build();
  return Sequential("flash_gpt2_medium", std::move(layers));
}

Sequential create_gpt2_large(DType_t io_dtype_ = DType_t::FP32) {
  constexpr size_t seq_len = 1024;
  constexpr size_t vocab_size = 50257;
  constexpr size_t embed_dim = 1280;
  constexpr size_t num_heads = 20;
  constexpr size_t num_layers = 36;
  constexpr float dropout = 0.1f;

  LayerBuilder builder;
  builder.input({seq_len})
      .dtype(io_dtype_)
      .embedding(vocab_size, embed_dim, "token_embed")
      .positional_embedding(embed_dim, seq_len, "pos_embed")
      .dropout(dropout);

  for (size_t i = 0; i < num_layers; ++i) {
    builder.gpt_block(embed_dim, num_heads, embed_dim * 4, dropout, true, "gelu");
  }

  builder.layernorm(dtype_eps(io_dtype_), true, "ln_f").legacy_dense(vocab_size, true, "head");

  auto layers = builder.build();
  return Sequential("gpt2_large", std::move(layers));
}

Sequential create_flash_gpt2_large(DType_t io_dtype_ = DType_t::FP32) {
  constexpr size_t seq_len = 1024;
  constexpr size_t vocab_size = 50257;
  constexpr size_t embed_dim = 1280;
  constexpr size_t num_heads = 20;
  constexpr size_t num_layers = 36;
  constexpr float dropout = 0.1f;

  LayerBuilder builder;
  builder.input({seq_len})
      .dtype(io_dtype_)
      .embedding(vocab_size, embed_dim, "token_embed")
      .positional_embedding(embed_dim, seq_len, "pos_embed")
      .dropout(dropout);

  for (size_t i = 0; i < num_layers; ++i) {
    builder.flash_gpt_block(embed_dim, num_heads, embed_dim * 4, dropout, true, "gelu");
  }

  builder.layernorm(dtype_eps(io_dtype_), true, "ln_f").legacy_dense(vocab_size, true, "head");

  auto layers = builder.build();
  return Sequential("flash_gpt2_large", std::move(layers));
}

// Register all models
void ExampleModels::register_defaults() {
  // MNIST
  register_model(create_mnist_cnn);

  // CIFAR-10
  register_model(create_cifar10_vgg);
  register_model(create_cifar10_resnet9);
  // CIFAR-100
  register_model(create_cifar100_resnet18);
  register_model(create_cifar100_wrn16_8);

  // Tiny ImageNet
  register_model(create_tiny_imagenet_resnet18);
  register_model(create_tiny_imagenet_wrn16_8);
  register_model(create_tiny_imagenet_resnet50);
  register_model(create_tiny_imagenet_vit);

  // ImageNet
  register_model(create_resnet50_imagenet);

  // GPT-2
  register_model(create_gpt2_small);
  register_model(create_flash_gpt2_small);

  register_model(create_gpt2_medium);
  register_model(create_flash_gpt2_medium);

  register_model(create_gpt2_large);
  register_model(create_flash_gpt2_large);
}

}  // namespace legacy
}  // namespace tnn
