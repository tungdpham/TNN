/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/example_models.hpp"
#include "nn/sequential.hpp"
#include <fcntl.h>

namespace tnn {

// Static member definitions
template <typename T> std::unordered_map<std::string, Sequential<T>> ExampleModels<T>::creators_;

Sequential<float> create_mnist_cnn() {
  auto model = SequentialBuilder<float>("mnist_cnn")
                   .input({1, 28, 28})
                   .conv2d(8, 5, 5, 1, 1, 0, 0, false, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")
                   .conv2d(16, 1, 1, 1, 1, 0, 0, false, "conv2_1x1")
                   .batchnorm(1e-5f, 0.1f, true, "bn2")
                   .activation("relu", "relu2")
                   .conv2d(48, 5, 5, 1, 1, 0, 0, false, "conv3")
                   .batchnorm(1e-5f, 0.1f, true, "bn3")
                   .activation("relu", "relu3")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                   .flatten(1, "flatten")
                   .dense(10, true, "output")
                   .build();
  return model;
}

Sequential<float> create_cifar10_vgg() {
  auto model = SequentialBuilder<float>("cifar10_vgg")
                   .input({3, 32, 32})
                   .conv2d(64, 3, 3, 1, 1, 1, 1, false, "conv0")
                   .batchnorm(1e-5f, 0.1f, true, "bn0")
                   .activation("relu", "relu0")
                   .conv2d(64, 3, 3, 1, 1, 1, 1, false, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool0")
                   .conv2d(128, 3, 3, 1, 1, 1, 1, false, "conv2")
                   .batchnorm(1e-5f, 0.1f, true, "bn2")
                   .activation("relu", "relu2")
                   .conv2d(128, 3, 3, 1, 1, 1, 1, false, "conv3")
                   .batchnorm(1e-5f, 0.1f, true, "bn3")
                   .activation("relu", "relu3")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool1")
                   .conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv4")
                   .batchnorm(1e-5f, 0.1f, true, "bn4")
                   .activation("relu", "relu4")
                   .conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv5")
                   .activation("relu", "relu5")
                   .conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv6")
                   .batchnorm(1e-5f, 0.1f, true, "bn5")
                   .activation("relu", "relu5")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                   .conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv7")
                   .batchnorm(1e-5f, 0.1f, true, "bn8")
                   .activation("relu", "relu7")
                   .conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv8")
                   .batchnorm(1e-5f, 0.1f, true, "bn9")
                   .activation("relu", "relu8")
                   .conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv9")
                   .batchnorm(1e-5f, 0.1f, true, "bn10")
                   .activation("relu", "relu9")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool3")
                   .flatten(1, "flatten")
                   .dense(512, true, "fc0")
                   .activation("relu", "relu10")
                   .dense(10, true, "fc1")
                   .build();

  return model;
}

Sequential<float> create_cifar10_resnet9() {
  auto model = SequentialBuilder<float>("cifar10_resnet9")
                   .input({3, 32, 32})
                   // Layer 1: 3 -> 64 -> 128 channels, 32x32 -> 16x16
                   .conv2d(64, 3, 3, 1, 1, 1, 1, false, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .conv2d(128, 3, 3, 1, 1, 1, 1, false, "conv2")
                   .batchnorm(1e-5f, 0.1f, true, "bn2")
                   .activation("relu", "relu2")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool1") // 32x32 -> 16x16

                   .basic_residual_block(128, 128, 1, "res_block1")
                   .basic_residual_block(128, 128, 1, "res_block2")

                   // Layer 2: 128 -> 256 channels, 16x16 -> 8x8
                   .conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv3")
                   .batchnorm(1e-5f, 0.1f, true, "bn3")
                   .activation("relu", "relu3")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool2") // 16x16 -> 8x8

                   .basic_residual_block(256, 256, 1, "res_block3")
                   .basic_residual_block(256, 256, 1, "res_block4")

                   // Layer 3: 256 -> 512 channels, 8x8 -> 4x4
                   .conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv4")
                   .batchnorm(1e-5f, 0.1f, true, "bn4")
                   .activation("relu", "relu4")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool3") // 8x8 -> 4x4

                   .basic_residual_block(512, 512, 1, "res_block5")

                   // Classification head
                   .avgpool2d(4, 4, 1, 1, 0, 0, "avgpool") // 4x4 -> 1x1
                   .flatten(1, "flatten")
                   .dense(10, true, "output")
                   .build();
  return model;
}

Sequential<float> create_cifar100_resnet18() {
  auto model = SequentialBuilder<float>("cifar100_resnet18")
                   .input({3, 32, 32})
                   .conv2d(32, 3, 3, 1, 1, 1, 1, false, "conv1")
                   .batchnorm(1e-3f, 0.1f, true, "bn1")
                   //  .groupnorm(32, 1e-5, true, "gn_0")
                   .activation("relu", "relu1")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "maxpool")
                   // Layer 1: 64 channels
                   .basic_residual_block(32, 64, 1, "layer1_block1")
                   .basic_residual_block(64, 64, 1, "layer1_block2")
                   // Layer 2: 128 channels with stride 2
                   .basic_residual_block(64, 128, 2, "layer2_block1")
                   .basic_residual_block(128, 128, 1, "layer2_block2")
                   // Layer 3: 128 channels with stride 2
                   .basic_residual_block(128, 256, 2, "layer3_block1")
                   .basic_residual_block(256, 256, 1, "layer3_block2")
                   // Layer 4: 256 channels with stride 2
                   .basic_residual_block(256, 512, 2, "layer4_block1")
                   .basic_residual_block(512, 512, 1, "layer4_block2")
                   // Global average pooling and classifier
                   .avgpool2d(2, 2, 1, 1, 0, 0, "avgpool")
                   .flatten(1, "flatten")
                   .dense(100, true, "fc")
                   .build();
  return model;
}

Sequential<float> create_cifar100_wrn16_8() {
  constexpr size_t width_factor = 8;
  constexpr float dropout_rate = 0.3f;

  constexpr size_t c1 = 16 * width_factor; // 128
  constexpr size_t c2 = 32 * width_factor; // 256
  constexpr size_t c3 = 64 * width_factor; // 512

  auto model = SequentialBuilder<float>("cifar100_wrn16_8")
                   .input({3, 32, 32})
                   .conv2d(16, 3, 3, 1, 1, 1, 1, true, "conv1")
                   // Group 1: 16 -> 128 channels (2 blocks, stride 1)
                   .wide_residual_block(16, c1, 1, dropout_rate, "group1_block1")
                   .wide_residual_block(c1, c1, 1, dropout_rate, "group1_block2")
                   // Group 2: 128 -> 256 channels (2 blocks, stride 2 for downsampling)
                   .wide_residual_block(c1, c2, 2, dropout_rate, "group2_block1")
                   .wide_residual_block(c2, c2, 1, dropout_rate, "group2_block2")
                   // Group 3: 256 -> 512 channels (2 blocks, stride 2 for downsampling)
                   .wide_residual_block(c2, c3, 2, dropout_rate, "group3_block1")
                   .wide_residual_block(c3, c3, 1, dropout_rate, "group3_block2")
                   // Final BN-ReLU before pooling
                   .batchnorm(1e-5f, 0.1f, true, "bn_final")
                   .activation("relu", "relu_final")
                   // Global average pooling: 8x8 -> 1x1
                   .avgpool2d(8, 8, 1, 1, 0, 0, "avgpool")
                   .flatten(1, "flatten")
                   .dense(100, true, "fc")
                   .build();

  return model;
}

Sequential<float> create_tiny_imagenet_resnet18() {
  auto model = SequentialBuilder<float>("tiny_imagenet_resnet18")
                   .input({3, 64, 64})
                   .conv2d(32, 3, 3, 1, 1, 1, 1, false, "conv1")
                   .batchnorm(1e-3f, 0.1f, true, "bn1")
                   //  .groupnorm(32, 1e-5, true, "gn_0")
                   .activation("relu", "relu1")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "maxpool")
                   // Layer 1: 64 channels
                   .basic_residual_block(32, 64, 1, "layer1_block1")
                   .basic_residual_block(64, 64, 1, "layer1_block2")
                   // Layer 2: 128 channels with stride 2
                   .basic_residual_block(64, 128, 2, "layer2_block1")
                   .basic_residual_block(128, 128, 1, "layer2_block2")
                   // Layer 3: 128 channels with stride 2
                   .basic_residual_block(128, 256, 2, "layer3_block1")
                   .basic_residual_block(256, 256, 1, "layer3_block2")
                   // Layer 4: 256 channels with stride 2
                   .basic_residual_block(256, 512, 2, "layer4_block1")
                   .basic_residual_block(512, 512, 1, "layer4_block2")
                   // Global average pooling and classifier
                   .avgpool2d(4, 4, 1, 1, 0, 0, "avgpool")
                   .flatten(1, "flatten")
                   .dense(200, true, "fc")
                   .build();
  return model;
}

Sequential<float> create_tiny_imagenet_wrn16_8() {
  constexpr size_t width_factor = 8;
  constexpr float dropout_rate = 0.3f;

  constexpr size_t c1 = 16 * width_factor; // 128
  constexpr size_t c2 = 32 * width_factor; // 256
  constexpr size_t c3 = 64 * width_factor; // 512

  auto model = SequentialBuilder<float>("tiny_imagenet_wrn16_8")
                   .input({3, 64, 64})
                   .conv2d(16, 3, 3, 1, 1, 1, 1, true, "conv1")
                   // Group 1: 16 -> 128 channels (2 blocks, stride 1)
                   .wide_residual_block(16, c1, 1, dropout_rate, "group1_block1")
                   .wide_residual_block(c1, c1, 1, dropout_rate, "group1_block2")
                   // Group 2: 128 -> 256 channels (2 blocks, stride 2 for downsampling)
                   .wide_residual_block(c1, c2, 2, dropout_rate, "group2_block1")
                   .wide_residual_block(c2, c2, 1, dropout_rate, "group2_block2")
                   // Group 3: 256 -> 512 channels (2 blocks, stride 2 for downsampling)
                   .wide_residual_block(c2, c3, 2, dropout_rate, "group3_block1")
                   .wide_residual_block(c3, c3, 1, dropout_rate, "group3_block2")
                   // Final BN-ReLU before pooling
                   .batchnorm(1e-5f, 0.1f, true, "bn_final")
                   .activation("relu", "relu_final")
                   // Global average pooling: 8x8 -> 1x1
                   .avgpool2d(8, 8, 1, 1, 0, 0, "avgpool")
                   .flatten(1, "flatten")
                   .dense(200, true, "fc")
                   .build();

  return model;
}

Sequential<float> create_tiny_imagenet_resnet50() {
  auto model = SequentialBuilder<float>("tiny_imagenet_resnet50")
                   .input({3, 64, 64})
                   .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(3, 3, 2, 2, 1, 1, "maxpool")
                   // Layer 1: 64 channels, 3 bottleneck blocks
                   .bottleneck_residual_block(64, 64, 256, 1, "layer1_block1")
                   .bottleneck_residual_block(256, 64, 256, 1, "layer1_block2")
                   .bottleneck_residual_block(256, 64, 256, 1, "layer1_block3")
                   // Layer 2: 128 channels, 4 bottleneck blocks with stride 2
                   .bottleneck_residual_block(256, 128, 512, 2, "layer2_block1")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block2")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block3")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block4")
                   // Layer 3: 256 channels, 6 bottleneck blocks with stride 2
                   .bottleneck_residual_block(512, 256, 1024, 2, "layer3_block1")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block2")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block3")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block4")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block5")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block6")
                   // Layer 4: 512 channels, 3 bottleneck blocks with stride 2
                   .bottleneck_residual_block(1024, 512, 2048, 2, "layer4_block1")
                   .bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block2")
                   .bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block3")
                   // Global average pooling and classifier
                   .avgpool2d(4, 4, 1, 1, 0, 0, "avgpool")
                   .flatten(1, "flatten")
                   .dense(200, true, "fc")
                   .build();
  return model;
}

Sequential<float> create_resnet50_imagenet() {
  auto model = SequentialBuilder<float>("imagenet_resnet50")
                   .input({3, 224, 224})
                   .conv2d(64, 7, 7, 2, 2, 3, 3, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(3, 3, 2, 2, 1, 1, "maxpool")
                   // Layer 1: 64 channels, 3 bottleneck blocks
                   .bottleneck_residual_block(64, 64, 256, 1, "layer1_block1")
                   .bottleneck_residual_block(256, 64, 256, 1, "layer1_block2")
                   .bottleneck_residual_block(256, 64, 256, 1, "layer1_block3")
                   // Layer 2: 128 channels, 4 bottleneck blocks with stride 2
                   .bottleneck_residual_block(256, 128, 512, 2, "layer2_block1")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block2")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block3")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block4")
                   // Layer 3: 256 channels, 6 bottleneck blocks with stride 2
                   .bottleneck_residual_block(512, 256, 1024, 2, "layer3_block1")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block2")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block3")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block4")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block5")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block6")
                   // Layer 4: 512 channels, 3 bottleneck blocks with stride 2
                   .bottleneck_residual_block(1024, 512, 2048, 2, "layer4_block1")
                   .bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block2")
                   .bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block3")
                   // Global average pooling and classifier
                   .avgpool2d(7, 7, 1, 1, 0, 0, "avgpool")
                   .flatten(1, "flatten")
                   .dense(1000, true, "fc")
                   .build();
  return model;
}

Sequential<float> create_tiny_imagenet_vit() {
  constexpr size_t patch_size = 4;
  constexpr size_t embed_dim = 256;
  constexpr size_t num_heads = 4;
  constexpr size_t mlp_ratio = 4;
  constexpr size_t depth = 2;
  constexpr size_t num_classes = 200;
  constexpr size_t num_patches = (64 / patch_size) * (64 / patch_size);
  constexpr size_t seq_len = num_patches + 1;

  SequentialBuilder<float> builder("tiny_imagenet_vit");
  builder.input({3, 64, 64})
      .conv2d(embed_dim, patch_size, patch_size, patch_size, patch_size, 0, 0, true, "patch_embed")
      .flatten(2, "flatten_patches")
      .transpose("transpose_patches")
      .class_token(embed_dim)
      .positional_embedding(embed_dim, seq_len)
      .dropout(0.1f);

  for (size_t i = 0; i < depth; ++i) {
    builder.residual(LayerBuilder<float>()
                         .input({seq_len, embed_dim})
                         .layernorm(1e-5f, true, "ln_attn")
                         .attention(embed_dim, num_heads, false, "attn")
                         .dropout(0.1f)
                         .build(),
                     {}, "linear", "encoder_" + std::to_string(i) + "_attn");

    builder.residual(LayerBuilder<float>()
                         .input({seq_len, embed_dim})
                         .layernorm(1e-5f, true, "ln_mlp")
                         .dense(embed_dim * mlp_ratio, false, "fc1")
                         .activation("gelu")
                         .dropout(0.1f)
                         .dense(embed_dim, false, "fc2")
                         .dropout(0.1f)
                         .build(),
                     {}, "linear", "encoder_" + std::to_string(i) + "_mlp");
  }

  builder.layernorm(1e-5f, true, "ln_final")
      .slice(1, 0, 1, "extract_cls")
      .flatten(1, "flatten_seq")
      .dense(num_classes, true, "head");

  auto model = builder.build();

  return model;
}

Sequential<float> create_gpt2() {
  constexpr size_t seq_len = 512;
  constexpr size_t vocab_size = 50257;
  constexpr size_t embed_dim = 768;
  constexpr size_t num_heads = 12;
  constexpr size_t layers = 12;
  constexpr float dropout = 0.1f;

  SequentialBuilder<float> builder("gpt2");
  builder.input({seq_len})
      .embedding(vocab_size, embed_dim, "token_embed")
      .positional_embedding(embed_dim, seq_len, "pos_embed")
      .dropout(dropout);

  for (size_t i = 0; i < layers; ++i) {
    builder.gpt_block(embed_dim, num_heads, embed_dim * 4, dropout, true, "gelu");
  }

  builder.layernorm(1e-5f, true, "ln_f").dense(vocab_size, true, "head");

  auto model = builder.build();
  return model;
}

// Register all models
template <typename T> void ExampleModels<T>::register_defaults() {
  // MNIST
  register_model(create_mnist_cnn());

  // CIFAR-10
  register_model(create_cifar10_vgg());
  register_model(create_cifar10_resnet9());
  // CIFAR-100
  register_model(create_cifar100_resnet18());
  register_model(create_cifar100_wrn16_8());

  // Tiny ImageNet
  register_model(create_tiny_imagenet_resnet18());
  register_model(create_tiny_imagenet_wrn16_8());
  register_model(create_tiny_imagenet_resnet50());
  register_model(create_tiny_imagenet_vit());

  // ImageNet
  register_model(create_resnet50_imagenet());

  // GPT-2
  register_model(create_gpt2());
}

// Explicit template instantiations
template class ExampleModels<float>;

} // namespace tnn
