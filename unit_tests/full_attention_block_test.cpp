/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/layers.hpp"
#include "tensor/tensor.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace tnn;

TEST(FullAttentionBlockTest, ForwardPassCPU) {
  size_t batch_size = 2;
  size_t embed_dim = 64;
  size_t num_heads = 4;
  size_t H = 2;
  size_t W = 5;

  // Input shape: [batch, embed_dim, H, W]
  Tensor<float> input({batch_size, embed_dim, H, W}, &getCPU());
  input.fill_random_uniform(-1.0f, 1.0f);

  auto attention = full_attention_block<float>(embed_dim, num_heads, "attn");
  attention->initialize();

  Tensor<float> output;
  attention->forward(input, output);

  // Check output shape
  EXPECT_EQ(output.batch_size(), batch_size);
  EXPECT_EQ(output.channels(), embed_dim);
  EXPECT_EQ(output.height(), H);
  EXPECT_EQ(output.width(), W);
}

TEST(FullAttentionBlockTest, BuilderTest) {
  LayerBuilder<float> builder;
  builder
      .input({64, 2, 5}) // embed_dim, H, W
      .full_attention(64, 4);

  auto layers = builder.build();
  EXPECT_EQ(layers.size(), 1);
  EXPECT_EQ(layers[0]->type(), "full_attention");
}
