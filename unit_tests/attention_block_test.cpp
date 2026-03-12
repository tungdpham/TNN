/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/blocks_impl/attention_block.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "nn/layers.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

TEST(AttentionBlockTest, ForwardPassCPU) {
  size_t batch_size = 2;
  size_t embed_dim = 64;
  size_t num_heads = 4;
  size_t L = 10;

  // Input shape: [batch, L, embed_dim]
  Tensor input = make_tensor<float>({batch_size, L, embed_dim}, getHost());
  input->fill_random_uniform(-1.0f, 1.0f);

  auto attention = std::make_unique<AttentionBlock>(embed_dim, num_heads, "attn");
  attention->init();

  Tensor output;
  attention->forward({input}, {output});

  // Check output shape
  auto output_shape = output->shape();
  EXPECT_EQ(output_shape.size(), 3);
  EXPECT_EQ(output_shape[0], batch_size);
  EXPECT_EQ(output_shape[1], L);
  EXPECT_EQ(output_shape[2], embed_dim);
}

TEST(AttentionBlockTest, BuilderTest) {
  LayerBuilder builder({10, 64});
  builder.attention(64, 4);

  auto layers = builder.build();
  EXPECT_EQ(layers.size(), 1);
  EXPECT_EQ(layers[0]->type(), "attention");
}
