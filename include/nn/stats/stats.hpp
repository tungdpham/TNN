#pragma once

#include <cstddef>

namespace tunx {
struct WorkspaceReq {
  size_t fwd_workspace;
  size_t bwd_workspace;
  // TODO: implement inf in engines
  size_t inf_workspace;
};

struct DenseStats {
  size_t batch_size;
  size_t in_features;
  size_t out_features;
  bool use_bias;
};

struct BatchNormStats {
  size_t batch_size;
  size_t height;
  size_t width;
  size_t channels;

  float epsilon;
  float momentum;
  bool use_relu;
};

struct Conv2DStats {
  size_t batch_size;
  size_t in_channels;
  size_t out_channels;
  size_t input_h;
  size_t input_w;
  size_t kernel_h;
  size_t kernel_w;
  size_t stride_h;
  size_t stride_w;
  size_t pad_h;
  size_t pad_w;

  bool use_bias;
};

struct LayerNormStats {
  size_t batch_size;
  size_t seq_len;
  size_t channels;

  float epsilon;
};

struct AvgPool2DStats {
  size_t batch_size;
  size_t height;
  size_t width;
  size_t channels;
  size_t pool_h;
  size_t pool_w;
  size_t stride_h;
  size_t stride_w;
  size_t pad_h;
  size_t pad_w;
};

struct MaxPool2DStats {
  size_t batch_size;
  size_t height;
  size_t width;
  size_t channels;
  size_t pool_h;
  size_t pool_w;
  size_t stride_h;
  size_t stride_w;
  size_t pad_h;
  size_t pad_w;
};

struct DropoutStats {
  size_t batch_size;
  size_t channels;
  size_t spatial_size;
  double dropout_rate;
};

struct EmbeddingStats {
  size_t num_indices;
  size_t vocab_size;
  size_t embed_dim;
  size_t padding_idx;
};

struct ClassTokenStats {
  size_t batch_size;
  size_t seq_len;
  size_t embed_dim;
};

struct ReLUStats {
  size_t batch_size;
  size_t spatial_size;
};

}  // namespace tunx
