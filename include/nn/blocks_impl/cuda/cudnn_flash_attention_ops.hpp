
/*
 * Copyright (c) 2025 Tung D. Pham
 */
#pragma once

#ifdef USE_CUDNN

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cudnn_frontend {
namespace graph {
class Graph;
}
} // namespace cudnn_frontend

namespace tnn {
namespace cuda {
namespace cudnn_flash_attention {

using GraphPtr = std::shared_ptr<cudnn_frontend::graph::Graph>;

GraphPtr create_sdpa_forward_graph(int64_t b, int64_t h, int64_t s, int64_t d, float attn_scale,
                                   bool is_causal);

void build_sdpa_forward_graph(const GraphPtr &graph, cudnnHandle_t handle);

size_t get_sdpa_forward_workspace_bytes(const GraphPtr &graph);

void run_sdpa_forward(const GraphPtr &graph, cudnnHandle_t handle, void *q, void *k, void *v,
                      void *o, void *workspace, cudaStream_t stream);

} // namespace cudnn_flash_attention
} // namespace cuda
} // namespace tnn

#endif
