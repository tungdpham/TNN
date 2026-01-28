
/*
 * Copyright (c) 2025 Tung D. Pham
 */
#pragma once

#ifdef USE_CUDNN

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstddef>
#include <cstdint>

namespace tnn {

struct AttentionStats;

namespace cuda {
namespace cudnn_flash_attention {

struct feHandle_t;

feHandle_t *initialize_fe_handle(cudnnHandle_t cudnn_handle, cudnnDataType_t io_data_type,
                                 cudnnDataType_t compute_data_type, AttentionStats &stats);

void destroy_fe_handle(feHandle_t *handle);

void run_forward(feHandle_t *handle, const AttentionStats &stats, const void *q_data,
                 const void *k_data, const void *v_data, void *o_data, void *workspace,
                 cudaStream_t stream);

} // namespace cudnn_flash_attention
} // namespace cuda
} // namespace tnn

#endif
