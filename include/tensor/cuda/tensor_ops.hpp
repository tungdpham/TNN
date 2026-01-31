/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDA
#include <cstddef>

#include "tensor/tensor.hpp"

namespace tnn {
namespace cuda {
template <typename T>
void im2col(const Tensor &input_tensor, T *col_data, size_t kernel_h, size_t kernel_w,
            size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
            cudaStream_t stream = 0);

template <typename T>
void col2im(const T *col_data, T *result_data, size_t batch_size, size_t channels, size_t height,
            size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w, cudaStream_t stream = 0);

template <typename T>
void pad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w, T value = T(0),
         cudaStream_t stream = 0);

template <typename T>
void unpad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w,
           cudaStream_t stream = 0);

template <typename T>
void crop(const Tensor &input, Tensor &result, const size_t start_h, const size_t start_w,
          const size_t end_h, const size_t end_w, cudaStream_t stream = 0);

template <typename T>
void slice_batch(const Tensor &input, Tensor &result, size_t start_batch, size_t end_batch,
                 cudaStream_t stream = 0);

template <typename T>
void split(const Tensor &input, std::vector<Tensor> &results, size_t num_splits,
           cudaStream_t stream = 0);

template <typename T>
void apply_softmax(Tensor &input, cudaStream_t stream = 0);

}  // namespace cuda

}  // namespace tnn

#endif