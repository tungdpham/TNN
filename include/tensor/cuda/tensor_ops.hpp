/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tensor/tensor.hpp"
#include <cstddef>

namespace tnn {
namespace cuda {
/**
 * @brief GPU im2col operation
 * Branches to CPU or GPU implementation based on tensor device type
 */
template <typename T>
void im2col(const Tensor<T, NCHW> &input_tensor, T *col_data, size_t kernel_h, size_t kernel_w,
            size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
            cudaStream_t stream = 0);

/**
 * @brief GPU col2im operation
 * Branches to CPU or GPU implementation based on tensor device type
 */
template <typename T>
void col2im(const T *col_data, T *result_data, size_t batch_size, size_t channels, size_t height,
            size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w, cudaStream_t stream = 0);

/**
 * @brief GPU tensor padding operation
 */
template <typename T, Layout L>
void pad(const Tensor<T, L> &input, Tensor<T, L> &result, size_t pad_h, size_t pad_w,
         T value = T(0), cudaStream_t stream = 0);

template <typename T>
void pad(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result, size_t pad_h, size_t pad_w,
         T value = T(0), cudaStream_t stream = 0);

/**
 * @brief GPU tensor unpadding operation
 */
template <typename T, Layout L>
void unpad(const Tensor<T, L> &input, Tensor<T, NCHW> &result, size_t pad_h, size_t pad_w,
           cudaStream_t stream = 0);

template <typename T>
void unpad(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result, size_t pad_h, size_t pad_w,
           cudaStream_t stream = 0);

/**
 * @brief GPU tensor crop operation
 */
template <typename T, Layout L>
void crop(const Tensor<T, L> &input, Tensor<T, L> &result, const size_t start_h,
          const size_t start_w, const size_t end_h, const size_t end_w, cudaStream_t stream = 0);

template <typename T>
void crop(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result, const size_t start_h,
          const size_t start_w, const size_t end_h, const size_t end_w, cudaStream_t stream = 0);

/**
 * @brief GPU batch slicing operation
 */
template <typename T, Layout L>
void slice_batch(const Tensor<T, L> &input, Tensor<T, L> &result, size_t start_batch,
                 size_t end_batch, cudaStream_t stream = 0);

/**
 * @brief GPU channel slicing operation
 */
template <typename T, Layout L>
void slice_channels(const Tensor<T, L> &input, Tensor<T, L> &result, size_t start_ch, size_t end_ch,
                    cudaStream_t stream = 0);

template <typename T>
void slice_channels(const Tensor<T, NCHW> &input, Tensor<T, NCHW> &result, size_t start_ch,
                    size_t end_ch, cudaStream_t stream = 0);

/**
 * @brief GPU tensor split operation
 */
template <typename T, Layout L>
void split(const Tensor<T, L> &input, std::vector<Tensor<T, L>> &results, size_t num_splits,
           cudaStream_t stream = 0);

template <typename T>
void split(const Tensor<T, NCHW> &input, std::vector<Tensor<T, NCHW>> &results, size_t num_splits,
           cudaStream_t stream = 0);

/**
 * @brief GPU softmax operation
 */
template <typename T, Layout L> void apply_softmax(Tensor<T, L> &input, cudaStream_t stream = 0);

template <typename T> void apply_softmax(Tensor<T, NCHW> &input, cudaStream_t stream = 0);

} // namespace cuda

} // namespace tnn