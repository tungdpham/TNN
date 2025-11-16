/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tensor/tensor.hpp"

namespace tnn {
namespace cuda {
/**
 * @brief GPU im2col operation
 * Branches to CPU or GPU implementation based on tensor device type
 */
template <typename T>
void im2col(const Tensor<T, NCHW> &input_tensor, T *col_data, size_t kernel_h, size_t kernel_w,
            size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0);

/**
 * @brief GPU col2im operation
 * Branches to CPU or GPU implementation based on tensor device type
 */
template <typename T>
void col2im(const T *col_data, T *result_data, size_t batch_size, size_t channels, size_t height,
            size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w);

/**
 * @brief GPU tensor padding operation
 */
template <typename T, Layout L>
Tensor<T, L> pad(const Tensor<T, L> &input, size_t pad_h, size_t pad_w, T value = T(0));

template <typename T>
Tensor<T, NCHW> pad(const Tensor<T, NCHW> &input, size_t pad_h, size_t pad_w, T value = T(0));

/**
 * @brief GPU tensor unpadding operation
 */
template <typename T, Layout L>
Tensor<T, NCHW> unpad(const Tensor<T, L> &input, size_t pad_h, size_t pad_w);

template <typename T>
Tensor<T, NCHW> unpad(const Tensor<T, NCHW> &input, size_t pad_h, size_t pad_w);

/**
 * @brief GPU tensor crop operation
 */
template <typename T, Layout L>
Tensor<T, L> crop(const Tensor<T, L> &input, const size_t start_h, const size_t start_w,
                  const size_t end_h, const size_t end_w);

template <typename T>
Tensor<T, NCHW> crop(const Tensor<T, NCHW> &input, const size_t start_h, const size_t start_w,
                     const size_t end_h, const size_t end_w);

/**
 * @brief GPU batch slicing operation
 */
template <typename T, Layout L>
Tensor<T, L> slice_batch(const Tensor<T, L> &input, size_t start_batch, size_t end_batch);

/**
 * @brief GPU channel slicing operation
 */
template <typename T, Layout L>
Tensor<T, L> slice_channels(const Tensor<T, L> &input, size_t start_ch, size_t end_ch);

template <typename T>
Tensor<T, NCHW> slice_channels(const Tensor<T, NCHW> &input, size_t start_ch, size_t end_ch);

/**
 * @brief GPU tensor split operation
 */
template <typename T, Layout L>
std::vector<Tensor<T, L>> split(const Tensor<T, L> &input, size_t num_splits);

template <typename T>
std::vector<Tensor<T, NCHW>> split(const Tensor<T, NCHW> &input, size_t num_splits);

/**
 * @brief GPU softmax operation
 */
template <typename T, Layout L> void apply_softmax(Tensor<T, L> &input);

template <typename T> void apply_softmax(Tensor<T, NCHW> &input);

} // namespace cuda

} // namespace tnn