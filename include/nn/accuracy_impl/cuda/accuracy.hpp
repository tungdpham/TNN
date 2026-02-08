/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace tnn {
namespace cuda {
namespace accuracy {

template <typename T>
int compute_class_corrects(const T *predictions, const T *targets, const size_t batch_size,
                           const size_t num_classes, float threshold = 0.5f,
                           cudaStream_t stream = 0);

}  // namespace accuracy
}  // namespace cuda
}  // namespace tnn
