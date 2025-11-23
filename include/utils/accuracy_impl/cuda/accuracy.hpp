/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>

namespace tnn {
namespace cuda {
namespace accuracy {

// Compute class accuracy
float compute_class_accuracy(const float *predictions, const float *targets,
                             const size_t batch_size, const size_t num_classes);

// Compute class corrects
int compute_class_corrects(const float *predictions, const float *targets, const size_t batch_size,
                           const size_t num_classes, float threshold = 0.5f);

} // namespace accuracy
} // namespace cuda
} // namespace tnn
