/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "matrix/matrix.hpp"
#include "nn/activations.hpp"
#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"

namespace tnn {
// Sequential model instantiations
template class Sequential<float>;
template class SequentialBuilder<float>;
} // namespace tnn