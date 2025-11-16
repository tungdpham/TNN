/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/sequential.hpp"

namespace tnn {
// Sequential model instantiations
template class Sequential<float>;
template class SequentialBuilder<float>;
} // namespace tnn