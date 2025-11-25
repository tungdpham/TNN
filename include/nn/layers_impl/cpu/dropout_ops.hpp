#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
template <typename T>
void compute_dropout_forward(const T *input_data, T *output_data, T *mask_data, size_t batch_size,
                             size_t channels, size_t spatial_size, T dropout_rate);
}
} // namespace tnn