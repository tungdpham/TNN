#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
namespace dropout {

template <typename T>
void compute_dropout_forward(const T *input_data, T *output_data, bool *mask_data,
                             size_t batch_size, size_t channels, size_t spatial_size,
                             T dropout_rate);

template <typename T>
void compute_dropout_backward(const T *grad_output_data, T *grad_input_data, const bool *mask_data,
                              size_t batch_size, size_t channels, size_t spatial_size, T scale);

}  // namespace dropout
}  // namespace cpu
}  // namespace tnn