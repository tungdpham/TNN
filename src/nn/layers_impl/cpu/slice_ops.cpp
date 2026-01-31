/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/slice_ops.hpp"

#include <cstring>
#include <numeric>

#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace slice {

template <typename T>
void slice_forward(const T *input, T *output, const std::vector<size_t> &input_shape, size_t axis,
                   size_t start, size_t length) {
  size_t outer_size = 1;
  for (size_t i = 0; i < axis; ++i) {
    outer_size *= input_shape[i];
  }

  size_t inner_size = 1;
  for (size_t i = axis + 1; i < input_shape.size(); ++i) {
    inner_size *= input_shape[i];
  }

  size_t axis_size = input_shape[axis];
  size_t copy_bytes = length * inner_size * sizeof(T);
  size_t src_stride_bytes = axis_size * inner_size * sizeof(T);
  size_t dst_stride_bytes = length * inner_size * sizeof(T);

  const char *src_byte_ptr = reinterpret_cast<const char *>(input);
  char *dst_byte_ptr = reinterpret_cast<char *>(output);

  src_byte_ptr += start * inner_size * sizeof(T);

  for (size_t i = 0; i < outer_size; ++i) {
    std::memcpy(dst_byte_ptr, src_byte_ptr, copy_bytes);
    src_byte_ptr += src_stride_bytes;
    dst_byte_ptr += dst_stride_bytes;
  }
}

template <typename T>
void slice_backward(const T *gradient, T *grad_input, const std::vector<size_t> &input_shape,
                    size_t axis, size_t start, size_t length) {
  size_t total_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  std::memset(grad_input, 0, total_elements * sizeof(T));

  size_t outer_size = 1;
  for (size_t i = 0; i < axis; ++i) {
    outer_size *= input_shape[i];
  }

  size_t inner_size = 1;
  for (size_t i = axis + 1; i < input_shape.size(); ++i) {
    inner_size *= input_shape[i];
  }

  size_t axis_size = input_shape[axis];
  size_t copy_bytes = length * inner_size * sizeof(T);
  size_t src_stride_bytes = length * inner_size * sizeof(T);
  size_t dst_stride_bytes = axis_size * inner_size * sizeof(T);

  const char *src_byte_ptr = reinterpret_cast<const char *>(gradient);
  char *dst_byte_ptr = reinterpret_cast<char *>(grad_input);

  dst_byte_ptr += start * inner_size * sizeof(T);

  for (size_t i = 0; i < outer_size; ++i) {
    std::memcpy(dst_byte_ptr, src_byte_ptr, copy_bytes);
    src_byte_ptr += src_stride_bytes;
    dst_byte_ptr += dst_stride_bytes;
  }
}

#define INSTANTIATE_SLICE(T)                                                           \
  template void slice_forward<T>(const T *input, T *output,                            \
                                 const std::vector<size_t> &input_shape, size_t axis,  \
                                 size_t start, size_t length);                         \
                                                                                       \
  template void slice_backward<T>(const T *gradient, T *grad_input,                    \
                                  const std::vector<size_t> &input_shape, size_t axis, \
                                  size_t start, size_t length);
INSTANTIATE_SLICE(fp16)
INSTANTIATE_SLICE(bf16)
INSTANTIATE_SLICE(float)
INSTANTIATE_SLICE(double)
#undef INSTANTIATE_SLICE

}  // namespace slice
}  // namespace cpu
}  // namespace tnn
