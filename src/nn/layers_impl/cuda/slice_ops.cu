/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cuda/slice_ops.hpp"
#include <vector>

#ifdef USE_CUDA

namespace tnn {
namespace cuda {
namespace slice {

template <typename T>
__global__ void slice_forward_kernel(const T *input, T *output, size_t outer_size,
                                     size_t inner_size, size_t axis_size, size_t start,
                                     size_t length, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  size_t i = idx % inner_size;
  size_t tmp = idx / inner_size;
  size_t l = tmp % length;
  size_t o = tmp / length;

  size_t input_idx = o * axis_size * inner_size + (start + l) * inner_size + i;
  output[idx] = input[input_idx];
}

template <typename T>
void slice_forward(const T *input, T *output, const std::vector<size_t> &input_shape, size_t axis,
                   size_t start, size_t length, cudaStream_t stream) {
  size_t outer_size = 1;
  for (size_t i = 0; i < axis; ++i) {
    outer_size *= input_shape[i];
  }

  size_t inner_size = 1;
  for (size_t i = axis + 1; i < input_shape.size(); ++i) {
    inner_size *= input_shape[i];
  }

  size_t axis_size = input_shape[axis];
  size_t total_elements = outer_size * length * inner_size;

  if (total_elements == 0)
    return;

  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  slice_forward_kernel<T><<<blocks, threads, 0, stream>>>(input, output, outer_size, inner_size,
                                                          axis_size, start, length, total_elements);
}

template <typename T>
__global__ void slice_backward_kernel(const T *gradient, T *grad_input, size_t outer_size,
                                      size_t inner_size, size_t axis_size, size_t start,
                                      size_t length, size_t total_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  size_t i = idx % inner_size;
  size_t tmp = idx / inner_size;
  size_t l = tmp % length;
  size_t o = tmp / length;

  size_t output_idx = o * axis_size * inner_size + (start + l) * inner_size + i;
  grad_input[output_idx] = gradient[idx];
}

template <typename T>
void slice_backward(const T *gradient, T *grad_input, const std::vector<size_t> &input_shape,
                    size_t axis, size_t start, size_t length, cudaStream_t stream) {

  size_t full_size = 1;
  for (size_t s : input_shape)
    full_size *= s;

  cudaMemsetAsync(grad_input, 0, full_size * sizeof(T), stream);

  size_t outer_size = 1;
  for (size_t i = 0; i < axis; ++i) {
    outer_size *= input_shape[i];
  }

  size_t inner_size = 1;
  for (size_t i = axis + 1; i < input_shape.size(); ++i) {
    inner_size *= input_shape[i];
  }

  size_t axis_size = input_shape[axis];
  size_t total_elements = outer_size * length * inner_size;

  if (total_elements == 0)
    return;

  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  slice_backward_kernel<T><<<blocks, threads, 0, stream>>>(
      gradient, grad_input, outer_size, inner_size, axis_size, start, length, total_elements);
}

template void slice_forward<float>(const float *input, float *output,
                                   const std::vector<size_t> &input_shape, size_t axis,
                                   size_t start, size_t length, cudaStream_t stream);
template void slice_forward<double>(const double *input, double *output,
                                    const std::vector<size_t> &input_shape, size_t axis,
                                    size_t start, size_t length, cudaStream_t stream);

template void slice_backward<float>(const float *gradient, float *grad_input,
                                    const std::vector<size_t> &input_shape, size_t axis,
                                    size_t start, size_t length, cudaStream_t stream);
template void slice_backward<double>(const double *gradient, double *grad_input,
                                     const std::vector<size_t> &input_shape, size_t axis,
                                     size_t start, size_t length, cudaStream_t stream);

} // namespace slice
} // namespace cuda
} // namespace tnn

#endif
