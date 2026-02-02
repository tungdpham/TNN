/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "tensor/cuda/tensor_ops.hpp"

#include <cstddef>

#include "tensor/cuda/tensor_kernels.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

#ifdef USE_CUDA
#include "cuda/error_handler.hpp"
#include "ops/cuda/kernels.hpp"

// Public API Implementation
namespace tnn {
namespace cuda {

template <typename T>
void im2col(const Tensor &input_tensor, T *col_data, size_t kernel_h, size_t kernel_w,
            size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, cudaStream_t stream) {
  std::vector<size_t> shape = input_tensor->shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("im2col: Input tensor must be 4-dimensional (NCHW)");
  }
  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  const size_t padded_h = height + 2 * pad_h;
  const size_t padded_w = width + 2 * pad_w;
  const size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  const size_t output_w = (padded_w - kernel_w) / stride_w + 1;

  const T *input_data = input_tensor->data_as<T>();

  cuda_im2col(input_data, col_data, batch_size, channels, height, width, kernel_h, kernel_w,
              stride_h, stride_w, pad_h, pad_w, output_h, output_w, stream);
  checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void col2im_padded(const T *col_data, T *result_data, size_t batch_size, size_t channels,
                   size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h,
                   size_t stride_w, size_t pad_h, size_t pad_w, cudaStream_t stream) {
  const size_t padded_h = height + 2 * pad_h;
  const size_t padded_w = width + 2 * pad_w;
  const size_t output_h = (padded_h - kernel_h) / stride_h + 1;
  const size_t output_w = (padded_w - kernel_w) / stride_w + 1;

  cuda_col2im(col_data, result_data, batch_size, channels, height, width, kernel_h, kernel_w,
              stride_h, stride_w, pad_h, pad_w, output_h, output_w, stream);
  checkCudaError(cudaGetLastError(), __func__, __FILE__, __LINE__);
}

template <typename T>
void col2im(const T *col_data, T *result_data, size_t batch_size, size_t channels, size_t height,
            size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w, cudaStream_t stream) {
  col2im_padded(col_data, result_data, batch_size, channels, height, width, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, stream);
}

template <typename T>
void pad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w, T value,
         cudaStream_t stream) {
  std::vector<size_t> shape = input->shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("pad: Input tensor must be 4-dimensional (NCHW)");
  }
  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  const T *input_data = input->data_as<T>();
  T *result_data = result->data_as<T>();

  cuda_pad(input_data, result_data, batch_size, channels, height, width, pad_h, pad_w, value,
           stream);
}

template <typename T>
void unpad(const Tensor &input, Tensor &result, size_t pad_h, size_t pad_w, cudaStream_t stream) {
  std::vector<size_t> shape = input->shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("unpad: Input tensor must be 4-dimensional (NCHW)");
  }

  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t padded_height = shape[2];
  const size_t padded_width = shape[3];

  if (padded_height <= 2 * pad_h || padded_width <= 2 * pad_w) {
    throw std::invalid_argument("Padding size too large for unpadding");
  }

  const size_t height = padded_height - 2 * pad_h;
  const size_t width = padded_width - 2 * pad_w;

  const T *input_data = input->data_as<T>();
  T *result_data = result->data_as<T>();

  cuda_unpad(input_data, result_data, batch_size, channels, height, width, pad_h, pad_w, stream);
}

template <typename T>
void crop(const Tensor &input, Tensor &result, const size_t start_h, const size_t start_w,
          const size_t end_h, const size_t end_w, cudaStream_t stream) {
  std::vector<size_t> shape = input->shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("crop: Input tensor must be 4-dimensional (NCHW)");
  }
  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  if (end_h >= height || end_w >= width || start_h > end_h || start_w > end_w) {
    throw std::invalid_argument("Invalid crop dimensions");
  }

  size_t new_height = end_h - start_h + 1;
  size_t new_width = end_w - start_w + 1;

  const T *input_data = input->data_as<T>();
  T *result_data = result->data_as<T>();

  cuda_crop(input_data, result_data, batch_size, channels, height, width, start_h, start_w,
            new_height, new_width, stream);
}

template <typename T>
void slice_batch(const Tensor &input, Tensor &result, size_t start_batch, size_t end_batch,
                 cudaStream_t stream) {
  std::vector<size_t> shape = input->shape();

  size_t batch_size = shape[0];

  if (end_batch > batch_size || start_batch > end_batch) {
    throw std::invalid_argument("Invalid batch slice range");
  }

  size_t new_batch_size = end_batch - start_batch;
  std::vector<size_t> new_shape = shape;
  new_shape[0] = new_batch_size;
  size_t batch_stride = input->stride(0);
  result = make_tensor(input->data_type(), new_shape, input->device());
  const T *input_data = input->data_as<T>();
  T *result_data = result->data_as<T>();

  size_t copy_size = (end_batch - start_batch) * batch_stride;
  ops::cuda::cuda_copy(&input_data[start_batch * batch_stride], result_data, copy_size, stream);
}

template <typename T>
void split(const Tensor &input, std::vector<Tensor> &results, size_t num_splits,
           cudaStream_t stream) {
  std::vector<size_t> shape = input->shape();
  size_t batch_size = shape[0];

  if (num_splits == 0 || num_splits > batch_size) {
    throw std::invalid_argument("Invalid number of splits");
  }

  results.resize(num_splits);
  size_t split_size = batch_size / num_splits;

  for (size_t i = 0; i < num_splits; ++i) {
    size_t start = i * split_size;
    size_t end = (i == num_splits - 1) ? batch_size : start + split_size;
    slice_batch<T>(input, results[i], start, end, stream);
  }
}

// Explicit template instantiations for float
#define INSTANTIATE_TENSOR_OPS(T)                                                                 \
  template void im2col<T>(const Tensor &, T *, size_t, size_t, size_t, size_t, size_t, size_t,    \
                          cudaStream_t);                                                          \
  template void col2im_padded<T>(const T *, T *, size_t, size_t, size_t, size_t, size_t, size_t,  \
                                 size_t, size_t, size_t, size_t, cudaStream_t);                   \
  template void col2im<T>(const T *, T *, size_t, size_t, size_t, size_t, size_t, size_t, size_t, \
                          size_t, size_t, size_t, cudaStream_t);                                  \
  template void pad<T>(const Tensor &, Tensor &, size_t, size_t, T, cudaStream_t);                \
  template void unpad<T>(const Tensor &, Tensor &, size_t, size_t, cudaStream_t);                 \
  template void crop<T>(const Tensor &, Tensor &, const size_t, const size_t, const size_t,       \
                        const size_t, cudaStream_t);                                              \
  template void slice_batch<T>(const Tensor &, Tensor &, size_t, size_t, cudaStream_t);           \
  template void split<T>(const Tensor &, std::vector<Tensor> &, size_t, cudaStream_t);

INSTANTIATE_TENSOR_OPS(fp16)
INSTANTIATE_TENSOR_OPS(bf16)
INSTANTIATE_TENSOR_OPS(float)
INSTANTIATE_TENSOR_OPS(double)
#undef INSTANTIATE_TENSOR_OPS

}  // namespace cuda

}  // namespace tnn

#endif  // USE_CUDA
