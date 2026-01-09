/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "tensor/cuda/tensor_ops.hpp"
#include "tensor/cuda/tensor_kernels.hpp"
#include <cstddef>

#ifdef USE_CUDA
#include "cuda/error_handler.hpp"
#include "ops/cuda/kernels.hpp"

// Public API Implementation
namespace tnn {
namespace cuda {

template <typename T>
void im2col(const Tensor<T> &input_tensor, T *col_data, size_t kernel_h, size_t kernel_w,
            size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, cudaStream_t stream) {
  std::vector<size_t> shape = input_tensor.shape();
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

  const T *input_data = input_tensor.data();

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
void pad(const Tensor<T> &input, Tensor<T> &result, size_t pad_h, size_t pad_w, T value,
         cudaStream_t stream) {
  std::vector<size_t> shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("pad: Input tensor must be 4-dimensional (NCHW)");
  }
  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  const T *input_data = input.data_ptr().get();
  T *result_data = result.data_ptr().get();

  cuda_pad(input_data, result_data, batch_size, channels, height, width, pad_h, pad_w, value,
           stream);
}

template <typename T>
void unpad(const Tensor<T> &input, Tensor<T> &result, size_t pad_h, size_t pad_w,
           cudaStream_t stream) {
  std::vector<size_t> shape = input.shape();
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

  const T *input_data = input.data_ptr().get();
  T *result_data = result.data_ptr().get();

  cuda_unpad(input_data, result_data, batch_size, channels, height, width, pad_h, pad_w, stream);
}

template <typename T>
void crop(const Tensor<T> &input, Tensor<T> &result, const size_t start_h, const size_t start_w,
          const size_t end_h, const size_t end_w, cudaStream_t stream) {
  std::vector<size_t> shape = input.shape();
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

  const T *input_data = input.data_ptr().get();
  T *result_data = result.data_ptr().get();

  cuda_crop(input_data, result_data, batch_size, channels, height, width, start_h, start_w,
            new_height, new_width, stream);
}

template <typename T>
void slice_batch(const Tensor<T> &input, Tensor<T> &result, size_t start_batch, size_t end_batch,
                 cudaStream_t stream) {
  std::vector<size_t> shape = input.shape();

  size_t batch_size = shape[0];

  if (end_batch > batch_size || start_batch > end_batch) {
    throw std::invalid_argument("Invalid batch slice range");
  }

  size_t new_batch_size = end_batch - start_batch;
  std::vector<size_t> new_shape = shape;
  new_shape[0] = new_batch_size;
  size_t batch_stride = input.stride(0);
  result.resize(new_shape, input.device());
  const T *input_data = input.data_ptr().get();
  T *result_data = result.data_ptr().get();

  size_t copy_size = (end_batch - start_batch) * batch_stride;
  cuda::cuda_copy(&input_data[start_batch * batch_stride], result_data, copy_size, stream);
}

template <typename T>
void slice_channels(const Tensor<T> &input, Tensor<T> &result, size_t start_ch, size_t end_ch,
                    cudaStream_t stream) {
  std::vector<size_t> shape = input.shape();

  if (shape.size() != 4) {
    throw std::invalid_argument("slice_channels: Input tensor must be 4-dimensional (NCHW)");
  }

  const size_t batch_size = shape[0];
  const size_t channels = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  if (end_ch >= channels || start_ch > end_ch) {
    throw std::invalid_argument("Invalid channel slice range");
  }

  const T *input_data = input.data_ptr().get();
  T *result_data = result.data_ptr().get();

  size_t channel_size = height * width;
  for (size_t n = 0; n < batch_size; ++n) {
    size_t src_offset = n * channels * channel_size + start_ch * channel_size;
    size_t dst_offset = n * result.shape()[1] * channel_size;
    size_t copy_size = result.shape()[1] * channel_size;

    cuda_copy(&input_data[src_offset], &result_data[dst_offset], copy_size, stream);
  }
}

template <typename T>
void split(const Tensor<T> &input, std::vector<Tensor<T>> &results, size_t num_splits,
           cudaStream_t stream) {
  std::vector<size_t> shape = input.shape();
  size_t batch_size = shape[0];

  if (num_splits == 0 || num_splits > batch_size) {
    throw std::invalid_argument("Invalid number of splits");
  }

  results.resize(num_splits);
  size_t split_size = batch_size / num_splits;

  for (size_t i = 0; i < num_splits; ++i) {
    size_t start = i * split_size;
    size_t end = (i == num_splits - 1) ? batch_size : start + split_size;
    slice_batch(input, results[i], start, end, stream);
  }
}

template <typename T> void apply_softmax(Tensor<T> &input, cudaStream_t stream) {
  auto shape = input.shape();
  if (shape.size() != 4) {
    throw std::invalid_argument("apply_softmax: Input tensor must be 4-dimensional (NCHW)");
  }

  const size_t batch_size = shape[0];
  const size_t num_classes = shape[1];
  const size_t height = shape[2];
  const size_t width = shape[3];

  T *data = input.data_ptr().get();

  cuda_softmax(data, batch_size, num_classes, height, width, stream);
}

// Explicit template instantiations for float
template void im2col<float>(const Tensor<float> &, float *, size_t, size_t, size_t, size_t, size_t,
                            size_t, cudaStream_t);
template void col2im_padded<float>(const float *, float *, size_t, size_t, size_t, size_t, size_t,
                                   size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void col2im<float>(const float *, float *, size_t, size_t, size_t, size_t, size_t, size_t,
                            size_t, size_t, size_t, size_t, cudaStream_t);

template void pad<float>(const Tensor<float> &, Tensor<float> &, size_t, size_t, float,
                         cudaStream_t);

template void unpad<float>(const Tensor<float> &, Tensor<float> &, size_t, size_t, cudaStream_t);

template void crop<float>(const Tensor<float> &, Tensor<float> &, const size_t, const size_t,
                          const size_t, const size_t, cudaStream_t);

template void slice_batch<float>(const Tensor<float> &, Tensor<float> &, size_t, size_t,
                                 cudaStream_t);

template void slice_channels<float>(const Tensor<float> &, Tensor<float> &, size_t, size_t,
                                    cudaStream_t);

template void split<float>(const Tensor<float> &, std::vector<Tensor<float>> &, size_t,
                           cudaStream_t);

template void apply_softmax<float>(Tensor<float> &, cudaStream_t);

// Explicit template instantiations for double
template void im2col<double>(const Tensor<double> &, double *, size_t, size_t, size_t, size_t,
                             size_t, size_t, cudaStream_t);
template void col2im_padded<double>(const double *, double *, size_t, size_t, size_t, size_t,
                                    size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void col2im<double>(const double *, double *, size_t, size_t, size_t, size_t, size_t,
                             size_t, size_t, size_t, size_t, size_t, cudaStream_t);

template void pad<double>(const Tensor<double> &, Tensor<double> &, size_t, size_t, double,
                          cudaStream_t);

template void unpad<double>(const Tensor<double> &, Tensor<double> &, size_t, size_t, cudaStream_t);

template void crop<double>(const Tensor<double> &, Tensor<double> &, const size_t, const size_t,
                           const size_t, const size_t, cudaStream_t);

template void slice_batch<double>(const Tensor<double> &, Tensor<double> &, size_t, size_t,
                                  cudaStream_t);

template void slice_channels<double>(const Tensor<double> &, Tensor<double> &, size_t, size_t,
                                     cudaStream_t);

template void split<double>(const Tensor<double> &, std::vector<Tensor<double>> &, size_t,
                            cudaStream_t);

template void apply_softmax<double>(Tensor<double> &, cudaStream_t);

} // namespace cuda

} // namespace tnn

#endif // USE_CUDA
