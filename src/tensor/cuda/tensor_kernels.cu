#include "cuda/error_handler.hpp"
#include "tensor/cuda/tensor_kernels.hpp"

namespace tnn {
namespace cuda {

constexpr int BLOCK_SIZE = 256;
constexpr int BLOCK_SIZE_2D = 16;

inline int get_num_blocks(size_t size) { return (size + BLOCK_SIZE - 1) / BLOCK_SIZE; }

inline dim3 get_2d_blocks(size_t height, size_t width) {
  return dim3((width + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
              (height + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D);
}

template <typename T>
__global__ void cuda_im2col_kernel(const T *input, T *col_data, size_t batch_size, size_t channels,
                                   size_t height, size_t width, size_t kernel_h, size_t kernel_w,
                                   size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                   size_t output_h, size_t output_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t col_height = channels * kernel_h * kernel_w;
  size_t total_elements = batch_size * col_height * output_h * output_w;

  if (idx < total_elements) {
    size_t w_out = idx % output_w;
    size_t temp = idx / output_w;
    size_t h_out = temp % output_h;
    temp = temp / output_h;
    size_t c_kh_kw = temp % col_height;
    size_t n = temp / col_height;

    size_t kw = c_kh_kw % kernel_w;
    size_t temp2 = c_kh_kw / kernel_w;
    size_t kh = temp2 % kernel_h;
    size_t c = temp2 / kernel_h;

    int h_in = (int)h_out * (int)stride_h - (int)pad_h + (int)kh;
    int w_in = (int)w_out * (int)stride_w - (int)pad_w + (int)kw;

    T value = 0;
    if (h_in >= 0 && (size_t)h_in < height && w_in >= 0 && (size_t)w_in < width) {
      size_t input_idx = ((n * channels + c) * height + (size_t)h_in) * width + (size_t)w_in;
      value = input[input_idx];
    }

    size_t col_idx = c_kh_kw * (batch_size * output_h * output_w) + n * (output_h * output_w) +
                     h_out * output_w + w_out;
    col_data[col_idx] = value;
  }
}

template <typename T>
void cuda_im2col(const T *input, T *col_data, size_t batch_size, size_t channels, size_t height,
                 size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                 size_t pad_h, size_t pad_w, size_t output_h, size_t output_w,
                 cudaStream_t stream) {
  size_t col_height = channels * kernel_h * kernel_w;
  size_t col_width = output_h * output_w;
  size_t total_elements = batch_size * col_height * col_width;
  int num_blocks = get_num_blocks(total_elements);

  cuda_im2col_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      input, col_data, batch_size, channels, height, width, kernel_h, kernel_w, stride_h, stride_w,
      pad_h, pad_w, output_h, output_w);
}

template <typename T>
__global__ void cuda_col2im_kernel(const T *col_data, T *output, size_t batch_size, size_t channels,
                                   size_t height, size_t width, size_t kernel_h, size_t kernel_w,
                                   size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                   size_t output_h, size_t output_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = batch_size * channels * height * width;

  if (idx < total_elements) {

    size_t w_in = idx % width;
    size_t temp = idx / width;
    size_t h_in = temp % height;
    temp = temp / height;
    size_t c = temp % channels;
    size_t n = temp / channels;

    T sum = T(0);
    const size_t spatial_out = output_h * output_w;
    const size_t batch_spatial = batch_size * spatial_out;

    for (size_t kh = 0; kh < kernel_h; ++kh) {
      int h_out_base = (int)h_in + (int)pad_h - (int)kh;

      if (h_out_base < 0 || (h_out_base % (int)stride_h) != 0)
        continue;

      size_t h_out = (size_t)h_out_base / stride_h;
      if (h_out >= output_h)
        continue;

      for (size_t kw = 0; kw < kernel_w; ++kw) {
        int w_out_base = (int)w_in + (int)pad_w - (int)kw;

        if (w_out_base < 0 || (w_out_base % (int)stride_w) != 0)
          continue;

        size_t w_out = (size_t)w_out_base / stride_w;
        if (w_out >= output_w)
          continue;

        size_t c_kh_kw = (c * kernel_h + kh) * kernel_w + kw;
        size_t col_idx = c_kh_kw * batch_spatial + n * spatial_out + h_out * output_w + w_out;
        sum += col_data[col_idx];
      }
    }

    output[idx] = sum;
  }
}

template <typename T>
void cuda_col2im(const T *col_data, T *output, size_t batch_size, size_t channels, size_t height,
                 size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                 size_t pad_h, size_t pad_w, size_t output_h, size_t output_w,
                 cudaStream_t stream) {

  size_t output_size = batch_size * channels * height * width;
  cudaMemsetAsync(output, 0, output_size * sizeof(T), stream);

  size_t col_height = channels * kernel_h * kernel_w;
  size_t total_elements = batch_size * col_height * output_h * output_w;
  int num_blocks = get_num_blocks(total_elements);

  cuda_col2im_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      col_data, output, batch_size, channels, height, width, kernel_h, kernel_w, stride_h, stride_w,
      pad_h, pad_w, output_h, output_w);
}

template <typename T>
__global__ void pad_kernel(const T *input, T *output, size_t batch_size, size_t channels,
                           size_t height, size_t width, size_t pad_h, size_t pad_w, T value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t padded_height = height + 2 * pad_h;
  size_t padded_width = width + 2 * pad_w;
  size_t total_elements = batch_size * channels * padded_height * padded_width;

  if (idx < total_elements) {

    size_t w_pad = idx % padded_width;
    size_t temp = idx / padded_width;
    size_t h_pad = temp % padded_height;
    temp = temp / padded_height;
    size_t c = temp % channels;
    size_t n = temp / channels;

    T result;

    if (h_pad < pad_h || h_pad >= height + pad_h || w_pad < pad_w || w_pad >= width + pad_w) {
      result = value;
    } else {

      size_t h = h_pad - pad_h;
      size_t w = w_pad - pad_w;
      size_t input_idx = ((n * channels + c) * height + h) * width + w;
      result = input[input_idx];
    }

    output[idx] = result;
  }
}

template <typename T>
void cuda_pad(const T *input, T *output, size_t batch_size, size_t channels, size_t height,
              size_t width, size_t pad_h, size_t pad_w, T value, cudaStream_t stream) {
  int num_blocks =
      get_num_blocks(batch_size * channels * (height + 2 * pad_h) * (width + 2 * pad_w));
  pad_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, batch_size, channels, height,
                                                       width, pad_h, pad_w, value);

  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void unpad_kernel(const T *input, T *output, size_t batch_size, size_t channels,
                             size_t height, size_t width, size_t pad_h, size_t pad_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t padded_height = height + 2 * pad_h;
  size_t padded_width = width + 2 * pad_w;
  size_t output_height = height;
  size_t output_width = width;
  size_t total_elements = batch_size * channels * output_height * output_width;

  if (idx < total_elements) {

    size_t w = idx % output_width;
    size_t temp = idx / output_width;
    size_t h = temp % output_height;
    temp = temp / output_height;
    size_t c = temp % channels;
    size_t n = temp / channels;

    size_t h_pad = h + pad_h;
    size_t w_pad = w + pad_w;
    size_t input_idx = ((n * channels + c) * padded_height + h_pad) * padded_width + w_pad;

    output[idx] = input[input_idx];
  }
}

template <typename T>
void cuda_unpad(const T *input, T *output, size_t batch_size, size_t channels, size_t height,
                size_t width, size_t pad_h, size_t pad_w, cudaStream_t stream) {
  int num_blocks = get_num_blocks(batch_size * channels * height * width);
  unpad_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, batch_size, channels,
                                                         height, width, pad_h, pad_w);

  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void crop_kernel(const T *input, T *output, size_t batch_size, size_t channels,
                            size_t height, size_t width, size_t start_h, size_t start_w,
                            size_t new_height, size_t new_width) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t total_elements = batch_size * channels * new_height * new_width;

  if (idx < total_elements) {

    size_t w = idx % new_width;
    size_t temp = idx / new_width;
    size_t h = temp % new_height;
    temp = temp / new_height;
    size_t c = temp % channels;
    size_t n = temp / channels;

    size_t h_in = h + start_h;
    size_t w_in = w + start_w;
    size_t input_idx = ((n * channels + c) * height + h_in) * width + w_in;

    output[idx] = input[input_idx];
  }
}

template <typename T>
void cuda_crop(const T *input, T *output, size_t batch_size, size_t channels, size_t height,
               size_t width, size_t start_h, size_t start_w, size_t new_height, size_t new_width,
               cudaStream_t stream) {
  int num_blocks = get_num_blocks(batch_size * channels * new_height * new_width);
  crop_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      input, output, batch_size, channels, height, width, start_h, start_w, new_height, new_width);

  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void softmax_kernel(T *data, size_t batch_size, size_t num_classes, size_t height,
                               size_t width) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t spatial_size = height * width;
  size_t total_spatial = batch_size * spatial_size;

  if (idx < total_spatial) {
    size_t w = idx % width;
    size_t temp = idx / width;
    size_t h = temp % height;
    size_t n = temp / height;

    size_t base_idx = (n * num_classes * height + h) * width + w;

    T max_val = data[base_idx];
    for (size_t c = 1; c < num_classes; ++c) {
      size_t offset = (c * height) * width;
      T val = data[base_idx + offset];
      if (val > max_val) {
        max_val = val;
      }
    }

    T sum = 0;
    for (size_t c = 0; c < num_classes; ++c) {
      size_t offset = (c * height) * width;
      size_t pos = base_idx + offset;
      T val = exp(data[pos] - max_val);
      data[pos] = val;
      sum += val;
    }

    for (size_t c = 0; c < num_classes; ++c) {
      size_t offset = (c * height) * width;
      size_t pos = base_idx + offset;
      data[pos] /= sum;
    }
  }
}

template <typename T>
void cuda_softmax(T *data, size_t batch_size, size_t num_classes, size_t height, size_t width,
                  cudaStream_t stream) {
  int num_blocks = get_num_blocks(batch_size * height * width);
  softmax_kernel<T>
      <<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, batch_size, num_classes, height, width);

  CUDA_CHECK(cudaGetLastError());
}

template void cuda_im2col<float>(const float *input, float *col_data, size_t batch_size,
                                 size_t channels, size_t height, size_t width, size_t kernel_h,
                                 size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                 size_t pad_w, size_t output_h, size_t output_w, cudaStream_t);

template void cuda_im2col<double>(const double *input, double *col_data, size_t batch_size,
                                  size_t channels, size_t height, size_t width, size_t kernel_h,
                                  size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                  size_t pad_w, size_t output_h, size_t output_w, cudaStream_t);

template void cuda_col2im<float>(const float *col_data, float *output, size_t batch_size,
                                 size_t channels, size_t height, size_t width, size_t kernel_h,
                                 size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                 size_t pad_w, size_t output_h, size_t output_w, cudaStream_t);

template void cuda_col2im<double>(const double *col_data, double *output, size_t batch_size,
                                  size_t channels, size_t height, size_t width, size_t kernel_h,
                                  size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                  size_t pad_w, size_t output_h, size_t output_w, cudaStream_t);

template void cuda_pad<float>(const float *input, float *output, size_t batch_size, size_t channels,
                              size_t height, size_t width, size_t pad_h, size_t pad_w, float value,
                              cudaStream_t);

template void cuda_pad<double>(const double *input, double *output, size_t batch_size,
                               size_t channels, size_t height, size_t width, size_t pad_h,
                               size_t pad_w, double value, cudaStream_t);

template void cuda_unpad<float>(const float *input, float *output, size_t batch_size,
                                size_t channels, size_t height, size_t width, size_t pad_h,
                                size_t pad_w, cudaStream_t);

template void cuda_unpad<double>(const double *input, double *output, size_t batch_size,
                                 size_t channels, size_t height, size_t width, size_t pad_h,
                                 size_t pad_w, cudaStream_t);

template void cuda_crop<float>(const float *input, float *output, size_t batch_size,
                               size_t channels, size_t height, size_t width, size_t start_h,
                               size_t start_w, size_t new_height, size_t new_width, cudaStream_t);

template void cuda_crop<double>(const double *input, double *output, size_t batch_size,
                                size_t channels, size_t height, size_t width, size_t start_h,
                                size_t start_w, size_t new_height, size_t new_width, cudaStream_t);

template void cuda_softmax<float>(float *data, size_t batch_size, size_t num_classes, size_t height,
                                  size_t width, cudaStream_t);

template void cuda_softmax<double>(double *data, size_t batch_size, size_t num_classes,
                                   size_t height, size_t width, cudaStream_t);

} // namespace cuda
} // namespace tnn