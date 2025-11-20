#include "nn/activations_impl/cuda/softmax_kernels.hpp"

#ifdef USE_CUDA

namespace tnn {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

__global__ void softmax_kernel(const float *input, float *output, size_t batch_size,
                               size_t channels, size_t height, size_t width) {
  const size_t spatial_size = height * width;
  const size_t channel_stride = spatial_size;
  const size_t batch_stride = channels * channel_stride;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_spatial = batch_size * spatial_size;

  if (idx < total_spatial) {
    size_t n = idx / spatial_size;
    size_t spatial_idx = idx % spatial_size;

    // Find max value for numerical stability
    float max_val = input[n * batch_stride + spatial_idx];
    for (size_t c = 1; c < channels; ++c) {
      size_t data_idx = n * batch_stride + c * channel_stride + spatial_idx;
      float val = input[data_idx];
      if (val > max_val) {
        max_val = val;
      }
    }

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (size_t c = 0; c < channels; ++c) {
      size_t data_idx = n * batch_stride + c * channel_stride + spatial_idx;
      float exp_val = expf(input[data_idx] - max_val);
      output[data_idx] = exp_val;
      sum_exp += exp_val;
    }

    // Normalize
    for (size_t c = 0; c < channels; ++c) {
      size_t data_idx = n * batch_stride + c * channel_stride + spatial_idx;
      output[data_idx] /= sum_exp;
    }
  }
}

__global__ void softmax_gradient_kernel(const float *softmax_values, float *grad_output,
                                        size_t batch_size, size_t channels, size_t height,
                                        size_t width) {
  const size_t spatial_size = height * width;
  const size_t channel_stride = spatial_size;
  const size_t batch_stride = channels * channel_stride;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_spatial = batch_size * spatial_size;

  if (idx < total_spatial) {
    size_t n = idx / spatial_size;
    size_t spatial_idx = idx % spatial_size;

    // Compute dot product
    float dot_product = 0.0f;
    for (size_t j = 0; j < channels; ++j) {
      size_t data_idx = n * batch_stride + j * channel_stride + spatial_idx;
      dot_product += softmax_values[data_idx] * grad_output[data_idx];
    }

    // Update gradient
    for (size_t i = 0; i < channels; ++i) {
      size_t data_idx = n * batch_stride + i * channel_stride + spatial_idx;
      float s_i = softmax_values[data_idx];
      float upstream_i = grad_output[data_idx];
      grad_output[data_idx] = s_i * (upstream_i - dot_product);
    }
  }
}

__global__ void softmax_kernel_double(const double *input, double *output, size_t batch_size,
                                      size_t channels, size_t height, size_t width) {
  const size_t spatial_size = height * width;
  const size_t channel_stride = spatial_size;
  const size_t batch_stride = channels * channel_stride;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_spatial = batch_size * spatial_size;

  if (idx < total_spatial) {
    size_t n = idx / spatial_size;
    size_t spatial_idx = idx % spatial_size;

    // Find max value for numerical stability
    double max_val = input[n * batch_stride + spatial_idx];
    for (size_t c = 1; c < channels; ++c) {
      size_t data_idx = n * batch_stride + c * channel_stride + spatial_idx;
      double val = input[data_idx];
      if (val > max_val) {
        max_val = val;
      }
    }

    // Compute exp and sum
    double sum_exp = 0.0;
    for (size_t c = 0; c < channels; ++c) {
      size_t data_idx = n * batch_stride + c * channel_stride + spatial_idx;
      double exp_val = exp(input[data_idx] - max_val);
      output[data_idx] = exp_val;
      sum_exp += exp_val;
    }

    // Normalize
    for (size_t c = 0; c < channels; ++c) {
      size_t data_idx = n * batch_stride + c * channel_stride + spatial_idx;
      output[data_idx] /= sum_exp;
    }
  }
}

__global__ void softmax_gradient_kernel_double(const double *softmax_values, double *grad_output,
                                               size_t batch_size, size_t channels, size_t height,
                                               size_t width) {
  const size_t spatial_size = height * width;
  const size_t channel_stride = spatial_size;
  const size_t batch_stride = channels * channel_stride;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_spatial = batch_size * spatial_size;

  if (idx < total_spatial) {
    size_t n = idx / spatial_size;
    size_t spatial_idx = idx % spatial_size;

    // Compute dot product
    double dot_product = 0.0;
    for (size_t j = 0; j < channels; ++j) {
      size_t data_idx = n * batch_stride + j * channel_stride + spatial_idx;
      dot_product += softmax_values[data_idx] * grad_output[data_idx];
    }

    // Update gradient
    for (size_t i = 0; i < channels; ++i) {
      size_t data_idx = n * batch_stride + i * channel_stride + spatial_idx;
      double s_i = softmax_values[data_idx];
      double upstream_i = grad_output[data_idx];
      grad_output[data_idx] = s_i * (upstream_i - dot_product);
    }
  }
}

template <>
void softmax<float>(const float *input, float *output, size_t batch_size, size_t channels,
                    size_t height, size_t width, cudaStream_t stream) {
  const size_t total_spatial = batch_size * height * width;
  const int numBlocks = (total_spatial + BLOCK_SIZE - 1) / BLOCK_SIZE;
  softmax_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, batch_size, channels, height,
                                                       width);
}

template <>
void softmax_gradient<float>(const float *input, float *grad_output, size_t batch_size,
                             size_t channels, size_t height, size_t width, cudaStream_t stream) {
  const size_t total_size = batch_size * channels * height * width;
  const size_t total_spatial = batch_size * height * width;

  // Allocate temporary memory for softmax values
  float *softmax_values;
  cudaMallocAsync(&softmax_values, total_size * sizeof(float), stream);

  // Compute softmax first
  const int numBlocks = (total_spatial + BLOCK_SIZE - 1) / BLOCK_SIZE;
  softmax_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, softmax_values, batch_size, channels,
                                                       height, width);

  // Compute gradient
  softmax_gradient_kernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
      softmax_values, grad_output, batch_size, channels, height, width);

  // Free temporary memory
  cudaFreeAsync(softmax_values, stream);
}

template <>
void softmax<double>(const double *input, double *output, size_t batch_size, size_t channels,
                     size_t height, size_t width, cudaStream_t stream) {
  const size_t total_spatial = batch_size * height * width;
  const int numBlocks = (total_spatial + BLOCK_SIZE - 1) / BLOCK_SIZE;
  softmax_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, output, batch_size, channels,
                                                              height, width);
}

template <>
void softmax_gradient<double>(const double *input, double *grad_output, size_t batch_size,
                              size_t channels, size_t height, size_t width, cudaStream_t stream) {
  const size_t total_size = batch_size * channels * height * width;
  const size_t total_spatial = batch_size * height * width;

  // Allocate temporary memory for softmax values
  double *softmax_values;
  cudaMallocAsync(&softmax_values, total_size * sizeof(double), stream);

  // Compute softmax first
  const int numBlocks = (total_spatial + BLOCK_SIZE - 1) / BLOCK_SIZE;
  softmax_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(input, softmax_values, batch_size,
                                                              channels, height, width);

  // Compute gradient
  softmax_gradient_kernel_double<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
      softmax_values, grad_output, batch_size, channels, height, width);

  // Free temporary memory
  cudaFreeAsync(softmax_values, stream);
}

} // namespace cuda
} // namespace tnn

#endif // USE_CUDA
