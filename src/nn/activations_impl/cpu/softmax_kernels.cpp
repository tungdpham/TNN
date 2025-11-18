#include "nn/activations_impl/cpu/softmax_kernels.hpp"

#include "threading/thread_handler.hpp"
#include <cmath>

namespace tnn {
namespace cpu {
template <typename T>
void softmax(const T *input, T *output, size_t batch_size, size_t channels, size_t height,
             size_t width) {
  const size_t spatial_size = height * width;
  const size_t channel_stride = spatial_size;
  const size_t batch_stride = channels * channel_stride;

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        const size_t spatial_idx = h * width + w;

        // Find max value for numerical stability
        T max_val = input[n * batch_stride + spatial_idx];
        for (size_t c = 1; c < channels; ++c) {
          const size_t idx = n * batch_stride + c * channel_stride + spatial_idx;
          T val = input[idx];
          if (val > max_val) {
            max_val = val;
          }
        }

        // Compute exp and sum
        T sum_exp = T(0);
        for (size_t c = 0; c < channels; ++c) {
          const size_t idx = n * batch_stride + c * channel_stride + spatial_idx;
          T exp_val = std::exp(input[idx] - max_val);
          output[idx] = exp_val;
          sum_exp += exp_val;
        }

        // Normalize
        for (size_t c = 0; c < channels; ++c) {
          const size_t idx = n * batch_stride + c * channel_stride + spatial_idx;
          output[idx] /= sum_exp;
        }
      }
    }
  });
}

template <typename T>
void softmax_gradient(const T *input, T *grad_output, size_t batch_size, size_t channels,
                      size_t height, size_t width) {
  const size_t spatial_size = height * width;
  const size_t channel_stride = spatial_size;
  const size_t batch_stride = channels * channel_stride;

  // First compute softmax values from input
  T *softmax_values = new T[batch_size * channels * spatial_size];
  softmax(input, softmax_values, batch_size, channels, height, width);

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        const size_t spatial_idx = h * width + w;

        // Compute dot product
        T dot_product = T(0);
        for (size_t j = 0; j < channels; ++j) {
          const size_t idx = n * batch_stride + j * channel_stride + spatial_idx;
          dot_product += softmax_values[idx] * grad_output[idx];
        }

        // Update gradient
        for (size_t i = 0; i < channels; ++i) {
          const size_t idx = n * batch_stride + i * channel_stride + spatial_idx;
          T s_i = softmax_values[idx];
          T upstream_i = grad_output[idx];
          grad_output[idx] = s_i * (upstream_i - dot_product);
        }
      }
    }
  });

  delete[] softmax_values;
}

template void softmax<float>(const float *input, float *output, size_t batch_size, size_t channels,
                             size_t height, size_t width);
template void softmax<double>(const double *input, double *output, size_t batch_size,
                              size_t channels, size_t height, size_t width);

template void softmax_gradient<float>(const float *input, float *grad_output, size_t batch_size,
                                      size_t channels, size_t height, size_t width);
template void softmax_gradient<double>(const double *input, double *grad_output, size_t batch_size,
                                       size_t channels, size_t height, size_t width);

} // namespace cpu
} // namespace tnn
