#include "nn/activations_impl/cpu/softmax_kernels.hpp"

#include <cmath>

#include "threading/thread_handler.hpp"
#include "type/type.hpp"

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

        T max_val = input[n * batch_stride + spatial_idx];
        for (size_t c = 1; c < channels; ++c) {
          const size_t idx = n * batch_stride + c * channel_stride + spatial_idx;
          T val = input[idx];
          if (val > max_val) {
            max_val = val;
          }
        }

        T sum_exp = T(0);
        for (size_t c = 0; c < channels; ++c) {
          const size_t idx = n * batch_stride + c * channel_stride + spatial_idx;
          T exp_val = exp(input[idx] - max_val);
          output[idx] = exp_val;
          sum_exp += exp_val;
        }

        for (size_t c = 0; c < channels; ++c) {
          const size_t idx = n * batch_stride + c * channel_stride + spatial_idx;
          output[idx] /= sum_exp;
        }
      }
    }
  });
}

template <typename T>
void softmax_gradient(const T *input, const T *grad_output, T *grad_input, size_t batch_size,
                      size_t channels, size_t height, size_t width) {
  const size_t spatial_size = height * width;
  const size_t channel_stride = spatial_size;
  const size_t batch_stride = channels * channel_stride;

  T *softmax_values = new T[batch_size * channels * spatial_size];
  softmax(input, softmax_values, batch_size, channels, height, width);

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        const size_t spatial_idx = h * width + w;

        T dot_product = T(0);
        for (size_t j = 0; j < channels; ++j) {
          const size_t idx = n * batch_stride + j * channel_stride + spatial_idx;
          dot_product += softmax_values[idx] * grad_output[idx];
        }

        for (size_t i = 0; i < channels; ++i) {
          const size_t idx = n * batch_stride + i * channel_stride + spatial_idx;
          T s_i = softmax_values[idx];
          T upstream_i = grad_output[idx];
          grad_input[idx] = s_i * (upstream_i - dot_product);
        }
      }
    }
  });

  delete[] softmax_values;
}

#define INSTANTIATE_SOFTMAX(T)                                                            \
  template void softmax<T>(const T *input, T *output, size_t batch_size, size_t channels, \
                           size_t height, size_t width);                                  \
  template void softmax_gradient<T>(const T *input, const T *grad_output, T *grad_input,  \
                                    size_t batch_size, size_t channels, size_t height,    \
                                    size_t width);
INSTANTIATE_SOFTMAX(fp16)
INSTANTIATE_SOFTMAX(bf16)
INSTANTIATE_SOFTMAX(float)
INSTANTIATE_SOFTMAX(double)
#undef INSTANTIATE_SOFTMAX

}  // namespace cpu
}  // namespace tnn
