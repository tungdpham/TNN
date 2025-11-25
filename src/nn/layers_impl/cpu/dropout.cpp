#include "nn/layers_impl/cpu/dropout_ops.hpp"

#include "threading/thread_handler.hpp"
#include <random>

namespace tnn {
namespace cpu {
template <typename T>
void compute_dropout_forward(const T *input_data, T *output_data, T *mask_data, size_t batch_size,
                             size_t channels, size_t spatial_size, T dropout_rate) {
  T scale = T(1) / (T(1) - dropout_rate);
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    const T *input_channel = input_data + (n * channels + c) * spatial_size;
    T *mask_channel = mask_data + (n * channels + c) * spatial_size;
    T *output_channel = output_data + (n * channels + c) * spatial_size;
    thread_local std::mt19937 local_generator(std::random_device{}());
    thread_local std::uniform_real_distribution<T> local_distribution(T(0), T(1));
    for (size_t i = 0; i < spatial_size; ++i) {
      if (local_distribution(local_generator) < dropout_rate) {
        mask_channel[i] = T(0);
        output_channel[i] = T(0);
      } else {
        mask_channel[i] = scale;
        output_channel[i] = input_channel[i] * scale;
      }
    }
  });
}

template void compute_dropout_forward<float>(const float *input_data, float *output_data,
                                             float *mask_data, size_t batch_size, size_t channels,
                                             size_t spatial_size, float dropout_rate);
template void compute_dropout_forward<double>(const double *input_data, double *output_data,
                                              double *mask_data, size_t batch_size, size_t channels,
                                              size_t spatial_size, double dropout_rate);

} // namespace cpu
} // namespace tnn