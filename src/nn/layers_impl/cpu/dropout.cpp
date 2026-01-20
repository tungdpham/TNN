#include "nn/layers_impl/cpu/dropout_ops.hpp"

#include "threading/thread_handler.hpp"
#include "type/type.hpp"
#include <algorithm>
#include <random>

namespace tnn {
namespace cpu {
namespace dropout {

constexpr size_t DROPOUT_BLOCK_SIZE = 1024;

template <typename T>
void compute_dropout_forward(const T *input_data, T *output_data, T *mask_data, size_t batch_size,
                             size_t channels, size_t spatial_size, T dropout_rate) {
  T scale = T(1) / (T(1) - dropout_rate);

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t offset = (n * channels + c) * spatial_size;
    const T *input_ptr = input_data + offset;
    T *mask_ptr = mask_data + offset;
    T *output_ptr = output_data + offset;

    thread_local std::mt19937 local_generator(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    T rng_buffer[DROPOUT_BLOCK_SIZE];

    for (size_t i = 0; i < spatial_size; i += DROPOUT_BLOCK_SIZE) {
      size_t current_block_size = std::min(DROPOUT_BLOCK_SIZE, spatial_size - i);

      for (size_t j = 0; j < current_block_size; ++j) {
        rng_buffer[j] = static_cast<T>(dist(local_generator));
      }

      for (size_t j = 0; j < current_block_size; ++j) {
        T r = rng_buffer[j];

        T keep_mask = static_cast<T>(r >= dropout_rate);

        T final_mask = keep_mask * scale;

        mask_ptr[i + j] = final_mask;
        output_ptr[i + j] = input_ptr[i + j] * final_mask;
      }
    }
  });
}

#define INSTANTIATE_DROPOUT(T)                                                                     \
  template void compute_dropout_forward<T>(const T *input_data, T *output_data, T *mask_data,      \
                                           size_t batch_size, size_t channels,                     \
                                           size_t spatial_size, T dropout_rate);
INSTANTIATE_DROPOUT(fp16)
INSTANTIATE_DROPOUT(float)
INSTANTIATE_DROPOUT(double)
#undef INSTANTIATE_DROPOUT

} // namespace dropout
} // namespace cpu
} // namespace tnn