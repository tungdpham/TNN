#include "nn/layers_impl/cpu/class_token_ops.hpp"
#include <cstring>

namespace tnn {
namespace cpu {

template <typename T>
void class_token_forward(const T *input, const T *token, T *output, size_t batch_size,
                         size_t channels, size_t spatial_size) {
  size_t L = spatial_size;
  size_t output_spatial = L + 1;

  // Iterate over Batch
  for (size_t n = 0; n < batch_size; ++n) {
    // Iterate over Channels
    for (size_t c = 0; c < channels; ++c) {
      size_t in_offset = n * (channels * L) + c * L;
      size_t out_offset = n * (channels * output_spatial) + c * output_spatial;
      size_t token_offset = c;

      // Copy token to first position
      output[out_offset] = token[token_offset];

      // Copy input spatial data to subsequent positions
      std::memcpy(output + out_offset + 1, input + in_offset, L * sizeof(T));
    }
  }
}

template <typename T>
void class_token_backward(const T *grad_output, T *grad_input, T *grad_token, size_t batch_size,
                          size_t channels, size_t spatial_size) {
  size_t L = spatial_size;
  size_t output_spatial = L + 1;

  // Initialize grad_token to 0 if not done already (usually layers zero out grads before
  // accumulation, but here we are responsible for calculating it from this batch). The framework
  // might expect us to *add* to existing gradients if accumulating batches, but usually backward
  // overwrites or adds. Based on DenseLayer implementation: "weight_gradients_.fill(T(0));" in
  // init, but in backward? DenseLayer uses gemm which adds if beta=1. Let's assume we should
  // accumulate into grad_token (which might be zeroed at start of backward pass by
  // optimizer/trainer). However, ParameterizedLayer doesn't zero gradients automatically. Let's
  // assume we ADD to grad_token.

  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t c = 0; c < channels; ++c) {
      size_t in_offset = n * (channels * L) + c * L;
      size_t out_offset = n * (channels * output_spatial) + c * output_spatial;
      size_t token_offset = c;

      // Gradient for token: sum over batch
      grad_token[token_offset] += grad_output[out_offset];

      // Gradient for input: copy from subsequent positions
      std::memcpy(grad_input + in_offset, grad_output + out_offset + 1, L * sizeof(T));
    }
  }
}

template void class_token_forward<float>(const float *input, const float *token, float *output,
                                         size_t batch_size, size_t channels, size_t spatial_size);
template void class_token_forward<double>(const double *input, const double *token, double *output,
                                          size_t batch_size, size_t channels, size_t spatial_size);

template void class_token_backward<float>(const float *grad_output, float *grad_input,
                                          float *grad_token, size_t batch_size, size_t channels,
                                          size_t spatial_size);
template void class_token_backward<double>(const double *grad_output, double *grad_input,
                                           double *grad_token, size_t batch_size, size_t channels,
                                           size_t spatial_size);

} // namespace cpu
} // namespace tnn
