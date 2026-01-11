#include "nn/layers_impl/cpu/layer_norm_ops.hpp"
#include <cmath>
#include <vector>

namespace tnn {
namespace cpu {
namespace layer_norm {
template <typename T>
void layer_norm_forward(const T *input, T *output, const T *gamma, const T *beta, size_t batch_size,
                        size_t channels, T epsilon) {

  size_t batch_stride = channels;

  for (size_t n = 0; n < batch_size; ++n) {
    T sum = 0;
    T sq_sum = 0;
    size_t base_idx = n * batch_stride;
    for (size_t c = 0; c < channels; ++c) {
      T val = input[base_idx + c];
      sum += val;
    }
    T mean = sum / channels;

    for (size_t c = 0; c < channels; ++c) {
      T val = input[base_idx + c];
      sq_sum += (val - mean) * (val - mean);
    }
    T var = sq_sum / channels;
    T inv_std = 1.0 / std::sqrt(var + epsilon);

    for (size_t c = 0; c < channels; ++c) {
      size_t idx = base_idx + c;
      T val = input[idx];
      T normalized = (val - mean) * inv_std;
      T g = gamma ? gamma[c] : T(1);
      T b = beta ? beta[c] : T(0);
      output[idx] = normalized * g + b;
    }
  }
}

template <typename T>
void layer_norm_backward(const T *grad_output, const T *input, const T *gamma, T *grad_input,
                         T *grad_gamma, T *grad_beta, size_t batch_size, size_t channels,
                         T epsilon) {

  size_t batch_stride = channels;

  for (size_t n = 0; n < batch_size; ++n) {
    size_t base_idx = n * batch_stride;

    T sum = 0;
    T sq_sum = 0;
    for (size_t c = 0; c < channels; ++c) {
      T val = input[base_idx + c];
      sum += val;
    }
    T mean = sum / channels;
    for (size_t c = 0; c < channels; ++c) {
      T val = input[base_idx + c];
      sq_sum += (val - mean) * (val - mean);
    }
    T var = sq_sum / channels;
    T inv_std = 1.0 / std::sqrt(var + epsilon);

    T sum_grad_normalized = 0;
    T sum_grad_gamma_normalized = 0;

    for (size_t c = 0; c < channels; ++c) {
      size_t idx = base_idx + c;
      T go = grad_output[idx];
      T val = input[idx];
      T normalized = (val - mean) * inv_std;
      T g = gamma ? gamma[c] : T(1);

      if (grad_gamma)
        grad_gamma[c] += go * normalized;
      if (grad_beta)
        grad_beta[c] += go;

      T dx_hat = go * g;
      sum_grad_normalized += dx_hat * normalized;
      sum_grad_gamma_normalized += dx_hat;
    }

    T factor = inv_std / channels;
    for (size_t c = 0; c < channels; ++c) {
      size_t idx = base_idx + c;
      T val = input[idx];
      T normalized = (val - mean) * inv_std;
      T g = gamma ? gamma[c] : T(1);
      T go = grad_output[idx];
      T dx_hat = go * g;

      grad_input[idx] = factor * (channels * dx_hat - sum_grad_gamma_normalized -
                                  normalized * sum_grad_normalized);
    }
  }
}

template void layer_norm_forward<float>(const float *input, float *output, const float *gamma,
                                        const float *beta, size_t batch_size, size_t channels,
                                        float epsilon);
template void layer_norm_forward<double>(const double *input, double *output, const double *gamma,
                                         const double *beta, size_t batch_size, size_t channels,
                                         double epsilon);

template void layer_norm_backward<float>(const float *grad_output, const float *input,
                                         const float *gamma, float *grad_input, float *grad_gamma,
                                         float *grad_beta, size_t batch_size, size_t channels,
                                         float epsilon);
template void layer_norm_backward<double>(const double *grad_output, const double *input,
                                          const double *gamma, double *grad_input,
                                          double *grad_gamma, double *grad_beta, size_t batch_size,
                                          size_t channels, double epsilon);

} // namespace layer_norm
} // namespace cpu
} // namespace tnn
