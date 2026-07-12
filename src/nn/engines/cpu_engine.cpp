#include "nn/engines/cpu_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "math/cpu/gemm.hpp"
#include "nn/stats/stats.hpp"
#include "threading/thread_handler.hpp"
#include "type/type.hpp"

namespace tunx {

namespace {

template <typename T>
void dense_fwd_impl(const T* input_data, const T* weight_data, T* output_data, size_t batch_size,
                    size_t input_features, size_t output_features) {
  cpu::gemm<T>(input_data, weight_data, output_data, batch_size, output_features, input_features,
               false, true, T(1.0), T(0.0));
}

template <typename T>
void dense_wgrad_impl(const T* input_data, const T* gradient_data, T* grad_weight_data,
                      size_t batch_size, size_t input_features, size_t output_features) {
  cpu::gemm<T>(gradient_data, input_data, grad_weight_data, output_features, input_features,
               batch_size, true, false, T(1.0), T(1.0));
}

template <typename T>
void dense_dgrad_impl(const T* gradient_data, const T* weight_data, T* grad_input_data,
                      size_t batch_size, size_t input_features, size_t output_features) {
  cpu::gemm<T>(gradient_data, weight_data, grad_input_data, batch_size, input_features,
               output_features, false, false, T(1.0), T(0.0));
}

template <typename T>
void dense_bgrad_impl(const T* current_grad_data, T* grad_bias_data, size_t batch_size,
                      size_t output_features) {
  parallel_for<size_t>(0, output_features, [&](size_t out_f) {
    T grad_sum = T(0);
    for (size_t n = 0; n < batch_size; ++n) {
      grad_sum += current_grad_data[n * output_features + out_f];
    }
    grad_bias_data[out_f] += grad_sum;
  });
}

template <typename T>
void add_bias_impl(T* output_data, const T* bias_data, size_t batch_size, size_t output_features) {
  parallel_for_2d(batch_size, output_features, [&](size_t n, size_t out_f) {
    output_data[n * output_features + out_f] += bias_data[out_f];
  });
}

// Helpers for avgpool NHWC
template <typename T>
void avgpool_fwd_impl(const T* input, T* output, size_t batch_size, size_t height, size_t width,
                      size_t channels, size_t pool_h, size_t pool_w, size_t stride_h,
                      size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                      size_t output_w) {
  parallel_for_2d(batch_size, output_h, [&](size_t b, size_t oh) {
    for (size_t ow = 0; ow < output_w; ++ow) {
      for (size_t c = 0; c < channels; ++c) {
        float sum = 0.0f;
        int count = 0;
        int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
        int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
        int h_end = std::min(h_start + static_cast<int>(pool_h), static_cast<int>(height));
        int w_end = std::min(w_start + static_cast<int>(pool_w), static_cast<int>(width));
        h_start = std::max(h_start, 0);
        w_start = std::max(w_start, 0);
        for (int h = h_start; h < h_end; ++h) {
          for (int w = w_start; w < w_end; ++w) {
            size_t input_idx = ((b * height + h) * width + w) * channels + c;
            sum += static_cast<float>(input[input_idx]);
            ++count;
          }
        }
        size_t output_idx = ((b * output_h + oh) * output_w + ow) * channels + c;
        output[output_idx] = static_cast<T>(count > 0 ? sum / count : 0.0f);
      }
    }
  });
}

template <typename T>
void avgpool_bwd_impl(const T* grad_output, T* grad_input, size_t batch_size, size_t input_h,
                      size_t input_w, size_t channels, size_t pool_h, size_t pool_w,
                      size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                      size_t output_w) {
  parallel_for<size_t>(0, batch_size, [&](size_t b) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        for (size_t c = 0; c < channels; ++c) {
          size_t output_idx = ((b * output_h + oh) * output_w + ow) * channels + c;
          float grad = static_cast<float>(grad_output[output_idx]);
          int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
          int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
          int h_end = std::min(h_start + static_cast<int>(pool_h), static_cast<int>(input_h));
          int w_end = std::min(w_start + static_cast<int>(pool_w), static_cast<int>(input_w));
          h_start = std::max(h_start, 0);
          w_start = std::max(w_start, 0);
          int count = (h_end - h_start) * (w_end - w_start);
          if (count == 0) continue;
          float grad_per_element = grad / count;
          for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
              size_t input_idx = ((b * input_h + h) * input_w + w) * channels + c;
              grad_input[input_idx] =
                  static_cast<T>(static_cast<float>(grad_input[input_idx]) + grad_per_element);
            }
          }
        }
      }
    }
  });
}

template <typename T>
void maxpool2d_fwd_impl(const T* input, T* output, int32* mask, size_t batch_size, size_t input_h,
                        size_t input_w, size_t channels, size_t pool_h, size_t pool_w,
                        size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                        size_t output_h, size_t output_w) {
  parallel_for_2d(batch_size, output_h, [&](size_t b, size_t oh) {
    for (size_t ow = 0; ow < output_w; ++ow) {
      int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
      int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
      int h_end = std::min(h_start + static_cast<int>(pool_h), static_cast<int>(input_h));
      int w_end = std::min(w_start + static_cast<int>(pool_w), static_cast<int>(input_w));
      h_start = std::max(h_start, 0);
      w_start = std::max(w_start, 0);

      size_t out_base_idx = ((b * output_h + oh) * output_w + ow) * channels;

      for (size_t c = 0; c < channels; ++c) {
        output[out_base_idx + c] = std::numeric_limits<T>::lowest();
        if (mask) mask[out_base_idx + c] = -1;
      }

      for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
          size_t in_base_idx = ((b * input_h + h) * input_w + w) * channels;

          for (size_t c = 0; c < channels; ++c) {
            T val = input[in_base_idx + c];
            if (val > output[out_base_idx + c]) {
              output[out_base_idx + c] = val;
              if (mask) {
                int h_start_unclamped = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                int w_start_unclamped = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
                int rel_h = h - h_start_unclamped;
                int rel_w = w - w_start_unclamped;
                mask[out_base_idx + c] = rel_h * static_cast<int>(pool_w) + rel_w;
              }
            }
          }
        }
      }
    }
  });
}

template <typename T>
void maxpool2d_bwd_impl(const T* grad_output, T* grad_input, const int32* mask, size_t batch_size,
                        size_t input_h, size_t input_w, size_t channels, size_t output_h,
                        size_t output_w, size_t pool_w, size_t stride_h, size_t stride_w,
                        size_t pad_h, size_t pad_w) {
  size_t total_elements = batch_size * input_h * input_w * channels;
  std::memset(grad_input, 0, total_elements * sizeof(T));
  parallel_for<size_t>(0, batch_size, [&](size_t b) {
    size_t total_outputs_per_batch = output_h * output_w * channels;
    for (size_t i = 0; i < total_outputs_per_batch; ++i) {
      size_t global_i = b * total_outputs_per_batch + i;
      int rel_idx = mask[global_i];
      if (rel_idx >= 0) {
        size_t c = i % channels;
        size_t ow = (i / channels) % output_w;
        size_t oh = (i / (channels * output_w)) % output_h;

        int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
        int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

        int rel_h = rel_idx / static_cast<int>(pool_w);
        int rel_w = rel_idx % static_cast<int>(pool_w);

        int h = h_start + rel_h;
        int w = w_start + rel_w;

        if (h >= 0 && h < static_cast<int>(input_h) && w >= 0 && w < static_cast<int>(input_w)) {
          size_t in_idx = ((b * input_h + h) * input_w + w) * channels + c;
          // CPU Engine operates single-threaded per element locally, but we need to ensure
          // atomicity if windows overlap. Since CPU parallel_for is over batches, overlapping
          // windows in the same batch are executed sequentially, so no atomics are needed.
          grad_input[in_idx] += grad_output[global_i];
        }
      }
    }
  });
}

// Helpers for class token
template <typename T>
void class_token_fwd_impl(const T* input, const T* token, T* output, size_t batch_size,
                          size_t seq_len, size_t embed_dim) {
  size_t S = seq_len;
  size_t E = embed_dim;
  size_t output_S = S + 1;
  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    T* out_seq = output + n * output_S * E;
    const T* in_seq = input + n * S * E;
    std::memcpy(out_seq, token, E * sizeof(T));
    std::memcpy(out_seq + E, in_seq, S * E * sizeof(T));
  });
}

template <typename T>
void class_token_bwd_impl(const T* grad_output, T* grad_input, T* grad_token, size_t batch_size,
                          size_t seq_len, size_t embed_dim) {
  size_t S = seq_len;
  size_t E = embed_dim;
  size_t output_S = S + 1;
  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    const T* grad_out_seq = grad_output + n * output_S * E;
    T* grad_in_seq = grad_input + n * S * E;
    std::memcpy(grad_in_seq, grad_out_seq + E, S * E * sizeof(T));
  });
  parallel_for<size_t>(0, E, [&](size_t e) {
    T sum = 0;
    for (size_t n = 0; n < batch_size; ++n) {
      const T* grad_out_seq = grad_output + n * output_S * E;
      sum += grad_out_seq[e];
    }
    grad_token[e] += sum;
  });
}

constexpr size_t DROPOUT_BLOCK_SIZE = 1024;

template <typename T>
void dropout_fwd_impl(const T* input_data, T* output_data, bool* mask_data, size_t batch_size,
                      size_t channels, size_t spatial_size, T dropout_rate) {
  T scale = T(1) / (T(1) - dropout_rate);
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t offset = (n * channels + c) * spatial_size;
    const T* input_ptr = input_data + offset;
    bool* mask_ptr = mask_data + offset;
    T* output_ptr = output_data + offset;
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
        mask_ptr[i + j] = static_cast<bool>(keep_mask);
        output_ptr[i + j] = input_ptr[i + j] * final_mask;
      }
    }
  });
}

template <typename T>
void dropout_bwd_impl(const T* grad_output_data, T* grad_input_data, const bool* mask_data,
                      size_t batch_size, size_t channels, size_t spatial_size, T scale) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t offset = (n * channels + c) * spatial_size;
    const T* grad_out_ptr = grad_output_data + offset;
    const bool* mask_ptr = mask_data + offset;
    T* grad_in_ptr = grad_input_data + offset;
    for (size_t i = 0; i < spatial_size; ++i) {
      grad_in_ptr[i] = mask_ptr[i] ? grad_out_ptr[i] * scale : T(0);
    }
  });
}

template <typename T>
void relu_fwd_impl(const T* input_data, T* output_data, bool* mask_data, size_t num_elements) {
  T zero = static_cast<T>(0);
  parallel_for<size_t>(0, num_elements, [&](size_t i) {
    bool is_positive = input_data[i] > zero;
    output_data[i] = is_positive ? input_data[i] : zero;
    if (mask_data) mask_data[i] = is_positive;
  });
}

template <typename T>
void relu_bwd_impl(const T* grad_output_data, T* grad_input_data, const bool* mask_data,
                   size_t num_elements) {
  parallel_for<size_t>(0, num_elements, [&](size_t i) {
    grad_input_data[i] = grad_output_data[i] * static_cast<T>(mask_data[i]);
  });
}

template <typename T_IO, typename T_PARAM>
void embedding_fwd_impl(const T_IO* input_data, const T_PARAM* weight_data, T_PARAM* output_data,
                        size_t num_indices, size_t vocab_size, size_t embed_dim,
                        size_t padding_idx) {
  parallel_for<size_t>(0, num_indices, [&](size_t i) {
    size_t idx = static_cast<size_t>(input_data[i]);
    if (idx >= vocab_size) idx = 0;
    T_PARAM* out_row = output_data + i * embed_dim;
    if (padding_idx < vocab_size && idx == padding_idx) {
      std::fill(out_row, out_row + embed_dim, T_PARAM(0));
    } else {
      const T_PARAM* w_row = weight_data + idx * embed_dim;
      std::memcpy(out_row, w_row, embed_dim * sizeof(T_PARAM));
    }
  });
}

template <typename T_IO, typename T_PARAM>
void embedding_bwd_impl(const T_IO* input_data, const T_PARAM* gradient_data,
                        T_PARAM* grad_weight_data, size_t num_indices, size_t vocab_size,
                        size_t embed_dim, size_t padding_idx) {
  parallel_for<size_t>(0, embed_dim, [&](size_t j) {
    for (size_t i = 0; i < num_indices; ++i) {
      size_t idx = static_cast<size_t>(input_data[i]);
      if (idx >= vocab_size) idx = 0;
      if (padding_idx < vocab_size && idx == padding_idx) continue;
      grad_weight_data[idx * embed_dim + j] += gradient_data[i * embed_dim + j];
    }
  });
}

template <typename T>
void batchnorm_infer_impl(const T* input, const float* running_mean, const float* running_var,
                          const float* gamma, const float* beta, T* output, size_t N, size_t C,
                          size_t S, float epsilon, bool affine, bool use_relu) {
  std::vector<float> scale(C);
  std::vector<float> bias(C);
  for (size_t c = 0; c < C; ++c) {
    float inv_std = 1.0f / std::sqrt(running_var[c] + epsilon);
    float g = affine ? gamma[c] : 1.0f;
    scale[c] = g * inv_std;
    bias[c] = beta[c] - (running_mean[c] * scale[c]);
  }
  size_t M = N * S;
  parallel_for<size_t>(0, M, [&](size_t i) {
    for (size_t c = 0; c < C; ++c) {
      size_t idx = i * C + c;
      float val = static_cast<float>(input[idx]);
      float out_val = val * scale[c] + bias[c];
      if (use_relu) {
        out_val = std::max(out_val, 0.0f);
      }
      output[idx] = static_cast<T>(out_val);
    }
  });
}

template <typename T>
void batchnorm_fwd_impl(const T* input, float* mean, float* inv_std, float* running_mean,
                        float* running_var, const float* gamma, const float* beta, T* output,
                        bool* relu_mask, size_t N, size_t C, size_t S, float momentum,
                        float epsilon, bool affine, bool use_relu) {
  size_t M = N * S;
  float inv_M = 1.0f / static_cast<float>(M);
  std::vector<float> partial_sums(N * C, 0.0f);
  std::vector<float> partial_sq_sums(N * C, 0.0f);
  parallel_for<size_t>(0, N, [&](size_t n) {
    for (size_t s = 0; s < S; ++s) {
      for (size_t c = 0; c < C; ++c) {
        size_t idx = n * S * C + s * C + c;
        float val = static_cast<float>(input[idx]);
        partial_sums[n * C + c] += val;
        partial_sq_sums[n * C + c] += val * val;
      }
    }
  });
  std::vector<float> scale(C);
  std::vector<float> bias_term(C);
  for (size_t c = 0; c < C; ++c) {
    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      sum += partial_sums[n * C + c];
      sq_sum += partial_sq_sums[n * C + c];
    }
    float mu = sum * inv_M;
    mean[c] = mu;
    float var = (sq_sum * inv_M) - (mu * mu);
    var = std::max(var, 0.0f);
    float istd = 1.0f / std::sqrt(var + epsilon);
    inv_std[c] = istd;
    float unbiased_var = (M > 1) ? (var * M) / static_cast<float>(M - 1) : 0.0f;
    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
    float g = affine ? gamma[c] : 1.0f;
    float b = beta[c];
    scale[c] = g * istd;
    bias_term[c] = b - (mu * scale[c]);
  }
  parallel_for<size_t>(0, M, [&](size_t i) {
    for (size_t c = 0; c < C; ++c) {
      size_t idx = i * C + c;
      float val = static_cast<float>(input[idx]) * scale[c] + bias_term[c];
      if (use_relu) {
        bool active = val > 0.0f;
        if (relu_mask) relu_mask[idx] = active;
        output[idx] = val * static_cast<float>(active);
      } else {
        output[idx] = static_cast<T>(val);
      }
    }
  });
}

template <typename T>
void batchnorm_bwd_impl(const T* grad_output, const T* input, const float* mean,
                        const float* inv_std, const float* gamma, float* d_gamma, float* d_beta,
                        T* grad_input, const bool* relu_mask, size_t N, size_t C, size_t S,
                        bool affine, bool use_relu) {
  size_t M = N * S;
  float inv_M = 1.0f / static_cast<float>(M);
  std::vector<float> partial_dy(N * C, 0.0f);
  std::vector<float> partial_dy_xn(N * C, 0.0f);
  parallel_for<size_t>(0, N, [&](size_t n) {
    for (size_t s = 0; s < S; ++s) {
      for (size_t c = 0; c < C; ++c) {
        size_t idx = n * S * C + s * C + c;
        float dy = (!use_relu || (relu_mask && relu_mask[idx]))
                       ? static_cast<float>(grad_output[idx])
                       : 0.0f;
        float x_hat = (static_cast<float>(input[idx]) - mean[c]) * inv_std[c];
        partial_dy[n * C + c] += dy;
        partial_dy_xn[n * C + c] += dy * x_hat;
      }
    }
  });
  std::vector<float> sum_dy(C, 0.0f);
  std::vector<float> sum_dy_xnorm(C, 0.0f);
  for (size_t c = 0; c < C; ++c) {
    float s_dy = 0.0f;
    float s_dy_xn = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      s_dy += partial_dy[n * C + c];
      s_dy_xn += partial_dy_xn[n * C + c];
    }
    sum_dy[c] = s_dy;
    sum_dy_xnorm[c] = s_dy_xn;
    if (affine && d_gamma && d_beta) {
      d_gamma[c] += s_dy_xn;
      d_beta[c] += s_dy;
    }
  }
  parallel_for<size_t>(0, M, [&](size_t i) {
    for (size_t c = 0; c < C; ++c) {
      size_t idx = i * C + c;
      float g = (affine && gamma) ? gamma[c] : 1.0;
      float term1 = (g * inv_std[c]) * inv_M;
      float dy = (!use_relu || (relu_mask && relu_mask[idx])) ? static_cast<float>(grad_output[idx])
                                                              : 0.0f;
      float x_hat = (static_cast<float>(input[idx]) - mean[c]) * inv_std[c];
      float term2 = static_cast<float>(M) * dy - sum_dy[c] - x_hat * sum_dy_xnorm[c];
      grad_input[idx] = static_cast<T>(term1 * term2);
    }
  });
}

template <typename T>
void layernorm_fwd_impl(const T* input, T* output, const T* gamma, const T* beta, size_t batch_size,
                        size_t channels, T epsilon) {
  size_t batch_stride = channels;
  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    T sum = 0;
    T sq_sum = 0;
    size_t base_idx = n * batch_stride;
    for (size_t c = 0; c < channels; ++c) {
      T val = input[base_idx + c];
      sum += val;
    }
    T mean = sum / static_cast<T>(channels);
    for (size_t c = 0; c < channels; ++c) {
      T val = input[base_idx + c];
      sq_sum += (val - mean) * (val - mean);
    }
    T var = sq_sum / static_cast<T>(channels);
    T inv_std = T(1) / static_cast<T>(std::sqrt(static_cast<double>(var + epsilon)));
    for (size_t c = 0; c < channels; ++c) {
      size_t idx = base_idx + c;
      T val = input[idx];
      T normalized = (val - mean) * inv_std;
      T g = gamma ? gamma[c] : T(1);
      T b = beta ? beta[c] : T(0);
      output[idx] = normalized * g + b;
    }
  });
}

template <typename T>
void layernorm_bwd_impl(const T* grad_output, const T* input, const T* gamma, T* grad_input,
                        T* grad_gamma, T* grad_beta, size_t batch_size, size_t channels,
                        T epsilon) {
  size_t batch_stride = channels;
  std::vector<T> means(batch_size);
  std::vector<T> inv_stds(batch_size);

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    size_t base_idx = n * batch_stride;
    T sum = 0;
    for (size_t c = 0; c < channels; ++c) sum += input[base_idx + c];
    T mean = sum / static_cast<T>(channels);
    means[n] = mean;
    T sq_sum = 0;
    for (size_t c = 0; c < channels; ++c) {
      T diff = input[base_idx + c] - mean;
      sq_sum += diff * diff;
    }
    inv_stds[n] =
        T(1) /
        static_cast<T>(std::sqrt(static_cast<double>(sq_sum / static_cast<T>(channels) + epsilon)));
  });

  parallel_for<size_t>(0, batch_size, [&](size_t n) {
    size_t base_idx = n * batch_stride;
    T mean = means[n];
    T inv_std = inv_stds[n];
    T sum_grad_normalized = 0;
    T sum_grad_gamma_normalized = 0;
    for (size_t c = 0; c < channels; ++c) {
      size_t idx = base_idx + c;
      T go = grad_output[idx];
      T val = input[idx];
      T normalized = (val - mean) * inv_std;
      T g = gamma ? gamma[c] : T(1);
      T dx_hat = go * g;
      sum_grad_normalized += dx_hat * normalized;
      sum_grad_gamma_normalized += dx_hat;
    }
    T factor = inv_std / static_cast<T>(channels);
    for (size_t c = 0; c < channels; ++c) {
      size_t idx = base_idx + c;
      T val = input[idx];
      T normalized = (val - mean) * inv_std;
      T g = gamma ? gamma[c] : T(1);
      T go = grad_output[idx];
      T dx_hat = go * g;
      grad_input[idx] = factor * (static_cast<T>(channels) * dx_hat - sum_grad_gamma_normalized -
                                  normalized * sum_grad_normalized);
    }
  });

  if (grad_gamma || grad_beta) {
    parallel_for<size_t>(0, channels, [&](size_t c) {
      T dgamma = 0;
      T dbeta = 0;
      for (size_t n = 0; n < batch_size; ++n) {
        size_t idx = n * batch_stride + c;
        T go = grad_output[idx];
        if (grad_gamma) {
          T val = input[idx];
          T normalized = (val - means[n]) * inv_stds[n];
          dgamma += go * normalized;
        }
        if (grad_beta) dbeta += go;
      }
      if (grad_gamma) grad_gamma[c] += dgamma;
      if (grad_beta) grad_beta[c] += dbeta;
    });
  }
}

// Naive NHWC Conv2D
template <typename T>
void conv2d_fwd_naive_impl(const T* input, const T* weight, const T* bias, T* output,
                           size_t batch_size, size_t input_h, size_t input_w, size_t in_channels,
                           size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h,
                           size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                           size_t output_w) {
  parallel_for_2d(batch_size, output_h, [&](size_t b, size_t oh) {
    for (size_t ow = 0; ow < output_w; ++ow) {
      for (size_t oc = 0; oc < out_channels; ++oc) {
        float sum = bias ? static_cast<float>(bias[oc]) : 0.0f;
        for (size_t kh = 0; kh < kernel_h; ++kh) {
          int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
          if (ih >= 0 && ih < static_cast<int>(input_h)) {
            for (size_t kw = 0; kw < kernel_w; ++kw) {
              int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);
              if (iw >= 0 && iw < static_cast<int>(input_w)) {
                for (size_t ic = 0; ic < in_channels; ++ic) {
                  size_t i_idx = ((b * input_h + ih) * input_w + iw) * in_channels + ic;
                  size_t w_idx = ((oc * kernel_h + kh) * kernel_w + kw) * in_channels + ic;
                  sum += static_cast<float>(input[i_idx]) * static_cast<float>(weight[w_idx]);
                }
              }
            }
          }
        }
        size_t o_idx = ((b * output_h + oh) * output_w + ow) * out_channels + oc;
        output[o_idx] = static_cast<T>(sum);
      }
    }
  });
}

template <typename T>
void conv2d_dgrad_naive_impl(const T* grad_output, const T* weight, T* grad_input,
                             size_t batch_size, size_t input_h, size_t input_w, size_t in_channels,
                             size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h,
                             size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                             size_t output_w) {
  size_t num_elements = batch_size * input_h * input_w * in_channels;
  std::memset(grad_input, 0, num_elements * sizeof(T));
  parallel_for<size_t>(0, batch_size, [&](size_t b) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
          size_t o_idx = ((b * output_h + oh) * output_w + ow) * out_channels + oc;
          float go = static_cast<float>(grad_output[o_idx]);
          for (size_t kh = 0; kh < kernel_h; ++kh) {
            int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
            if (ih >= 0 && ih < static_cast<int>(input_h)) {
              for (size_t kw = 0; kw < kernel_w; ++kw) {
                int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);
                if (iw >= 0 && iw < static_cast<int>(input_w)) {
                  for (size_t ic = 0; ic < in_channels; ++ic) {
                    size_t w_idx = ((oc * kernel_h + kh) * kernel_w + kw) * in_channels + ic;
                    size_t i_idx = ((b * input_h + ih) * input_w + iw) * in_channels + ic;
                    grad_input[i_idx] = static_cast<T>(static_cast<float>(grad_input[i_idx]) +
                                                       go * static_cast<float>(weight[w_idx]));
                  }
                }
              }
            }
          }
        }
      }
    }
  });
}

template <typename T>
void conv2d_wgrad_naive_impl(const T* grad_output, const T* input, T* grad_weight,
                             size_t batch_size, size_t input_h, size_t input_w, size_t in_channels,
                             size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h,
                             size_t stride_w, size_t pad_h, size_t pad_w, size_t output_h,
                             size_t output_w) {
  parallel_for<size_t>(0, out_channels, [&](size_t oc) {
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t oh = 0; oh < output_h; ++oh) {
        for (size_t ow = 0; ow < output_w; ++ow) {
          size_t o_idx = ((b * output_h + oh) * output_w + ow) * out_channels + oc;
          float go = static_cast<float>(grad_output[o_idx]);
          for (size_t kh = 0; kh < kernel_h; ++kh) {
            int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
            if (ih >= 0 && ih < static_cast<int>(input_h)) {
              for (size_t kw = 0; kw < kernel_w; ++kw) {
                int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);
                if (iw >= 0 && iw < static_cast<int>(input_w)) {
                  for (size_t ic = 0; ic < in_channels; ++ic) {
                    size_t w_idx = ((oc * kernel_h + kh) * kernel_w + kw) * in_channels + ic;
                    size_t i_idx = ((b * input_h + ih) * input_w + iw) * in_channels + ic;
                    grad_weight[w_idx] = static_cast<T>(static_cast<float>(grad_weight[w_idx]) +
                                                        go * static_cast<float>(input[i_idx]));
                  }
                }
              }
            }
          }
        }
      }
    }
  });
}

template <typename T>
void conv2d_bgrad_naive_impl(const T* grad_output, T* grad_bias, size_t batch_size,
                             size_t out_channels, size_t output_h, size_t output_w) {
  parallel_for<size_t>(0, out_channels, [&](size_t oc) {
    float sum = 0.0f;
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t oh = 0; oh < output_h; ++oh) {
        for (size_t ow = 0; ow < output_w; ++ow) {
          size_t o_idx = ((b * output_h + oh) * output_w + ow) * out_channels + oc;
          sum += static_cast<float>(grad_output[o_idx]);
        }
      }
    }
    grad_bias[oc] += static_cast<T>(sum);
  });
}

template <typename T>
void legacy_avgpool2d_fwd_impl(const T* input_data, T* output_data, size_t batch_size,
                               size_t channels, size_t input_h, size_t input_w, size_t output_h,
                               size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                               size_t stride_w, size_t pad_h, size_t pad_w) {
  const T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t input_offset = (n * channels + c) * input_h * input_w;
    size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
        long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

        long h_start_valid = std::max(0L, h_start);
        long w_start_valid = std::max(0L, w_start);
        long h_end_valid =
            std::min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
        long w_end_valid =
            std::min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

        T sum = T(0);

        for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
          for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
            sum += input_data[input_offset + ih * input_w + iw];
          }
        }

        size_t output_idx = output_offset + out_h * output_w + out_w;
        output_data[output_idx] = sum * pool_size_inv;
      }
    }
  });
}

template <typename T>
void legacy_avgpool2d_bwd_impl(const T* gradient_data, T* grad_input_data, size_t batch_size,
                               size_t channels, size_t input_h, size_t input_w, size_t output_h,
                               size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                               size_t stride_w, size_t pad_h, size_t pad_w) {
  const T pool_size_inv = T(1.0) / T(pool_h * pool_w);

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t input_offset = (n * channels + c) * input_h * input_w;
    size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        size_t output_idx = output_offset + out_h * output_w + out_w;

        const T grad_val = gradient_data[output_idx] * pool_size_inv;

        long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
        long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

        long h_start_valid = std::max(0L, h_start);
        long w_start_valid = std::max(0L, w_start);
        long h_end_valid =
            std::min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
        long w_end_valid =
            std::min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

        for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
          for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
            grad_input_data[input_offset + ih * input_w + iw] += grad_val;
          }
        }
      }
    }
  });
}

template <typename T>
void legacy_batchnorm_inf_impl(const T* input_data, const float* running_mean_data,
                               const float* running_var_data, const float* gamma_data,
                               const float* beta_data, T* output_data, size_t batch_size,
                               size_t channels, size_t spatial_size, float epsilon, bool affine) {
  size_t channel_stride = channels * spatial_size;

  parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
    float mean_val = running_mean_data[c];
    float var_val = running_var_data[c];
    float std_val = std::sqrt(var_val + epsilon);
    const float inv_std = 1.0f / std_val;

    size_t base_idx = n * channel_stride + c * spatial_size;

    const T* input_ptr = input_data + base_idx;
    T* output_ptr = output_data + base_idx;

    if (affine) {
      const float gamma_val = gamma_data[c];
      const float beta_val = beta_data[c];

      for (size_t i = 0; i < spatial_size; ++i) {
        float normalized_val = (static_cast<float>(input_ptr[i]) - mean_val) * inv_std;
        output_ptr[i] = static_cast<T>(gamma_val * normalized_val + beta_val);
      }
    } else {
      for (size_t i = 0; i < spatial_size; ++i) {
        output_ptr[i] = static_cast<T>((static_cast<float>(input_ptr[i]) - mean_val) * inv_std);
      }
    }
  });
}

template <typename T>
void legacy_batchnorm_fwd_impl(const T* input, float* mean, float* inv_std, float* running_mean,
                               float* running_var, const float* gamma, const float* beta, T* output,
                               float* norm_cache, size_t N, size_t C, size_t S, float momentum,
                               float epsilon, bool affine) {
  size_t total_elements = N * S;
  size_t channel_stride = C * S;
  const float inv_total = 1.0f / static_cast<float>(total_elements);

  parallel_for<size_t>(0, C, [&](size_t c) {
    float sum = 0.0f;
    size_t c_offset = c * S;

    for (size_t n = 0; n < N; ++n) {
      size_t n_offset = n * channel_stride;
      size_t base_idx = n_offset + c_offset;
      const T* input_ptr = input + base_idx;

      for (size_t s = 0; s < S; ++s) {
        sum += static_cast<float>(input_ptr[s]);
      }
    }

    float mu = sum * inv_total;
    mean[c] = mu;

    float var_sum = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      size_t n_offset = n * channel_stride;
      size_t base_idx = n_offset + c_offset;
      const T* input_ptr = input + base_idx;

      for (size_t s = 0; s < S; ++s) {
        float diff = static_cast<float>(input_ptr[s]) - mu;
        var_sum += diff * diff;
      }
    }

    float var = var_sum * inv_total;

    inv_std[c] = 1.0f / std::sqrt(var + epsilon);

    float unbiased_var = var_sum / static_cast<float>(total_elements - 1);

    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
  });

  parallel_for_2d(N, C, [&](size_t n, size_t c) {
    const float mu = mean[c];
    const float istd = inv_std[c];

    size_t n_offset = n * channel_stride;
    size_t c_offset = c * S;
    size_t base_idx = n_offset + c_offset;

    const T* input_ptr = input + base_idx;
    T* output_ptr = output + base_idx;
    float* norm_ptr = norm_cache ? (norm_cache + base_idx) : nullptr;

    for (size_t s = 0; s < S; ++s) {
      float x = static_cast<float>(input_ptr[s]);
      float norm = (x - mu) * istd;

      if (norm_ptr) norm_ptr[s] = norm;

      if (affine) {
        output_ptr[s] = static_cast<T>(norm * gamma[c] + beta[c]);
      } else {
        output_ptr[s] = static_cast<T>(norm);
      }
    }
  });
}

template <typename T>
void legacy_batchnorm_bwd_impl(const T* grad_output, const float* norm_input, const float* inv_std,
                               const float* gamma, float* d_gamma, float* d_beta, T* grad_input,
                               size_t N, size_t C, size_t S, bool affine) {
  size_t channel_stride = C * S;
  size_t M = N * S;
  const float inv_M = 1.0f / static_cast<float>(M);

  parallel_for<size_t>(0, C, [&](size_t c) {
    float sum_dy = 0.0f;
    float sum_dy_x_norm = 0.0f;
    size_t c_offset = c * S;

    for (size_t n = 0; n < N; ++n) {
      size_t n_offset = n * channel_stride;
      size_t base_idx = n_offset + c_offset;

      for (size_t s = 0; s < S; ++s) {
        size_t idx = base_idx + s;
        float dy = static_cast<float>(grad_output[idx]);
        float x_hat = norm_input[idx];

        sum_dy += dy;
        sum_dy_x_norm += dy * x_hat;
      }
    }

    if (affine) {
      d_gamma[c] += sum_dy_x_norm;
      d_beta[c] += sum_dy;
    } else {
      d_gamma[c] = sum_dy_x_norm;
      d_beta[c] = sum_dy;
    }
  });

  parallel_for_2d(N, C, [&](size_t n, size_t c) {
    const float g = (affine && gamma) ? gamma[c] : 1.0f;
    const float istd = inv_std[c];

    const float sum_dy = d_beta[c];
    const float sum_dy_x_norm = d_gamma[c];

    size_t n_offset = n * channel_stride;
    size_t c_offset = c * S;
    size_t base_idx = n_offset + c_offset;

    const float term1 = (g * istd) * inv_M;

    for (size_t s = 0; s < S; ++s) {
      size_t idx = base_idx + s;
      float dy = static_cast<float>(grad_output[idx]);
      float x_hat = norm_input[idx];

      float term2 = static_cast<float>(M) * dy - sum_dy - (x_hat * sum_dy_x_norm);
      grad_input[idx] = static_cast<T>(term1 * term2);
    }
  });
}

template <typename T>
void legacy_maxpool2d_fwd_impl(const T* input_data, T* output_data, size_t batch_size,
                               size_t channels, size_t input_h, size_t input_w, size_t output_h,
                               size_t output_w, size_t pool_h, size_t pool_w, size_t stride_h,
                               size_t stride_w, size_t pad_h, size_t pad_w, size_t* mask_indices) {
  const T MIN_VALUE = std::numeric_limits<T>::lowest();

  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t input_offset = (n * channels + c) * input_h * input_w;
    size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t out_h = 0; out_h < output_h; ++out_h) {
      for (size_t out_w = 0; out_w < output_w; ++out_w) {
        long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
        long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);
        long h_end = h_start + pool_h;
        long w_end = w_start + pool_w;

        long h_start_valid = std::max(0L, h_start);
        long w_start_valid = std::max(0L, w_start);
        long h_end_valid = std::min(static_cast<long>(input_h), h_end);
        long w_end_valid = std::min(static_cast<long>(input_w), w_end);

        T max_val = MIN_VALUE;
        size_t max_idx = 0;

        for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
          for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
            size_t cur_input_idx = input_offset + ih * input_w + iw;
            T val = input_data[cur_input_idx];

            if (val > max_val) {
              max_val = val;
              max_idx = cur_input_idx;
            }
          }
        }

        size_t out_idx = output_offset + out_h * output_w + out_w;
        output_data[out_idx] = max_val;
        mask_indices[out_idx] = max_idx;
      }
    }
  });
}

template <typename T>
void legacy_maxpool2d_bwd_impl(const T* gradient_data, T* grad_input_data, size_t batch_size,
                               size_t channels, size_t output_h, size_t output_w,
                               const size_t* mask_indices) {
  parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    size_t output_offset = (n * channels + c) * output_h * output_w;

    for (size_t i = 0; i < output_h * output_w; ++i) {
      size_t out_idx = output_offset + i;
      size_t input_idx = mask_indices[out_idx];
      grad_input_data[input_idx] += gradient_data[out_idx];
    }
  });
}

}  // namespace

void* CPUEngine::create_backend_handle() {
  return nullptr;  // TODO: make something proper
}

WorkspaceReq CPUEngine::query_dense_graph(void* backend_handle, const DenseStats& stats,
                                          DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_avgpool_graph(void*, const AvgPool2DStats&, DTypeDesc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_maxpool2d_graph(void*, const MaxPool2DStats&, DTypeDesc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_class_token_graph(void* backend_handle, const ClassTokenStats& stats,
                                                DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_dropout_graph(void*, const DropoutStats&, DTypeDesc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_embedding_graph(void* backend_handle, const EmbeddingStats& stats,
                                              DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_relu_graph(void*, const ReLUStats&, DTypeDesc) { return {0, 0, 0}; }

WorkspaceReq CPUEngine::query_batchnorm_graph(void* backend_handle, const BatchNormStats& stats,
                                              DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_conv2d_graph(void* backend_handle, const Conv2DStats& stats,
                                           DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CPUEngine::query_layernorm_graph(void* backend_handle, const LayerNormStats& stats,
                                              DTypeDesc type_desc) {
  return WorkspaceReq{0, 0, 0};
}

WorkspaceReq CPUEngine::query_sdpa_graph(void* backend_handle, const AttentionStats& stats,
                                         DTypeDesc type_desc) {
  return WorkspaceReq{0, 0, 0};
}

void CPUEngine::dense_fwd(void* backend_handle, const DenseStats& stats, const void* input, const void* weight,
                          const void* bias, void* output, void* workspace, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dense_fwd_impl<T>(static_cast<const T*>(input), static_cast<const T*>(weight),
                      static_cast<T*>(output), stats.batch_size, stats.in_features,
                      stats.out_features);
    if (bias) {
      add_bias_impl<T>(static_cast<T*>(output), static_cast<const T*>(bias), stats.batch_size,
                       stats.out_features);
    }
  });
}

void CPUEngine::dense_wgrad(void*, const DenseStats& stats, const void* grad_output,
                            const void* input, void* grad_weight, void* workspace,
                            DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dense_wgrad_impl<T>(static_cast<const T*>(input), static_cast<const T*>(grad_output),
                        static_cast<T*>(grad_weight), stats.batch_size, stats.in_features,
                        stats.out_features);
  });
}

void CPUEngine::dense_dgrad(void*, const DenseStats& stats, const void* grad_output,
                            const void* weight, void* grad_input, void* workspace,
                            DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dense_dgrad_impl<T>(static_cast<const T*>(grad_output), static_cast<const T*>(weight),
                        static_cast<T*>(grad_input), stats.batch_size, stats.in_features,
                        stats.out_features);
  });
}

void CPUEngine::dense_bgrad(void*, const DenseStats& stats, const void* grad_output,
                            void* grad_bias, void* workspace, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dense_bgrad_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_bias),
                        stats.batch_size, stats.out_features);
  });
}

void CPUEngine::avgpool_fwd(void*, const AvgPool2DStats& stats, const void* input, void* output,
                            void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    avgpool_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), stats.batch_size,
                        stats.height, stats.width, stats.channels, stats.pool_h, stats.pool_w,
                        stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w, output_h,
                        output_w);
  });
}

void CPUEngine::avgpool_bwd(void*, const AvgPool2DStats& stats, const void* grad_output,
                            void* grad_input, void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    avgpool_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                        stats.batch_size, stats.height, stats.width, stats.channels, stats.pool_h,
                        stats.pool_w, stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w,
                        output_h, output_w);
  });
}

void CPUEngine::maxpool2d_fwd(void*, const MaxPool2DStats& stats, const void* input, void* output,
                              void* mask, void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    maxpool2d_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output),
                          static_cast<int*>(mask), stats.batch_size, stats.height, stats.width,
                          stats.channels, stats.pool_h, stats.pool_w, stats.stride_h,
                          stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CPUEngine::maxpool2d_infer(void*, const MaxPool2DStats& stats, const void* input, void* output,
                                void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    maxpool2d_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output),
                          static_cast<int32*>(nullptr), stats.batch_size, stats.height, stats.width,
                          stats.channels, stats.pool_h, stats.pool_w, stats.stride_h,
                          stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CPUEngine::maxpool2d_bwd(void*, const MaxPool2DStats& stats, const void* grad_output,
                              void* grad_input, const void* mask, void* workspace,
                              DTypeDesc type_desc) {
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    maxpool2d_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                          static_cast<const int*>(mask), stats.batch_size, stats.height,
                          stats.width, stats.channels, output_h, output_w, stats.pool_w,
                          stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w);
  });
}

void CPUEngine::class_token_fwd(void*, const ClassTokenStats& stats, const void* input,
                                const void* token, void* output, void* workspace,
                                DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    class_token_fwd_impl<T>(static_cast<const T*>(input), static_cast<const T*>(token),
                            static_cast<T*>(output), stats.batch_size, stats.seq_len,
                            stats.embed_dim);
  });
}

void CPUEngine::class_token_bwd(void*, const ClassTokenStats& stats, const void* grad_output,
                                void* grad_input, void* grad_token, void* workspace,
                                DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    class_token_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                            static_cast<T*>(grad_token), stats.batch_size, stats.seq_len,
                            stats.embed_dim);
  });
}

void CPUEngine::dropout_fwd(void*, const DropoutStats& stats, const void* input, void* output,
                            bool* mask, void* workspace, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dropout_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), mask,
                        stats.batch_size, stats.channels, stats.spatial_size,
                        static_cast<T>(stats.dropout_rate));
  });
}

void CPUEngine::dropout_bwd(void*, const DropoutStats& stats, const void* grad_output,
                            void* grad_input, const bool* mask, double scale, void* workspace,
                            DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    dropout_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input), mask,
                        stats.batch_size, stats.channels, stats.spatial_size,
                        static_cast<T>(scale));
  });
}

void CPUEngine::relu_fwd(void*, const ReLUStats& stats, const void* input, void* output, bool* mask,
                         void* workspace, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    relu_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), mask,
                     stats.batch_size * stats.spatial_size);
  });
}

void CPUEngine::relu_bwd(void*, const ReLUStats& stats, const void* grad_output, void* grad_input,
                         const bool* mask, void* workspace, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    relu_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input), mask,
                     stats.batch_size * stats.spatial_size);
  });
}

void CPUEngine::embedding_fwd(void*, const EmbeddingStats& stats, const void* input,
                              const void* weight, void* output, void* workspace,
                              DTypeDesc type_desc) {
  DISPATCH_ANY_DTYPE2(type_desc.io_dtype, type_desc.param_dtype, T_IO, T_PARAM, {
    embedding_fwd_impl<T_IO, T_PARAM>(static_cast<const T_IO*>(input),
                                      static_cast<const T_PARAM*>(weight),
                                      static_cast<T_PARAM*>(output), stats.num_indices,
                                      stats.vocab_size, stats.embed_dim, stats.padding_idx);
  });
}

void CPUEngine::embedding_bwd(void*, const EmbeddingStats& stats, const void* grad_output,
                              const void* input, void* grad_weight, void* workspace,
                              DTypeDesc type_desc) {
  DISPATCH_ANY_DTYPE2(type_desc.io_dtype, type_desc.param_dtype, T_IO, T_PARAM, {
    embedding_bwd_impl<T_IO, T_PARAM>(static_cast<const T_IO*>(input),
                                      static_cast<const T_PARAM*>(grad_output),
                                      static_cast<T_PARAM*>(grad_weight), stats.num_indices,
                                      stats.vocab_size, stats.embed_dim, stats.padding_idx);
  });
}

void CPUEngine::batchnorm_fwd(void*, const BatchNormStats& stats, const void* input,
                              const void* gamma, const void* beta, void* output, void*, void*,
                              void* next_running_mean, void* next_running_var, void* batch_mean,
                              void* batch_invar, void* relu_mask, void* workspace,
                              DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    batchnorm_fwd_impl<T>(static_cast<const T*>(input), static_cast<float*>(batch_mean),
                          static_cast<float*>(batch_invar), static_cast<float*>(next_running_mean),
                          static_cast<float*>(next_running_var), static_cast<const float*>(gamma),
                          static_cast<const float*>(beta), static_cast<T*>(output),
                          static_cast<bool*>(relu_mask), stats.batch_size, stats.channels,
                          stats.height * stats.width, stats.momentum, stats.epsilon,
                          gamma != nullptr, stats.use_relu);
  });
}

void CPUEngine::batchnorm_infer(void*, const BatchNormStats& stats, const void* input,
                                const void* gamma, const void* beta, const void* saved_mean,
                                const void* saved_var, void* output, void* workspace,
                                DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    batchnorm_infer_impl<T>(static_cast<const T*>(input), static_cast<const float*>(saved_mean),
                            static_cast<const float*>(saved_var), static_cast<const float*>(gamma),
                            static_cast<const float*>(beta), static_cast<T*>(output),
                            stats.batch_size, stats.channels, stats.height * stats.width,
                            stats.epsilon, gamma != nullptr, stats.use_relu);
  });
}

void CPUEngine::batchnorm_bwd(void*, const BatchNormStats& stats, const void* grad_output,
                              const void* input, const void* relu_mask, const void* gamma,
                              void* grad_input, void* grad_gamma, void* grad_beta,
                              const void* batch_mean, const void* batch_invar, void* workspace,
                              DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    batchnorm_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<const T*>(input),
                          static_cast<const float*>(batch_mean),
                          static_cast<const float*>(batch_invar), static_cast<const float*>(gamma),
                          static_cast<float*>(grad_gamma), static_cast<float*>(grad_beta),
                          static_cast<T*>(grad_input), static_cast<const bool*>(relu_mask),
                          stats.batch_size, stats.channels, stats.height * stats.width,
                          gamma != nullptr, stats.use_relu);
  });
}

void CPUEngine::conv2d_fwd(void*, const Conv2DStats& stats, const void* input, const void* weight,
                           const void* bias, void* output, void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    conv2d_fwd_naive_impl<T>(static_cast<const T*>(input), static_cast<const T*>(weight),
                             static_cast<const T*>(bias), static_cast<T*>(output), stats.batch_size,
                             stats.input_h, stats.input_w, stats.in_channels, stats.out_channels,
                             stats.kernel_h, stats.kernel_w, stats.stride_h, stats.stride_w,
                             stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CPUEngine::conv2d_dgrad(void*, const Conv2DStats& stats, const void* grad_output,
                             const void* weight, void* grad_input, void* workspace,
                             DTypeDesc type_desc) {
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    conv2d_dgrad_naive_impl<T>(static_cast<const T*>(grad_output), static_cast<const T*>(weight),
                               static_cast<T*>(grad_input), stats.batch_size, stats.input_h,
                               stats.input_w, stats.in_channels, stats.out_channels, stats.kernel_h,
                               stats.kernel_w, stats.stride_h, stats.stride_w, stats.pad_h,
                               stats.pad_w, output_h, output_w);
  });
}

void CPUEngine::conv2d_wgrad(void*, const Conv2DStats& stats, const void* grad_output,
                             const void* input, void* grad_weight, void* workspace,
                             DTypeDesc type_desc) {
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    conv2d_wgrad_naive_impl<T>(static_cast<const T*>(grad_output), static_cast<const T*>(input),
                               static_cast<T*>(grad_weight), stats.batch_size, stats.input_h,
                               stats.input_w, stats.in_channels, stats.out_channels, stats.kernel_h,
                               stats.kernel_w, stats.stride_h, stats.stride_w, stats.pad_h,
                               stats.pad_w, output_h, output_w);
  });
}

void CPUEngine::conv2d_bgrad(void*, const Conv2DStats& stats, const void* grad_output,
                             void* grad_bias, void* workspace, DTypeDesc type_desc) {
  size_t output_h = (stats.input_h + 2 * stats.pad_h - stats.kernel_h) / stats.stride_h + 1;
  size_t output_w = (stats.input_w + 2 * stats.pad_w - stats.kernel_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    conv2d_bgrad_naive_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_bias),
                               stats.batch_size, stats.out_channels, output_h, output_w);
  });
}

void CPUEngine::layernorm_fwd(void*, const LayerNormStats& stats, const void* input,
                              const void* gamma, const void* beta, void* output, void* mean,
                              void* inv_variance, void* workspace, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    layernorm_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output),
                          static_cast<const T*>(gamma), static_cast<const T*>(beta),
                          stats.batch_size, stats.channels, static_cast<T>(stats.epsilon));
  });
}

void CPUEngine::layernorm_infer(void*, const LayerNormStats& stats, const void* input,
                                const void* gamma, const void* beta, void* output, void* workspace,
                                DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    layernorm_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output),
                          static_cast<const T*>(gamma), static_cast<const T*>(beta),
                          stats.batch_size, stats.channels, static_cast<T>(stats.epsilon));
  });
}

void CPUEngine::layernorm_bwd(void*, const LayerNormStats& stats, const void* grad_output,
                              const void* input, const void* gamma, const void* mean,
                              const void* inv_variance, void* grad_input, void* grad_gamma,
                              void* grad_beta, void* workspace, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    layernorm_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<const T*>(input),
                          static_cast<const T*>(gamma), static_cast<T*>(grad_input),
                          static_cast<T*>(grad_gamma), static_cast<T*>(grad_beta), stats.batch_size,
                          stats.channels, static_cast<T>(stats.epsilon));
  });
}

// Legacy APIs

void CPUEngine::legacy_dense_fwd(void*, const void* input, const void* weight, void* output,
                                 size_t batch_size, size_t in_features, size_t out_features,
                                 DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::gemm<T>(static_cast<const T*>(input), static_cast<const T*>(weight),
                 static_cast<T*>(output), batch_size, out_features, in_features, false, true,
                 T(1.0), T(0.0));
  });
}

void CPUEngine::legacy_dense_wgrad(void*, const void* input, const void* grad_output,
                                   void* grad_weight, size_t batch_size, size_t in_features,
                                   size_t out_features, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::gemm<T>(static_cast<const T*>(grad_output), static_cast<const T*>(input),
                 static_cast<T*>(grad_weight), out_features, in_features, batch_size, true, false,
                 T(1.0), T(0.0));
  });
}

void CPUEngine::legacy_dense_dgrad(void*, const void* grad_output, const void* weight,
                                   void* grad_input, size_t batch_size, size_t in_features,
                                   size_t out_features, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::gemm<T>(static_cast<const T*>(grad_output), static_cast<const T*>(weight),
                 static_cast<T*>(grad_input), batch_size, in_features, out_features, false, false,
                 T(1.0), T(0.0));
  });
}

void CPUEngine::legacy_dense_bgrad(void*, const void* grad_output, void* grad_bias,
                                   size_t batch_size, size_t out_features, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    const T* go = static_cast<const T*>(grad_output);
    T* gb = static_cast<T*>(grad_bias);
    for (size_t j = 0; j < out_features; ++j) {
      T sum = 0;
      for (size_t i = 0; i < batch_size; ++i) {
        sum += go[i * out_features + j];
      }
      gb[j] = sum;
    }
  });
}

void CPUEngine::legacy_dense_add_bias(void*, void* output, const void* bias, size_t batch_size,
                                      size_t out_features, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    add_bias_impl<T>(static_cast<T*>(output), static_cast<const T*>(bias), batch_size,
                     out_features);
  });
}

void CPUEngine::legacy_avgpool2d_fwd(void*, const void* input, void* output, size_t batch_size,
                                     size_t channels, size_t input_h, size_t input_w,
                                     size_t output_h, size_t output_w, size_t pool_h, size_t pool_w,
                                     size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                     DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_avgpool2d_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), batch_size,
                                 channels, input_h, input_w, output_h, output_w, pool_h, pool_w,
                                 stride_h, stride_w, pad_h, pad_w);
  });
}

void CPUEngine::legacy_avgpool2d_bwd(void*, const void* grad_output, void* grad_input,
                                     size_t batch_size, size_t channels, size_t input_h,
                                     size_t input_w, size_t output_h, size_t output_w,
                                     size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                                     size_t pad_h, size_t pad_w, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_avgpool2d_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                                 batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
                                 pool_w, stride_h, stride_w, pad_h, pad_w);
  });
}

void CPUEngine::legacy_maxpool2d_fwd(void*, const void* input, void* output, size_t batch_size,
                                     size_t channels, size_t input_h, size_t input_w,
                                     size_t output_h, size_t output_w, size_t pool_h, size_t pool_w,
                                     size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w,
                                     void* mask_indices, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_maxpool2d_fwd_impl<T>(static_cast<const T*>(input), static_cast<T*>(output), batch_size,
                                 channels, input_h, input_w, output_h, output_w, pool_h, pool_w,
                                 stride_h, stride_w, pad_h, pad_w,
                                 static_cast<size_t*>(mask_indices));
  });
}

void CPUEngine::legacy_maxpool2d_bwd(void*, const void* grad_output, void* grad_input,
                                     size_t batch_size, size_t channels, size_t output_h,
                                     size_t output_w, const void* mask_indices,
                                     DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_maxpool2d_bwd_impl<T>(static_cast<const T*>(grad_output), static_cast<T*>(grad_input),
                                 batch_size, channels, output_h, output_w,
                                 static_cast<const size_t*>(mask_indices));
  });
}

void CPUEngine::legacy_batchnorm_fwd(void*, const void* input, void* batch_mean,
                                     void* batch_inv_std, void* running_mean, void* running_var,
                                     const void* gamma, const void* beta, void* output, void* norm,
                                     size_t batch_size, size_t channels, size_t spatial_size,
                                     float momentum, float epsilon, bool affine,
                                     DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_batchnorm_fwd_impl<T>(
        static_cast<const T*>(input), static_cast<float*>(batch_mean),
        static_cast<float*>(batch_inv_std), static_cast<float*>(running_mean),
        static_cast<float*>(running_var), static_cast<const float*>(gamma),
        static_cast<const float*>(beta), static_cast<T*>(output), static_cast<float*>(norm),
        batch_size, channels, spatial_size, momentum, epsilon, affine);
  });
}

void CPUEngine::legacy_batchnorm_infer(void*, const void* input, const void* running_mean,
                                       const void* running_var, const void* gamma, const void* beta,
                                       void* output, size_t batch_size, size_t channels,
                                       size_t spatial_size, float epsilon, bool affine,
                                       DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_batchnorm_inf_impl<T>(
        static_cast<const T*>(input), static_cast<const float*>(running_mean),
        static_cast<const float*>(running_var), static_cast<const float*>(gamma),
        static_cast<const float*>(beta), static_cast<T*>(output), batch_size, channels,
        spatial_size, epsilon, affine);
  });
}

void CPUEngine::legacy_batchnorm_bwd(void*, const void* grad_output, const void* norm_input,
                                     const void* inv_std, const void* gamma, void* d_gamma,
                                     void* d_beta, void* grad_input, size_t batch_size,
                                     size_t channels, size_t spatial_size, bool affine,
                                     DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    legacy_batchnorm_bwd_impl<T>(
        static_cast<const T*>(grad_output), static_cast<const float*>(norm_input),
        static_cast<const float*>(inv_std), static_cast<const float*>(gamma),
        static_cast<float*>(d_gamma), static_cast<float*>(d_beta), static_cast<T*>(grad_input),
        batch_size, channels, spatial_size, affine);
  });
}

void CPUEngine::legacy_conv2d_fwd(void*, const void* col_data, const void* weight_data,
                                  void* output_data, size_t output_size, size_t kernel_size,
                                  size_t out_channels, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::gemm<T>(static_cast<const T*>(weight_data), static_cast<const T*>(col_data),
                 static_cast<T*>(output_data), out_channels, output_size, kernel_size, false, false,
                 T(1.0), T(0.0));
  });
}

void CPUEngine::legacy_conv2d_wgrad(void*, const void* col_data, const void* gradient_data,
                                    void* grad_weight_data, size_t output_size, size_t kernel_size,
                                    size_t out_channels, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::gemm<T>(static_cast<const T*>(gradient_data), static_cast<const T*>(col_data),
                 static_cast<T*>(grad_weight_data), out_channels, kernel_size, output_size, false,
                 true, T(1.0), T(1.0));
  });
}

void CPUEngine::sdpa_fwd(void* backend_handle, const AttentionStats& stats, const void* q_data,
                         const void* k_data, const void* v_data, void* o_data, void* stats_data,
                         void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("SDPA forward is not yet implemented for CPUEngine");
}

void CPUEngine::sdpa_bwd(void* backend_handle, const AttentionStats& stats, const void* q_data,
                         const void* k_data, const void* v_data, const void* o_data,
                         const void* dO_data, const void* stats_data, void* dQ_data, void* dK_data,
                         void* dV_data, void* workspace, DTypeDesc type_desc) {
  throw std::runtime_error("SDPA backward is not yet implemented for CPUEngine");
}

// --- Legacy APIs ---

void CPUEngine::legacy_conv2d_dgrad(void*, const void* gradient_data, const void* weight_data,
                                    void* col_grad_data, size_t output_size, size_t kernel_size,
                                    size_t out_channels, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    cpu::gemm<T>(static_cast<const T*>(weight_data), static_cast<const T*>(gradient_data),
                 static_cast<T*>(col_grad_data), kernel_size, output_size, out_channels, true,
                 false, T(1.0), T(0.0));
  });
}

void CPUEngine::legacy_conv2d_bgrad(void*, const void* gradient_data, void* grad_bias_data,
                                    size_t batch_size, size_t output_h, size_t output_w,
                                    size_t out_channels, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    conv2d_bgrad_naive_impl<T>(static_cast<const T*>(gradient_data),
                               static_cast<T*>(grad_bias_data), batch_size, out_channels, output_h,
                               output_w);
  });
}

void CPUEngine::legacy_conv2d_add_bias(void*, void* output_data, const void* bias_data,
                                       size_t batch_size, size_t output_h, size_t output_w,
                                       size_t out_channels, DTypeDesc type_desc) {
  DISPATCH_DTYPE(type_desc.compute_dtype, T, {
    add_bias_impl<T>(static_cast<T*>(output_data), static_cast<const T*>(bias_data),
                     batch_size * output_h * output_w, out_channels);
  });
}

}  // namespace tunx
