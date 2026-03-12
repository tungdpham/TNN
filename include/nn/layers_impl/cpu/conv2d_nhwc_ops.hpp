#pragma once

#include <cstddef>

namespace tnn {
namespace cpu {
namespace conv2d_nhwc {

template <typename T>
void forward(const T *input, const T *weights, const T *bias, T *output, size_t batch, size_t in_h,
             size_t in_w, size_t in_c, size_t out_c, size_t k_h, size_t k_w, size_t s_h, size_t s_w,
             size_t p_h, size_t p_w, size_t out_h, size_t out_w, bool use_bias) {
  for (size_t n = 0; n < batch; ++n) {
    for (size_t oh = 0; oh < out_h; ++oh) {
      for (size_t ow = 0; ow < out_w; ++ow) {
        for (size_t oc = 0; oc < out_c; ++oc) {
          T sum = use_bias ? bias[oc] : T(0);
          for (size_t kh = 0; kh < k_h; ++kh) {
            int ih = static_cast<int>(oh * s_h + kh) - static_cast<int>(p_h);
            if (ih < 0 || ih >= static_cast<int>(in_h)) continue;

            for (size_t kw = 0; kw < k_w; ++kw) {
              int iw = static_cast<int>(ow * s_w + kw) - static_cast<int>(p_w);
              if (iw < 0 || iw >= static_cast<int>(in_w)) continue;

              for (size_t ic = 0; ic < in_c; ++ic) {
                // Input: [N, H, W, C]
                // Weight: [OC, KH, KW, IC]
                sum += input[((n * in_h + ih) * in_w + iw) * in_c + ic] *
                       weights[((oc * k_h + kh) * k_w + kw) * in_c + ic];
              }
            }
          }
          output[((n * out_h + oh) * out_w + ow) * out_c + oc] = sum;
        }
      }
    }
  }
}

template <typename T>
void backward_data(const T *grad_output, const T *weights, T *grad_input, size_t batch, size_t in_h,
                   size_t in_w, size_t in_c, size_t out_c, size_t k_h, size_t k_w, size_t s_h,
                   size_t s_w, size_t p_h, size_t p_w, size_t out_h, size_t out_w) {
  // Initialize grad_input with zeros
  for (size_t i = 0; i < batch * in_h * in_w * in_c; ++i) {
    grad_input[i] = T(0);
  }

  for (size_t n = 0; n < batch; ++n) {
    for (size_t oh = 0; oh < out_h; ++oh) {
      for (size_t ow = 0; ow < out_w; ++ow) {
        for (size_t oc = 0; oc < out_c; ++oc) {
          T grad_val = grad_output[((n * out_h + oh) * out_w + ow) * out_c + oc];
          for (size_t kh = 0; kh < k_h; ++kh) {
            int ih = static_cast<int>(oh * s_h + kh) - static_cast<int>(p_h);
            if (ih < 0 || ih >= static_cast<int>(in_h)) continue;

            for (size_t kw = 0; kw < k_w; ++kw) {
              int iw = static_cast<int>(ow * s_w + kw) - static_cast<int>(p_w);
              if (iw < 0 || iw >= static_cast<int>(in_w)) continue;

              for (size_t ic = 0; ic < in_c; ++ic) {
                grad_input[((n * in_h + ih) * in_w + iw) * in_c + ic] +=
                    grad_val * weights[((oc * k_h + kh) * k_w + kw) * in_c + ic];
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void backward_weights(const T *input, const T *grad_output, T *grad_weights, size_t batch,
                      size_t in_h, size_t in_w, size_t in_c, size_t out_c, size_t k_h, size_t k_w,
                      size_t s_h, size_t s_w, size_t p_h, size_t p_w, size_t out_h, size_t out_w) {
  // Initialize grad_weights with zeros
  for (size_t i = 0; i < out_c * k_h * k_w * in_c; ++i) {
    grad_weights[i] = T(0);
  }

  for (size_t n = 0; n < batch; ++n) {
    for (size_t oh = 0; oh < out_h; ++oh) {
      for (size_t ow = 0; ow < out_w; ++ow) {
        for (size_t oc = 0; oc < out_c; ++oc) {
          T grad_val = grad_output[((n * out_h + oh) * out_w + ow) * out_c + oc];
          for (size_t kh = 0; kh < k_h; ++kh) {
            int ih = static_cast<int>(oh * s_h + kh) - static_cast<int>(p_h);
            if (ih < 0 || ih >= static_cast<int>(in_h)) continue;

            for (size_t kw = 0; kw < k_w; ++kw) {
              int iw = static_cast<int>(ow * s_w + kw) - static_cast<int>(p_w);
              if (iw < 0 || iw >= static_cast<int>(in_w)) continue;

              for (size_t ic = 0; ic < in_c; ++ic) {
                grad_weights[((oc * k_h + kh) * k_w + kw) * in_c + ic] +=
                    grad_val * input[((n * in_h + ih) * in_w + iw) * in_c + ic];
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void backward_bias(const T *grad_output, T *grad_bias, size_t batch, size_t out_h, size_t out_w,
                   size_t out_c) {
  // Initialize grad_bias with zeros
  for (size_t i = 0; i < out_c; ++i) {
    grad_bias[i] = T(0);
  }

  for (size_t n = 0; n < batch; ++n) {
    for (size_t oh = 0; oh < out_h; ++oh) {
      for (size_t ow = 0; ow < out_w; ++ow) {
        for (size_t oc = 0; oc < out_c; ++oc) {
          grad_bias[oc] += grad_output[((n * out_h + oh) * out_w + ow) * out_c + oc];
        }
      }
    }
  }
}

}  // namespace conv2d_nhwc
}  // namespace cpu
}  // namespace tnn
