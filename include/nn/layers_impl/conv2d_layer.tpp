/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "conv2d_layer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "utils/ops.hpp"
#include "utils/parallel_for.hpp"

#ifdef USE_TBB
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#endif

namespace tnn {

// Constructor
template <typename T>
Conv2DLayer<T>::Conv2DLayer(size_t in_channels, size_t out_channels,
                            size_t kernel_h, size_t kernel_w, size_t stride_h,
                            size_t stride_w, size_t pad_h, size_t pad_w,
                            bool use_bias,
                            std::unique_ptr<ActivationFunction<T>> activation,
                            const std::string &name)
    : ParameterizedLayer<T>(name), in_channels_(in_channels),
      out_channels_(out_channels), kernel_h_(kernel_h), kernel_w_(kernel_w),
      stride_h_(stride_h), stride_w_(stride_w), pad_h_(pad_h), pad_w_(pad_w),
      use_bias_(use_bias), activation_(std::move(activation)),
      micro_batch_im2col_matrices_() {
  weights_ = Tensor<T>(out_channels, in_channels, kernel_h, kernel_w);
  weight_gradients_ = Tensor<T>(out_channels, in_channels, kernel_h, kernel_w);

  if (use_bias_) {
    bias_ = Tensor<T>(out_channels, 1, 1, 1);
    bias_gradients_ = Tensor<T>(out_channels, 1, 1, 1);
  }

  // Xavier/Glorot initialization
  T fan_in = static_cast<T>(in_channels * kernel_h * kernel_w);
  T fan_out = static_cast<T>(out_channels * kernel_h * kernel_w);
  T std_dev = std::sqrt(T(2.0) / (fan_in + fan_out));
  weights_.fill_random_normal(T(0), std_dev);
}

template <typename T>
Tensor<T> Conv2DLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  if (input.channels() != in_channels_) {
    std::cerr << "Input shape: " << input.channels()
              << " channels, expected: " << in_channels_ << " channels"
              << std::endl;
    throw std::invalid_argument("Input channel size mismatch in Conv2DLayer");
  }

  micro_batch_input_shapes_[micro_batch_id] = {input.batch_size(), input.channels(), input.height(), input.width()};

  const size_t batch_size = input.batch_size();
  const size_t input_h = input.height();
  const size_t input_w = input.width();

  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  // im2col transformation
  Matrix<T> col_matrix =
      input.im2col(kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);

  Tensor<T> output(batch_size, out_channels_, output_h, output_w, false);

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;

  T *output_flat = (T *)malloc(sizeof(T) * out_channels_ * output_size);
  compute_conv_forward(col_matrix.data(), weights_.data(), output_flat,
                       output_size, kernel_size, out_channels_);

  // cache the value since we are no longer using it
  micro_batch_im2col_matrices_[micro_batch_id] = std::move(col_matrix);

  utils::cnhw_to_nchw(output_flat, output.data(), batch_size, out_channels_,
                      output_h, output_w);

  free(output_flat);

  if (use_bias_) {
    add_bias_to_output(output.data(), bias_.data(), batch_size, output_h,
                       output_w, out_channels_);
  }

  micro_batch_pre_activations_[micro_batch_id] = output.clone();

  if (activation_) {
    activation_->apply(output);
  }

  return output;
}

template <typename T>
Tensor<T> Conv2DLayer<T>::backward(const Tensor<T> &gradient,
                                   size_t micro_batch_id) {
  auto it_input_shape = micro_batch_input_shapes_.find(micro_batch_id);
  auto it_pre_act = micro_batch_pre_activations_.find(micro_batch_id);
  auto it_im2col = micro_batch_im2col_matrices_.find(micro_batch_id);

  if (it_input_shape == micro_batch_input_shapes_.end()) {
    throw std::runtime_error("No cached input shape found for micro-batch ID: " +
                             std::to_string(micro_batch_id));
  }
  
  if (it_im2col == micro_batch_im2col_matrices_.end()) {
    throw std::runtime_error(
        "No cached im2col matrix found for micro-batch ID: " +
        std::to_string(micro_batch_id));
  }
  if (activation_ && it_pre_act == micro_batch_pre_activations_.end()) {
    throw std::runtime_error(
        "No cached pre-activation output found for micro-batch ID: " +
        std::to_string(micro_batch_id));
  }

  const auto& input_shape = it_input_shape->second;
  const Matrix<T> &cached_im2col_matrix = it_im2col->second;

  const size_t batch_size = input_shape[0];
  const size_t input_h = input_shape[2];
  const size_t input_w = input_shape[3];
  const size_t output_h = gradient.height();
  const size_t output_w = gradient.width();

  Tensor<T> current_grad = gradient.clone();

  if (activation_) {
    activation_->compute_gradient_inplace(it_pre_act->second, current_grad);
  }

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;

  T *gradient_flat = (T *)malloc(sizeof(T) * out_channels_ * output_size);

  utils::nchw_to_cnhw(current_grad.data(), gradient_flat, batch_size,
                      out_channels_, output_h, output_w);

  compute_weight_gradients(cached_im2col_matrix.data(), gradient_flat,
                           weight_gradients_.data(), output_size, kernel_size,
                           out_channels_);

  if (use_bias_) {
    compute_bias_gradients(current_grad.data(), bias_gradients_.data(),
                           batch_size, output_h, output_w, out_channels_);
  }

  Matrix<T> col_grad_matrix(kernel_size, output_size);
  compute_input_gradients(gradient_flat, weights_.data(),
                          col_grad_matrix.data(), output_size, kernel_size,
                          out_channels_);

  Tensor<T> grad_input = Tensor<T>::col2im(
      col_grad_matrix, batch_size, in_channels_, input_h, input_w, kernel_h_,
      kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);

  free(gradient_flat);
  return grad_input;
}

template <typename T>
void Conv2DLayer<T>::compute_conv_forward(const T *col_data,
                                          const T *weight_data, T *output_data,
                                          const size_t output_size,
                                          const size_t kernel_size,
                                          const size_t out_channels) const {
  T *col_data_transposed = (T *)malloc(sizeof(T) * kernel_size * output_size);
  utils::transpose_2d_inplace(col_data, col_data_transposed, kernel_size,
                              output_size);
  
  utils::parallel_for_2d(out_channels, output_size, 
    [&](size_t oc, size_t os) {
      output_data[oc * output_size + os] = utils::simd_dot_product(
          &weight_data[oc * kernel_size],
          &col_data_transposed[os * kernel_size], kernel_size);
    });
  
  free(col_data_transposed);
}

template <typename T>
void Conv2DLayer<T>::compute_weight_gradients(const T *col_data,
                                              const T *gradient_data,
                                              T *weight_grad_data,
                                              const size_t output_size,
                                              const size_t kernel_size,
                                              const size_t out_channels) const {
  utils::parallel_for_2d(out_channels, kernel_size, 
    [&](size_t oc, size_t ks) {
      weight_grad_data[oc * kernel_size + ks] +=
          utils::simd_dot_product(&gradient_data[oc * output_size],
                                  &col_data[ks * output_size], output_size);
    });
}

template <typename T>
void Conv2DLayer<T>::compute_input_gradients(const T *gradient_data,
                                             const T *weight_data,
                                             T *col_grad_data,
                                             const size_t output_size,
                                             const size_t kernel_size,
                                             const size_t out_channels) const {
  T *gradient_transposed = (T *)malloc(sizeof(T) * output_size * out_channels);
  utils::transpose_2d_inplace(gradient_data, gradient_transposed,
                              out_channels, output_size);

  T *weights_transposed = (T *)malloc(sizeof(T) * kernel_size * out_channels);
  utils::transpose_2d_inplace(weight_data, weights_transposed, out_channels,
                              kernel_size);

  utils::parallel_for_2d(kernel_size, output_size, 
    [&](size_t ks, size_t os) {
      col_grad_data[ks * output_size + os] = utils::simd_dot_product(
          &weights_transposed[ks * out_channels],
          &gradient_transposed[os * out_channels], out_channels);
    });

  free(gradient_transposed);
  free(weights_transposed);
}

template <typename T>
void Conv2DLayer<T>::compute_bias_gradients(const T *gradient_data,
                                            T *bias_grad_data,
                                            const size_t batch_size, const size_t output_h,
                                            const size_t output_w,
                                            const size_t out_channels) const {
  const size_t N_stride = out_channels * output_h * output_w;
  const size_t C_stride = output_h * output_w;

  utils::parallel_for_range<size_t>(0, out_channels, [&](size_t oc) {
    T grad_sum = T(0);
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t oh = 0; oh < output_h; ++oh) {
        for (size_t ow = 0; ow < output_w; ++ow) {
          grad_sum += gradient_data[n * N_stride + oc * C_stride +
                                       oh * output_w + ow];
        }
      }
    }
    bias_grad_data[oc] += grad_sum;
  });
}

template <typename T>
void Conv2DLayer<T>::add_bias_to_output(T *output_data, const T *bias_data,
                                        const size_t batch_size, const size_t output_h,
                                        const size_t output_w,
                                        const size_t out_channels) const {
  const size_t N_stride = out_channels * output_h * output_w;
  const size_t C_stride = output_h * output_w;
  const size_t H_stride = output_w;
  const size_t W_stride = 1;

  utils::parallel_for_2d(batch_size, out_channels, [&](size_t n, size_t oc) {
    T bias_val = bias_data[oc];
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        output_data[n * N_stride + oc * C_stride + oh * H_stride +
                    ow * W_stride] += bias_val;
      }
    }
  });
}

template <typename T> std::string Conv2DLayer<T>::type() const {
  return "conv2d";
}

template <typename T> LayerConfig Conv2DLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["in_channels"] = in_channels_;
  config.parameters["out_channels"] = out_channels_;
  config.parameters["kernel_h"] = kernel_h_;
  config.parameters["kernel_w"] = kernel_w_;
  config.parameters["stride_h"] = stride_h_;
  config.parameters["stride_w"] = stride_w_;
  config.parameters["pad_h"] = pad_h_;
  config.parameters["pad_w"] = pad_w_;
  config.parameters["use_bias"] = use_bias_;
  config.parameters["activation"] =
      activation_ ? activation_->name() : std::string("none");
  config.parameters["optimized"] = std::string("native");
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> Conv2DLayer<T>::clone() const {
  auto activation_clone = activation_ ? activation_->clone() : nullptr;
  return std::make_unique<Conv2DLayer<T>>(
      in_channels_, out_channels_, kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_h_, pad_w_, use_bias_, std::move(activation_clone), this->name_);
}

template <typename T>
std::vector<size_t> Conv2DLayer<T>::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("Conv2DLayer expects 4D input");
  }

  size_t output_h = (input_shape[2] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[3] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {input_shape[0], out_channels_, output_h, output_w};
}

template <typename T>
void Conv2DLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.reserve(use_bias_ ? 2 : 1);
  params.push_back(&weights_);
  if (use_bias_) {
    params.push_back(&bias_);
  }
}

template <typename T>
void Conv2DLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.reserve(use_bias_ ? 2 : 1);
  grads.push_back(&weight_gradients_);
  if (use_bias_) {
    grads.push_back(&bias_gradients_);
  }
}

template <typename T> void Conv2DLayer<T>::clear_gradients() {
  weight_gradients_.fill(T(0));
  if (use_bias_) {
    bias_gradients_.fill(T(0));
  }
}

} // namespace tnn
