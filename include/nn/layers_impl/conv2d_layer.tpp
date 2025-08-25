#include "conv2d_layer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../../utils/ops.hpp"
#include "../parallel_for.hpp"

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
  weights_ = Tensor<T>(
      out_channels, in_channels, kernel_h, kernel_w);
  weight_gradients_ = Tensor<T>(
      out_channels, in_channels, kernel_h, kernel_w);

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
Tensor<T> Conv2DLayer<T>::forward(const Tensor<T> &input, int micro_batch_id) {
  if (input.channels() != in_channels_) {
    std::cerr << "Input shape: " << input.channels()
              << " channels, expected: " << in_channels_ << " channels"
              << std::endl;
    throw std::invalid_argument("Input channel size mismatch in Conv2DLayer");
  }

  micro_batch_inputs_[micro_batch_id] = input.clone();

  const size_t batch_size = input.batch_size();
  const size_t input_h = input.height();
  const size_t input_w = input.width();

  // Calculate output dimensions
  const size_t output_h = (input_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  const size_t output_w = (input_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  // im2col transformation
  Matrix<T> col_matrix = input.im2col(kernel_h_, kernel_w_, stride_h_,
                                      stride_w_, pad_h_, pad_w_);
  micro_batch_im2col_matrices_[micro_batch_id] = col_matrix;

  Tensor<T> output(batch_size, out_channels_, output_h, output_w);

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;

  // Perform convolution
  auto output_flat = std::make_unique<T[]>(out_channels_ * output_size);
  compute_conv_forward(col_matrix.data(), weights_.data(), output_flat.get(),
                       output_size, kernel_size, out_channels_);

  T *output_data = output.data();

  const size_t N_stride = output.stride(0);
  const size_t C_stride = output.stride(1);
  const size_t H_stride = output.stride(2);
  const size_t W_stride = output.stride(3);

#ifdef USE_TBB
  tnn::parallel_for_2d(batch_size, out_channels_, [&](size_t n, size_t oc) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        size_t flat_idx =
            oc * output_size + n * (output_h * output_w) + oh * output_w + ow;
        output_data[n * N_stride + oc * C_stride + oh * H_stride +
                    ow * W_stride] = output_flat[flat_idx];
      }
    }
  });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t oc = 0; oc < out_channels_; ++oc) {
      for (size_t oh = 0; oh < output_h; ++oh) {
        for (size_t ow = 0; ow < output_w; ++ow) {
          size_t flat_idx = oc * output_size + n * (output_h * output_w) +
                            oh * output_w + ow;
          output_data[n * N_stride + oc * C_stride + oh * H_stride +
                      ow * W_stride] = output_flat[flat_idx];
        }
      }
    }
  }
#endif

  // Add bias if enabled
  if (use_bias_) {
#ifdef USE_TBB
    tnn::parallel_for_2d(batch_size, out_channels_, [&](size_t n, size_t oc) {
      T bias_val = bias_(oc, 0, 0, 0);
      for (size_t oh = 0; oh < output_h; ++oh) {
        for (size_t ow = 0; ow < output_w; ++ow) {
          output(n, oc, oh, ow) += bias_val;
        }
      }
    });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t oc = 0; oc < out_channels_; ++oc) {
        T bias_val = bias_(oc, 0, 0, 0);
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            output(n, oc, oh, ow) += bias_val;
          }
        }
      }
    }
#endif
  }

  // Store pre-activation output
  micro_batch_pre_activations_[micro_batch_id] = output.clone();

  // Apply activation if provided
  if (activation_) {
    activation_->apply(output);
  }

  return output;
}


template <typename T>
Tensor<T> Conv2DLayer<T>::backward(const Tensor<T> &grad_output,
                                   int micro_batch_id) {
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  auto it_pre_act = micro_batch_pre_activations_.find(micro_batch_id);
  auto it_im2col = micro_batch_im2col_matrices_.find(micro_batch_id);

  if (it_input == micro_batch_inputs_.end()) {
    throw std::runtime_error("No cached input found for micro-batch ID: " +
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

  const Tensor<T> &last_input = it_input->second;
  const Matrix<T> &cached_im2col_matrix = it_im2col->second;

  const size_t batch_size = last_input.batch_size();
  const size_t input_h = last_input.height();
  const size_t input_w = last_input.width();
  const size_t output_h = grad_output.height();
  const size_t output_w = grad_output.width();

  Tensor<T> current_grad = grad_output;

  if (activation_) {
    current_grad =
        activation_->compute_gradient(it_pre_act->second, &current_grad);
  }

  weight_gradients_.fill(T(0));
  if (use_bias_) {
    bias_gradients_.fill(T(0));
  }

  size_t kernel_size = in_channels_ * kernel_h_ * kernel_w_;
  size_t output_size = batch_size * output_h * output_w;

  auto grad_output_flat = std::make_unique<T[]>(out_channels_ * output_size);

#ifdef USE_TBB
  tnn::parallel_for_2d(batch_size, out_channels_, [&](size_t n, size_t oc) {
    for (size_t oh = 0; oh < output_h; ++oh) {
      for (size_t ow = 0; ow < output_w; ++ow) {
        grad_output_flat[oc * output_size + n * (output_h * output_w) +
                         oh * output_w + ow] = current_grad(n, oc, oh, ow);
      }
    }
  });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t oc = 0; oc < out_channels_; ++oc) {
      for (size_t oh = 0; oh < output_h; ++oh) {
        for (size_t ow = 0; ow < output_w; ++ow) {
          grad_output_flat[oc * output_size + n * (output_h * output_w) +
                           oh * output_w + ow] = current_grad(n, oc, oh, ow);
        }
      }
    }
  }
#endif

  compute_weight_gradients(cached_im2col_matrix.data(),
                          grad_output_flat.get(), weight_gradients_.data(),
                          output_size, kernel_size, out_channels_);

  if (use_bias_) {
#ifdef USE_TBB
    tnn::parallel_for_range<size_t>(0, out_channels_, [&](size_t oc) {
      T grad_sum = T(0);
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            grad_sum += current_grad(n, oc, oh, ow);
          }
        }
      }
      bias_gradients_(oc, 0, 0, 0) = grad_sum;
    });
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t oc = 0; oc < out_channels_; ++oc) {
      T grad_sum = T(0);
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t oh = 0; oh < output_h; ++oh) {
          for (size_t ow = 0; ow < output_w; ++ow) {
            grad_sum += current_grad(n, oc, oh, ow);
          }
        }
      }
      bias_gradients_(oc, 0, 0, 0) = grad_sum;
    }
#endif
  }

  Matrix<T> col_grad_matrix(kernel_size, output_size);
  compute_input_gradients(grad_output_flat.get(), weights_.data(),
                         col_grad_matrix.data(), output_size, kernel_size,
                         out_channels_);

  Tensor<T> grad_input = Tensor<T>::col2im(
      col_grad_matrix, batch_size, in_channels_, input_h, input_w, kernel_h_,
      kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);

  return grad_input;
}

template <typename T>
void Conv2DLayer<T>::compute_conv_forward(const T *col_data, const T *weight_data,
                                          T *output_data,
                                          const size_t output_size,
                                          const size_t kernel_size,
                                          const size_t out_channels) const {
  // Transpose for better memory access
  auto col_data_transposed = std::make_unique<T[]>(kernel_size * output_size);
  utils::transpose_2d(col_data, col_data_transposed.get(), kernel_size,
                      output_size);

#ifdef USE_TBB
  tbb::parallel_for(
      tbb::blocked_range2d<size_t>(0, out_channels, 0, output_size),
      [&](const tbb::blocked_range2d<size_t> &r) {
        for (size_t oc = r.rows().begin(); oc != r.rows().end(); ++oc) {
          for (size_t os = r.cols().begin(); os != r.cols().end(); ++os) {
            output_data[oc * output_size + os] =
                utils::simd_dot_product(
                    &weight_data[oc * kernel_size],
                    &col_data_transposed[os * kernel_size], kernel_size);
          }
        }
      });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t oc = 0; oc < out_channels; ++oc) {
    for (size_t os = 0; os < output_size; ++os) {
      output_data[oc * output_size + os] = utils::simd_dot_product(
          &weight_data[oc * kernel_size],
          &col_data_transposed[os * kernel_size], kernel_size);
    }
  }
#endif
}

template <typename T>
void Conv2DLayer<T>::compute_weight_gradients(const T *col_data,
                                              const T *grad_output_data,
                                              T *weight_grad_data,
                                              const size_t output_size,
                                              const size_t kernel_size,
                                              const size_t out_channels) const {
#ifdef USE_TBB
  tbb::parallel_for(
      tbb::blocked_range2d<size_t>(0, out_channels, 0, kernel_size),
      [&](const tbb::blocked_range2d<size_t> &r) {
        for (size_t oc = r.rows().begin(); oc != r.rows().end(); ++oc) {
          for (size_t ks = r.cols().begin(); ks != r.cols().end(); ++ks) {
            weight_grad_data[oc * kernel_size + ks] =
                utils::simd_dot_product(
                    &grad_output_data[oc * output_size],
                    &col_data[ks * output_size], output_size);
          }
        }
      });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t oc = 0; oc < out_channels; ++oc) {
    for (size_t ks = 0; ks < kernel_size; ++ks) {
      weight_grad_data[oc * kernel_size + ks] =
          utils::simd_dot_product(
              &grad_output_data[oc * output_size],
              &col_data[ks * output_size], output_size);
    }
  }
#endif
}

template <typename T>
void Conv2DLayer<T>::compute_input_gradients(const T *grad_output_data,
                                             const T *weight_data,
                                             T *col_grad_data,
                                             const size_t output_size,
                                             const size_t kernel_size,
                                             const size_t out_channels) const {
  auto grad_output_transposed =
      std::make_unique<T[]>(out_channels * output_size);
  utils::transpose_2d(grad_output_data, grad_output_transposed.get(),
                      out_channels, output_size);

  auto weights_transposed = std::make_unique<T[]>(kernel_size * out_channels);
  utils::transpose_2d(weight_data, weights_transposed.get(), out_channels,
                      kernel_size);

#ifdef USE_TBB
  tbb::parallel_for(
      tbb::blocked_range2d<size_t>(0, kernel_size, 0, output_size),
      [&](const tbb::blocked_range2d<size_t> &r) {
        for (size_t ks = r.rows().begin(); ks != r.rows().end(); ++ks) {
          for (size_t os = r.cols().begin(); os != r.cols().end(); ++os) {
            col_grad_data[ks * output_size + os] =
                utils::simd_dot_product(
                    &weights_transposed[ks * out_channels],
                    &grad_output_transposed[os * out_channels], out_channels);
          }
        }
      });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t ks = 0; ks < kernel_size; ++ks) {
    for (size_t os = 0; os < output_size; ++os) {
      col_grad_data[ks * output_size + os] =
          utils::simd_dot_product(
              &weights_transposed[ks * out_channels],
              &grad_output_transposed[os * out_channels], out_channels);
    }
  }
#endif
}


template <typename T>
std::string Conv2DLayer<T>::type() const {
  return "conv2d";
}

template <typename T>
LayerConfig Conv2DLayer<T>::get_config() const {
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

template <typename T>
std::unique_ptr<Layer<T>> Conv2DLayer<T>::clone() const {
  auto activation_clone = activation_ ? activation_->clone() : nullptr;
  return std::make_unique<Conv2DLayer<T>>(
      in_channels_, out_channels_, kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_h_, pad_w_, use_bias_, std::move(activation_clone), this->name_);
}

template <typename T>
std::vector<size_t>
Conv2DLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  if (input_shape.size() != 4) {
    throw std::invalid_argument("Conv2DLayer expects 4D input");
  }

  size_t output_h = (input_shape[2] + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  size_t output_w = (input_shape[3] + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  return {input_shape[0], out_channels_, output_h, output_w};
}

// Protected method implementations
template <typename T>
void Conv2DLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  params.push_back(&weights_);
  if (use_bias_) {
    params.push_back(&bias_);
  }
}

template <typename T>
void Conv2DLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  grads.push_back(&weight_gradients_);
  if (use_bias_) {
    grads.push_back(&bias_gradients_);
  }
}

template <typename T>
void Conv2DLayer<T>::update_parameters_impl(Optimizer<T> &optimizer) {
  std::vector<Tensor<T> *> params = this->parameters();
  std::vector<Tensor<T> *> grads = this->gradients();
  if (params.size() != grads.size()) {
    throw std::runtime_error(
        "Parameter and gradient size mismatch in Conv2DLayer");
  }
  optimizer.update(params, grads);
}

} // namespace tnn
