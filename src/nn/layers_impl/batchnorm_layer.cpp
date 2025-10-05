/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/batchnorm_layer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "utils/avx2.hpp"
#include "utils/ops.hpp"

namespace tnn {

template <typename T>
inline void compute_channel_mean(const T *input_data, T *mean_data, size_t batch_size,
                                 size_t channels, size_t spatial_size) {
  const size_t total_elements = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements);

  utils::parallel_for<size_t>(0, channels, [&](size_t c) {
    T sum = T(0);
    const size_t channel_stride = channels * spatial_size;
    const size_t c_offset = c * spatial_size;

    for (size_t n = 0; n < batch_size; ++n) {
      const T *batch_channel_ptr = input_data + n * channel_stride + c_offset;
      sum += utils::avx2_sum(batch_channel_ptr, spatial_size);
    }

    mean_data[c] = sum * inv_total;
  });
}

template <typename T>
inline void compute_channel_variance_optimized(const T *input_data, const T *mean_data, T *var_data,
                                               size_t batch_size, size_t channels,
                                               size_t spatial_size) {
  const size_t total_elements = batch_size * spatial_size;
  const T inv_total = T(1) / static_cast<T>(total_elements);

  utils::parallel_for<size_t>(0, channels, [&](size_t c) {
    T sum_sq = T(0);
    const T mean_val = mean_data[c];
    const size_t channel_stride = channels * spatial_size;
    const size_t c_offset = c * spatial_size;

    for (size_t n = 0; n < batch_size; ++n) {
      const T *batch_channel_ptr = input_data + n * channel_stride + c_offset;

      for (size_t i = 0; i < spatial_size; ++i) {
        T diff = batch_channel_ptr[i] - mean_val;
        sum_sq += diff * diff;
      }
    }

    var_data[c] = sum_sq * inv_total;
  });
}

template <typename T>
inline void normalize_and_scale_optimized(const T *input_data, const T *mean_data,
                                          const T *std_data, const T *gamma_data,
                                          const T *beta_data, T *output_data, T *normalized_data,
                                          size_t batch_size, size_t channels, size_t spatial_size,
                                          bool affine) {
  const size_t channel_stride = channels * spatial_size;

  utils::parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    const T mean_val = mean_data[c];
    const T std_val = std_data[c];
    const T inv_std = T(1) / std_val;

    const size_t n_offset = n * channel_stride;
    const size_t c_offset = c * spatial_size;
    const size_t base_idx = n_offset + c_offset;

    const T *input_ptr = input_data + base_idx;
    T *normalized_ptr = normalized_data + base_idx;
    T *output_ptr = output_data + base_idx;

    // Normalize: (x - mean) / std - vectorized with AVX2
    utils::avx2_sub_mul_scalar(input_ptr, mean_val, inv_std, normalized_ptr, spatial_size);

    if (affine) {
      const T gamma_val = gamma_data[c];
      const T beta_val = beta_data[c];

      // Scale and shift: gamma * normalized + beta - vectorized FMA with AVX2
      utils::avx2_mul_add_scalar(normalized_ptr, gamma_val, beta_val, output_ptr, spatial_size);
    } else {
      utils::avx2_copy(normalized_ptr, output_ptr, spatial_size);
    }
  });
}

template <typename T>
inline void compute_affine_gradients_optimized(const T *gradient_data, const T *normalized_data,
                                               T *gamma_grad, T *beta_grad, size_t batch_size,
                                               size_t channels, size_t spatial_size) {
  const size_t channel_stride = channels * spatial_size;

  utils::parallel_for<size_t>(0, channels, [&](size_t c) {
    T gamma_sum = T(0);
    T beta_sum = T(0);
    const size_t c_offset = c * spatial_size;

    for (size_t n = 0; n < batch_size; ++n) {
      const size_t base_idx = n * channel_stride + c_offset;
      const T *grad_ptr = gradient_data + base_idx;
      const T *norm_ptr = normalized_data + base_idx;

      for (size_t i = 0; i < spatial_size; ++i) {
        gamma_sum += grad_ptr[i] * norm_ptr[i];
        beta_sum += grad_ptr[i];
      }
    }

    gamma_grad[c] += gamma_sum;
    beta_grad[c] += beta_sum;
  });
}

template <typename T>
BatchNormLayer<T>::BatchNormLayer(size_t num_features, T epsilon, T momentum, bool affine,
                                  const std::string &name)
    : ParameterizedLayer<T>(name), num_features_(num_features), epsilon_(epsilon),
      momentum_(momentum), affine_(affine) {

  if (affine_) {
    gamma_ = Tensor<T>(num_features, 1, 1, 1);
    beta_ = Tensor<T>(num_features, 1, 1, 1);
    gamma_gradients_ = Tensor<T>(num_features, 1, 1, 1);
    beta_gradients_ = Tensor<T>(num_features, 1, 1, 1);

    gamma_.fill(T(1));
    beta_.fill(T(0));
  }

  running_mean_ = Tensor<T>(num_features, 1, 1, 1);
  running_var_ = Tensor<T>(num_features, 1, 1, 1);
  running_mean_.fill(T(0));
  running_var_.fill(T(1));
}

template <typename T>
Tensor<T> BatchNormLayer<T>::forward(const Tensor<T> &input, size_t micro_batch_id) {
  if (input.channels() != num_features_) {
    throw std::invalid_argument("Input channels must match num_features in BatchNormLayer");
  }

  micro_batch_inputs_[micro_batch_id] = input.clone();

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height = input.height();
  const size_t width = input.width();
  const size_t spatial_size = height * width;

  Tensor<T> output(input.shape(), nullptr);

  if (this->is_training_) {
    Tensor<T> batch_mean(channels, 1, 1, 1, nullptr);
    Tensor<T> batch_var(channels, 1, 1, 1, nullptr);
    Tensor<T> batch_std(channels, 1, 1, 1, nullptr);

    compute_channel_mean(input.data(), batch_mean.data(), batch_size, channels, spatial_size);

    compute_channel_variance_optimized(input.data(), batch_mean.data(), batch_var.data(),
                                       batch_size, channels, spatial_size);

    for (size_t c = 0; c < channels; ++c) {
      batch_std(c, 0, 0, 0) = std::sqrt(batch_var(c, 0, 0, 0) + epsilon_);
    }

    micro_batch_std_[micro_batch_id] = batch_std.clone();

    Tensor<T> normalized(input.shape());

    normalize_and_scale_optimized(input.data(), batch_mean.data(), batch_std.data(),
                                  affine_ ? gamma_.data() : nullptr,
                                  affine_ ? beta_.data() : nullptr, output.data(),
                                  normalized.data(), batch_size, channels, spatial_size, affine_);

    micro_batch_normalized_[micro_batch_id] = normalized.clone();

    utils::parallel_for<size_t>(0, channels, [&](size_t c) {
      running_mean_(c, 0, 0, 0) =
          (T(1) - momentum_) * running_mean_(c, 0, 0, 0) + momentum_ * batch_mean(c, 0, 0, 0);
      running_var_(c, 0, 0, 0) =
          (T(1) - momentum_) * running_var_(c, 0, 0, 0) + momentum_ * batch_var(c, 0, 0, 0);
    });

  } else {
    utils::parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
      T mean_val = running_mean_(c, 0, 0, 0);
      T var_val = running_var_(c, 0, 0, 0);
      T std_val = std::sqrt(var_val + epsilon_);
      const T inv_std = T(1) / std_val;

      const size_t channel_stride = channels * spatial_size;
      const size_t base_idx = n * channel_stride + c * spatial_size;

      const T *input_ptr = input.data() + base_idx;
      T *output_ptr = output.data() + base_idx;

      if (affine_) {
        const T gamma_val = gamma_(c, 0, 0, 0);
        const T beta_val = beta_(c, 0, 0, 0);

        for (size_t i = 0; i < spatial_size; ++i) {
          T normalized_val = (input_ptr[i] - mean_val) * inv_std;
          output_ptr[i] = gamma_val * normalized_val + beta_val;
        }
      } else {
        for (size_t i = 0; i < spatial_size; ++i) {
          output_ptr[i] = (input_ptr[i] - mean_val) * inv_std;
        }
      }
    });
  }

  return output;
}

template <typename T>
Tensor<T> BatchNormLayer<T>::backward(const Tensor<T> &gradient, size_t micro_batch_id) {
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  auto it_normalized = micro_batch_normalized_.find(micro_batch_id);
  auto it_std = micro_batch_std_.find(micro_batch_id);

  if (it_input == micro_batch_inputs_.end() || it_normalized == micro_batch_normalized_.end() ||
      it_std == micro_batch_std_.end()) {
    throw std::runtime_error("No cached data found for micro-batch ID in BatchNormLayer: " +
                             std::to_string(micro_batch_id));
  }

  const Tensor<T> &input = it_input->second;
  const Tensor<T> &normalized = it_normalized->second;
  const Tensor<T> &std_val = it_std->second;

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height = input.height();
  const size_t width = input.width();
  const size_t spatial_size = height * width;
  const size_t total_elements = batch_size * spatial_size;
  const size_t channel_stride = channels * spatial_size;

  Tensor<T> grad_input(input.shape(), nullptr);

  if (affine_) {
    compute_affine_gradients_optimized(gradient.data(), normalized.data(), gamma_gradients_.data(),
                                       beta_gradients_.data(), batch_size, channels, spatial_size);
  }

  Tensor<T> grad_normalized(input.shape(), nullptr);
  if (affine_) {
    utils::parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
      const T gamma_val = gamma_(c, 0, 0, 0);
      const size_t base_idx = n * channel_stride + c * spatial_size;

      const T *grad_ptr = gradient.data() + base_idx;
      T *grad_norm_ptr = grad_normalized.data() + base_idx;

      utils::avx2_mul_scalar(grad_ptr, gamma_val, grad_norm_ptr, spatial_size);
    });
  } else {
    utils::avx2_copy(gradient.data(), grad_normalized.data(), gradient.size());
  }

  const T inv_total = T(1) / static_cast<T>(total_elements);

  Tensor<T> sum_grad_normalized(channels, 1, 1, 1, nullptr);
  Tensor<T> sum_grad_normalized_times_normalized(channels, 1, 1, 1, nullptr);

  utils::parallel_for<size_t>(0, channels, [&](size_t c) {
    T sum_grad_norm = T(0);
    T sum_grad_norm_x_norm = T(0);
    const size_t c_offset = c * spatial_size;

    for (size_t n = 0; n < batch_size; ++n) {
      const size_t base_idx = n * channel_stride + c_offset;
      const T *grad_norm_ptr = grad_normalized.data() + base_idx;
      const T *norm_ptr = normalized.data() + base_idx;

      for (size_t i = 0; i < spatial_size; ++i) {
        sum_grad_norm += grad_norm_ptr[i];
        sum_grad_norm_x_norm += grad_norm_ptr[i] * norm_ptr[i];
      }
    }

    sum_grad_normalized(c, 0, 0, 0) = sum_grad_norm;
    sum_grad_normalized_times_normalized(c, 0, 0, 0) = sum_grad_norm_x_norm;
  });

  utils::parallel_for_2d<size_t>(batch_size, channels, [&](size_t n, size_t c) {
    const T std_val_c = std_val(c, 0, 0, 0);
    const T inv_std = T(1) / std_val_c;
    const T sum_grad_norm = sum_grad_normalized(c, 0, 0, 0);
    const T sum_grad_norm_x_norm = sum_grad_normalized_times_normalized(c, 0, 0, 0);

    const size_t base_idx = n * channel_stride + c * spatial_size;
    const T *grad_norm_ptr = grad_normalized.data() + base_idx;
    const T *norm_ptr = normalized.data() + base_idx;
    T *grad_input_ptr = grad_input.data() + base_idx;

    for (size_t i = 0; i < spatial_size; ++i) {
      // ∂L/∂x = (1/N) * (1/σ) * [N * ∂L/∂x̂ - Σ(∂L/∂x̂) - x̂ * Σ(∂L/∂x̂ * x̂)]
      grad_input_ptr[i] = inv_std * inv_total *
                          (static_cast<T>(total_elements) * grad_norm_ptr[i] - sum_grad_norm -
                           norm_ptr[i] * sum_grad_norm_x_norm);
    }
  });

  return grad_input;
}

template <typename T> std::string BatchNormLayer<T>::type() const { return "batchnorm"; }

template <typename T> LayerConfig BatchNormLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["num_features"] = num_features_;
  config.parameters["epsilon"] = epsilon_;
  config.parameters["momentum"] = momentum_;
  config.parameters["affine"] = affine_;
  return config;
}

template <typename T> std::unique_ptr<Layer<T>> BatchNormLayer<T>::clone() const {
  return std::make_unique<BatchNormLayer<T>>(num_features_, epsilon_, momentum_, affine_,
                                             this->name_);
}

template <typename T>
std::vector<size_t>
BatchNormLayer<T>::compute_output_shape(const std::vector<size_t> &input_shape) const {
  return input_shape;
}

template <typename T> void BatchNormLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  if (affine_) {
    params.push_back(&gamma_);
    params.push_back(&beta_);
  }
}

template <typename T> void BatchNormLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  if (affine_) {
    grads.push_back(&gamma_gradients_);
    grads.push_back(&beta_gradients_);
  }
}

template <typename T>
std::unique_ptr<Layer<T>> BatchNormLayer<T>::create_from_config(const LayerConfig &config) {
  size_t num_features = config.get<size_t>("num_features");
  T epsilon = config.get<T>("epsilon");
  T momentum = config.get<T>("momentum");
  bool affine = config.get<bool>("affine");

  return std::make_unique<BatchNormLayer<T>>(num_features, epsilon, momentum, affine, config.name);
}

template <typename T> void BatchNormLayer<T>::clear_gradients() {
  gamma_gradients_.fill(T(0));
  beta_gradients_.fill(T(0));
}

template <typename T>
uint64_t BatchNormLayer<T>::forward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t spatial_size = num_elements / (batch_size * num_features_);

  // mean computation: sum reduction across batch and spatial dimensions
  uint64_t mean_flops = batch_size * spatial_size * num_features_;

  // variance computation: (x - mean)^2 for each element + sum reduction
  uint64_t var_flops = 2 * num_elements + mean_flops;

  // normalization: (x - mean) / sqrt(var + epsilon) for each element
  uint64_t norm_flops = 3 * num_elements; // subtract, sqrt+add, divide

  // scale and shift (if affine): gamma * normalized + beta
  uint64_t affine_flops = affine_ ? (2 * num_elements) : 0;

  return mean_flops + var_flops + norm_flops + affine_flops;
}

template <typename T>
uint64_t BatchNormLayer<T>::backward_flops(const std::vector<size_t> &input_shape) const {
  size_t num_elements =
      std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
  size_t batch_size = input_shape[0];
  size_t spatial_size = num_elements / (batch_size * num_features_);

  // Gradient w.r.t. gamma and beta (if affine)
  uint64_t param_grad_flops = affine_ ? (2 * batch_size * spatial_size * num_features_) : 0;

  // Gradient w.r.t. input involves multiple terms:
  // 1. Direct gradient scaling: 2 * num_elements
  // 2. Mean gradient compensation: 3 * num_elements
  // 3. Variance gradient compensation: 4 * num_elements
  uint64_t input_grad_flops = 9 * num_elements;

  return param_grad_flops + input_grad_flops;
}

template <typename T>
uint64_t BatchNormLayer<T>::forward_complexity(const std::vector<size_t> &input_shape) {
  return static_cast<uint64_t>(
      std::min(forward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

template <typename T>
uint64_t BatchNormLayer<T>::backward_complexity(const std::vector<size_t> &input_shape) {
  return static_cast<uint64_t>(
      std::min(backward_flops(input_shape), static_cast<uint64_t>(UINT32_MAX)));
}

// Explicit template instantiations
template class BatchNormLayer<float>;
template class BatchNormLayer<double>;

} // namespace tnn
