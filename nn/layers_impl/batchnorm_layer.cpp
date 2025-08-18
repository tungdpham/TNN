#include "batchnorm_layer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../parallel_for.hpp"
#include "parameterized_layer.hpp"

namespace tnn {

// Constructor
template <typename T>
BatchNormLayer<T>::BatchNormLayer(size_t num_features, T epsilon, T momentum,
                                  bool affine, const std::string &name)
    : ParameterizedLayer<T>(name), num_features_(num_features),
      epsilon_(epsilon), momentum_(momentum), affine_(affine) {

  if (affine_) {
    gamma_ = Tensor<T>(std::vector<size_t>{num_features, 1, 1, 1});
    beta_ = Tensor<T>(std::vector<size_t>{num_features, 1, 1, 1});
    gamma_gradients_ = Tensor<T>(std::vector<size_t>{num_features, 1, 1, 1});
    beta_gradients_ = Tensor<T>(std::vector<size_t>{num_features, 1, 1, 1});

    // Initialize gamma to 1 and beta to 0
    gamma_.fill(T(1));
    beta_.fill(T(0));
  }

  // Initialize running statistics
  running_mean_ = Tensor<T>(std::vector<size_t>{num_features, 1, 1, 1});
  running_var_ = Tensor<T>(std::vector<size_t>{num_features, 1, 1, 1});
  running_mean_.fill(T(0));
  running_var_.fill(T(1));
}

// Forward pass
template <typename T>
Tensor<T> BatchNormLayer<T>::forward(const Tensor<T> &input,
                                     int micro_batch_id) {
  if (input.channels() != num_features_) {
    throw std::invalid_argument(
        "Input channels must match num_features in BatchNormLayer");
  }

  micro_batch_inputs_[micro_batch_id] = input;

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height = input.height();
  const size_t width = input.width();

  Tensor<T> output(input.shape());

  if (this->is_training_) {
    // Training mode: compute batch statistics
    Tensor<T> batch_mean(std::vector<size_t>{channels, 1, 1, 1});
    Tensor<T> batch_var(std::vector<size_t>{channels, 1, 1, 1});

    // Compute mean for each channel
    batch_mean.fill(T(0));
    const size_t spatial_size = height * width;
    const size_t total_elements = batch_size * spatial_size;

#if defined(USE_TBB)
    tnn::parallel_for_range<size_t>(0, channels, [&](size_t c) {
      T sum = T(0);
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            sum += input(n, c, h, w);
          }
        }
      }
      batch_mean(c, 0, 0, 0) = sum / static_cast<T>(total_elements);
    });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t c = 0; c < channels; ++c) {
      T sum = T(0);
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            sum += input(n, c, h, w);
          }
        }
      }
      batch_mean(c, 0, 0, 0) = sum / static_cast<T>(total_elements);
    }
#endif

    // Compute variance for each channel
    batch_var.fill(T(0));
#if defined(USE_TBB)
    tnn::parallel_for_range<size_t>(0, channels, [&](size_t c) {
      T sum_sq_diff = T(0);
      T mean_val = batch_mean(c, 0, 0, 0);
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            T diff = input(n, c, h, w) - mean_val;
            sum_sq_diff += diff * diff;
          }
        }
      }
      batch_var(c, 0, 0, 0) = sum_sq_diff / static_cast<T>(total_elements);
    });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t c = 0; c < channels; ++c) {
      T sum_sq_diff = T(0);
      T mean_val = batch_mean(c, 0, 0, 0);
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            T diff = input(n, c, h, w) - mean_val;
            sum_sq_diff += diff * diff;
          }
        }
      }
      batch_var(c, 0, 0, 0) = sum_sq_diff / static_cast<T>(total_elements);
    }
#endif

    // Compute standard deviation
    Tensor<T> batch_std(std::vector<size_t>{channels, 1, 1, 1});
    for (size_t c = 0; c < channels; ++c) {
      batch_std(c, 0, 0, 0) = std::sqrt(batch_var(c, 0, 0, 0) + epsilon_);
    }

    // Store for backward pass
    micro_batch_mean_[micro_batch_id] = batch_mean;
    micro_batch_var_[micro_batch_id] = batch_var;
    micro_batch_std_[micro_batch_id] = batch_std;

    // Normalize
    Tensor<T> normalized(input.shape());
#if defined(USE_TBB)
    tnn::parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
      T mean_val = batch_mean(c, 0, 0, 0);
      T std_val = batch_std(c, 0, 0, 0);
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          normalized(n, c, h, w) = (input(n, c, h, w) - mean_val) / std_val;
        }
      }
    });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        T mean_val = batch_mean(c, 0, 0, 0);
        T std_val = batch_std(c, 0, 0, 0);
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            normalized(n, c, h, w) = (input(n, c, h, w) - mean_val) / std_val;
          }
        }
      }
    }
#endif

    micro_batch_normalized_[micro_batch_id] = normalized;

    // Apply affine transformation if enabled
    if (affine_) {
#if defined(USE_TBB)
      tnn::parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
        T gamma_val = gamma_(c, 0, 0, 0);
        T beta_val = beta_(c, 0, 0, 0);
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            output(n, c, h, w) =
                gamma_val * normalized(n, c, h, w) + beta_val;
          }
        }
      });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
          T gamma_val = gamma_(c, 0, 0, 0);
          T beta_val = beta_(c, 0, 0, 0);
          for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
              output(n, c, h, w) =
                  gamma_val * normalized(n, c, h, w) + beta_val;
            }
          }
        }
      }
#endif
    } else {
      output = normalized;
    }

    // Update running statistics
#if defined(USE_TBB)
    tnn::parallel_for_range<size_t>(0, channels, [&](size_t c) {
      running_mean_(c, 0, 0, 0) =
          (T(1) - momentum_) * running_mean_(c, 0, 0, 0) +
          momentum_ * batch_mean(c, 0, 0, 0);
      running_var_(c, 0, 0, 0) =
          (T(1) - momentum_) * running_var_(c, 0, 0, 0) +
          momentum_ * batch_var(c, 0, 0, 0);
    });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t c = 0; c < channels; ++c) {
      running_mean_(c, 0, 0, 0) =
          (T(1) - momentum_) * running_mean_(c, 0, 0, 0) +
          momentum_ * batch_mean(c, 0, 0, 0);
      running_var_(c, 0, 0, 0) =
          (T(1) - momentum_) * running_var_(c, 0, 0, 0) +
          momentum_ * batch_var(c, 0, 0, 0);
    }
#endif

  } else {
    // Inference mode: use running statistics
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        T mean_val = running_mean_(c, 0, 0, 0);
        T var_val = running_var_(c, 0, 0, 0);
        T std_val = std::sqrt(var_val + epsilon_);

        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            T normalized_val = (input(n, c, h, w) - mean_val) / std_val;
            if (affine_) {
              output(n, c, h, w) =
                  gamma_(c, 0, 0, 0) * normalized_val + beta_(c, 0, 0, 0);
            } else {
              output(n, c, h, w) = normalized_val;
            }
          }
        }
      }
    }
  }

  return output;
}

// Backward pass
template <typename T>
Tensor<T> BatchNormLayer<T>::backward(const Tensor<T> &grad_output,
                                      int micro_batch_id) {
  auto it_input = micro_batch_inputs_.find(micro_batch_id);
  auto it_normalized = micro_batch_normalized_.find(micro_batch_id);
  auto it_mean = micro_batch_mean_.find(micro_batch_id);
  auto it_var = micro_batch_var_.find(micro_batch_id);
  auto it_std = micro_batch_std_.find(micro_batch_id);

  if (it_input == micro_batch_inputs_.end() ||
      it_normalized == micro_batch_normalized_.end() ||
      it_mean == micro_batch_mean_.end() ||
      it_var == micro_batch_var_.end() || it_std == micro_batch_std_.end()) {
    throw std::runtime_error(
        "No cached data found for micro-batch ID in BatchNormLayer: " +
        std::to_string(micro_batch_id));
  }

  const Tensor<T> &input = it_input->second;
  const Tensor<T> &normalized = it_normalized->second;
  const Tensor<T> &mean = it_mean->second;
  const Tensor<T> &var = it_var->second;
  const Tensor<T> &std_val = it_std->second;

  const size_t batch_size = input.batch_size();
  const size_t channels = input.channels();
  const size_t height = input.height();
  const size_t width = input.width();
  const size_t spatial_size = height * width;
  const size_t total_elements = batch_size * spatial_size;

  Tensor<T> grad_input(input.shape());

  if (affine_) {
    // Initialize gradients
    gamma_gradients_.fill(T(0));
    beta_gradients_.fill(T(0));

    // Compute gradients for gamma and beta
#if defined(USE_TBB)
    tnn::parallel_for_range<size_t>(0, channels, [&](size_t c) {
      T gamma_grad_sum = T(0);
      T beta_grad_sum = T(0);

      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            gamma_grad_sum +=
                grad_output(n, c, h, w) * normalized(n, c, h, w);
            beta_grad_sum += grad_output(n, c, h, w);
          }
        }
      }

      gamma_gradients_(c, 0, 0, 0) = gamma_grad_sum;
      beta_gradients_(c, 0, 0, 0) = beta_grad_sum;
    });
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t c = 0; c < channels; ++c) {
      T gamma_grad_sum = T(0);
      T beta_grad_sum = T(0);

      for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            gamma_grad_sum +=
                grad_output(n, c, h, w) * normalized(n, c, h, w);
            beta_grad_sum += grad_output(n, c, h, w);
          }
        }
      }

      gamma_gradients_(c, 0, 0, 0) = gamma_grad_sum;
      beta_gradients_(c, 0, 0, 0) = beta_grad_sum;
    }
#endif
  }

  // Compute gradient w.r.t. normalized input
  Tensor<T> grad_normalized(input.shape());
  if (affine_) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        T gamma_val = gamma_(c, 0, 0, 0);
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            grad_normalized(n, c, h, w) = grad_output(n, c, h, w) * gamma_val;
          }
        }
      }
    }
  } else {
    grad_normalized = grad_output;
  }

  // Compute gradients w.r.t. variance and mean
  Tensor<T> grad_var(std::vector<size_t>{channels, 1, 1, 1});
  Tensor<T> grad_mean(std::vector<size_t>{channels, 1, 1, 1});

  grad_var.fill(T(0));
  grad_mean.fill(T(0));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (size_t c = 0; c < channels; ++c) {
    T mean_val = mean(c, 0, 0, 0);
    T std_val_c = std_val(c, 0, 0, 0);
    T var_val = var(c, 0, 0, 0);

    T grad_var_sum = T(0);
    T grad_mean_sum = T(0);

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T x_centered = input(n, c, h, w) - mean_val;
          grad_var_sum += grad_normalized(n, c, h, w) * x_centered *
                          (-T(0.5)) * std::pow(var_val + epsilon_, -T(1.5));
          grad_mean_sum += grad_normalized(n, c, h, w) * (-T(1)) / std_val_c;
        }
      }
    }

    grad_var(c, 0, 0, 0) = grad_var_sum;
    grad_mean(c, 0, 0, 0) = grad_mean_sum;
  }

  // Compute gradient w.r.t. input
#if defined(USE_TBB)
  tnn::parallel_for_2d(batch_size, channels, [&](size_t n, size_t c) {
    T mean_val = mean(c, 0, 0, 0);
    T std_val_c = std_val(c, 0, 0, 0);
    T grad_var_val = grad_var(c, 0, 0, 0);
    T grad_mean_val = grad_mean(c, 0, 0, 0);

    for (size_t h = 0; h < height; ++h) {
      for (size_t w = 0; w < width; ++w) {
        T x_centered = input(n, c, h, w) - mean_val;

        grad_input(n, c, h, w) =
            grad_normalized(n, c, h, w) / std_val_c +
            grad_var_val * T(2) * x_centered /
                static_cast<T>(total_elements) +
            grad_mean_val / static_cast<T>(total_elements);
      }
    }
  });
#else
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t c = 0; c < channels; ++c) {
      T mean_val = mean(c, 0, 0, 0);
      T std_val_c = std_val(c, 0, 0, 0);
      T grad_var_val = grad_var(c, 0, 0, 0);
      T grad_mean_val = grad_mean(c, 0, 0, 0);

      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T x_centered = input(n, c, h, w) - mean_val;

          grad_input(n, c, h, w) =
              grad_normalized(n, c, h, w) / std_val_c +
              grad_var_val * T(2) * x_centered /
                  static_cast<T>(total_elements) +
              grad_mean_val / static_cast<T>(total_elements);
        }
      }
    }
  }
#endif

  return grad_input;
}

// Virtual method implementations
template <typename T>
std::string BatchNormLayer<T>::type() const {
  return "batchnorm";
}

template <typename T>
LayerConfig BatchNormLayer<T>::get_config() const {
  LayerConfig config;
  config.name = this->name_;
  config.parameters["num_features"] = num_features_;
  config.parameters["epsilon"] = epsilon_;
  config.parameters["momentum"] = momentum_;
  config.parameters["affine"] = affine_;
  return config;
}

template <typename T>
std::unique_ptr<Layer<T>> BatchNormLayer<T>::clone() const {
  return std::make_unique<BatchNormLayer<T>>(num_features_, epsilon_,
                                             momentum_, affine_, this->name_);
}

template <typename T>
std::vector<size_t> BatchNormLayer<T>::compute_output_shape(
    const std::vector<size_t> &input_shape) const {
  return input_shape; // Batch normalization doesn't change shape
}

// Protected method implementations
template <typename T>
void BatchNormLayer<T>::collect_parameters(std::vector<Tensor<T> *> &params) {
  if (affine_) {
    params.push_back(&gamma_);
    params.push_back(&beta_);
  }
}

template <typename T>
void BatchNormLayer<T>::collect_gradients(std::vector<Tensor<T> *> &grads) {
  if (affine_) {
    grads.push_back(&gamma_gradients_);
    grads.push_back(&beta_gradients_);
  }
}

template <typename T>
void BatchNormLayer<T>::update_parameters_impl(Optimizer<T> &optimizer) {
  std::vector<Tensor<T> *> params = this->parameters();
  std::vector<Tensor<T> *> grads = this->gradients();
  if (params.size() != grads.size()) {
    throw std::runtime_error(
        "Parameter and gradient size mismatch in BatchNormLayer");
  }
  optimizer.update(params, grads);
}

template <typename T>
std::unique_ptr<Layer<T>>
BatchNormLayer<T>::create_from_config(const LayerConfig &config) {
  size_t num_features = config.get<size_t>("num_features");
  T epsilon = config.get<T>("epsilon");
  T momentum = config.get<T>("momentum");
  bool affine = config.get<bool>("affine");

  return std::make_unique<BatchNormLayer<T>>(num_features, epsilon, momentum,
                                             affine, config.name);
}

// Explicit template instantiations
template class BatchNormLayer<float>;
template class BatchNormLayer<double>;

} // namespace tnn
