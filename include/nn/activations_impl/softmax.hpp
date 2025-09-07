/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once


template <typename T = float>
class Softmax : public ActivationFunction<T> {
public:
  void apply(Tensor<T> &tensor) const override {
    size_t batch_size = tensor.batch_size();
    
    size_t height = tensor.height();
    size_t width = tensor.width();

    utils::parallel_for_range<size_t>(0, batch_size, [&](size_t n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          apply_softmax_spatial(tensor, n, h, w);
        }
      }
    });
  }

  void apply_with_bias(Tensor<T> &tensor,
                       const Tensor<T> &bias) const override {
    if (tensor.shape() != bias.shape()) {
      throw std::invalid_argument("Tensor and bias must have the same shape");
    }

    
    T *data = tensor.data();
    const T *bias_data = bias.data();
    size_t size = tensor.size();

    utils::parallel_for_range<size_t>(0, size, [&](size_t i) {
      data[i] += bias_data[i];
    });

    
    apply(tensor);
  }

  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override {
    if (bias != T(0)) {
      T *data = tensor.data();
      size_t size = tensor.size();

      utils::parallel_for_range<size_t>(0, size, [&](size_t i) {
        data[i] += bias;
      });
    }

    apply(tensor);
  }

  Tensor<T> compute_gradient(
      const Tensor<T> &pre_activation_values,
      const Tensor<T> *upstream_gradient = nullptr) const override {
    Tensor<T> gradient(pre_activation_values.shape());

    
    Tensor<T> softmax_values = pre_activation_values; 
    apply(softmax_values); 

    size_t batch_size = pre_activation_values.batch_size();
    size_t channels = pre_activation_values.channels();
    size_t height = pre_activation_values.height();
    size_t width = pre_activation_values.width();

    if (upstream_gradient == nullptr) {
      throw std::invalid_argument(
          "Upstream gradient must be provided for softmax gradient computation");
    } else {
      
      if (upstream_gradient->shape() != pre_activation_values.shape()) {
        throw std::invalid_argument("Upstream gradient must have the same "
                                    "shape as pre-activation values");
      }

      utils::parallel_for_range<size_t>(0, batch_size, [&](size_t n) {
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            
            T dot_product = T(0);
            for (size_t j = 0; j < channels; ++j) {
              dot_product +=
                  softmax_values(n, j, h, w) * (*upstream_gradient)(n, j, h, w);
            }

            
            for (size_t i = 0; i < channels; ++i) {
              T s_i = softmax_values(n, i, h, w);
              T upstream_i = (*upstream_gradient)(n, i, h, w);
              gradient(n, i, h, w) = s_i * (upstream_i - dot_product);
            }
          }
        }
      });
    }

    return gradient;
  }

  void compute_gradient_inplace(
      const Tensor<T> &pre_activation_values,
      Tensor<T> &upstream_gradient) const override {
    size_t batch_size = pre_activation_values.batch_size();
    size_t channels = pre_activation_values.channels();
    size_t height = pre_activation_values.height();
    size_t width = pre_activation_values.width();

    
    if (upstream_gradient.shape() != pre_activation_values.shape()) {
      throw std::invalid_argument("Upstream gradient must have the same "
                                  "shape as pre-activation values");
    }

    
    Tensor<T> softmax_values = pre_activation_values; 
    apply(softmax_values); 

    utils::parallel_for_range<size_t>(0, batch_size, [&](size_t n) {
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          
          T dot_product = T(0);
          for (size_t j = 0; j < channels; ++j) {
            dot_product +=
                softmax_values(n, j, h, w) * upstream_gradient(n, j, h, w);
          }

          
          for (size_t i = 0; i < channels; ++i) {
            T s_i = softmax_values(n, i, h, w);
            T upstream_i = upstream_gradient(n, i, h, w);
            upstream_gradient(n, i, h, w) = s_i * (upstream_i - dot_product);
          }
        }
      }
    });
  }

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override {
    (void)tensor; 
    (void)channel; 
    
    
    throw std::runtime_error("Channel-wise softmax is not supported. Use full "
                             "tensor softmax instead.");
  }

  void apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                                    const std::vector<T> &bias) const override {
    (void)tensor; 
    (void)channel; 
    (void)bias; 
    throw std::runtime_error("Channel-wise softmax is not supported. Use full "
                             "tensor softmax instead.");
  }

  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override {
    if (batch_idx < 0 || batch_idx >= static_cast<int>(tensor.batch_size())) {
      throw std::invalid_argument("Batch index out of bounds");
    }

    
    size_t height = tensor.height();
    size_t width = tensor.width();

    utils::parallel_for_range<size_t>(0, height, [&](size_t h) {
      for (size_t w = 0; w < width; ++w) {
        apply_softmax_spatial(tensor, batch_idx, h, w);
      }
    });
  }

  
  
  void apply_spatial(Tensor<T> &tensor, int batch, int channel, int height,
                     int width) const override {
    (void)channel; 
    
    
    
    apply_softmax_spatial(tensor, batch, height, width);
  }

  std::string name() const override { return "softmax"; }

  std::unique_ptr<ActivationFunction<T>> clone() const override {
    return std::make_unique<Softmax<T>>();
  }

protected:
  void apply_single_value(T &value) const override {
    (void)value; 
    
    
    throw std::runtime_error("Single value softmax is not supported. Softmax "
                             "requires normalization across channels.");
  }

  T compute_single_gradient(T pre_activation_value) const override {
    (void)pre_activation_value; 
    
    
    throw std::runtime_error("Single value softmax gradient is not supported. "
                             "Use compute_gradient instead.");
  }

private:
  void apply_softmax_spatial(Tensor<T> &tensor, size_t n, size_t h,
                             size_t w) const {
    size_t channels = tensor.channels();

    
    T max_val = tensor(n, 0, h, w);
    for (size_t c = 1; c < channels; ++c) {
      T val = tensor(n, c, h, w);
      if (val > max_val)
        max_val = val;
    }

    
    T sum = T(0);
    for (size_t c = 0; c < channels; ++c) {
      T val = tensor(n, c, h, w);
      tensor(n, c, h, w) = std::exp(val - max_val);
      sum += tensor(n, c, h, w);
    }

    
    for (size_t c = 0; c < channels; ++c) {
      tensor(n, c, h, w) /= sum;
    }
  }
};