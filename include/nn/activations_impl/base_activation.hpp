/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tensor/tensor.hpp"

namespace tnn {
template <typename T = float> class ActivationFunction {
public:
  virtual ~ActivationFunction() = default;

  virtual void apply(Tensor<T> &tensor) const = 0;
  virtual void apply_with_bias(Tensor<T> &tensor, const Tensor<T> &bias) const = 0;
  virtual void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const = 0;

  virtual Tensor<T> compute_gradient(const Tensor<T> &input,
                                     const Tensor<T> *upstream_gradient = nullptr) const = 0;
  virtual void compute_gradient_inplace(const Tensor<T> &input,
                                        Tensor<T> &upstream_gradient) const = 0;

  virtual void apply_channel_wise(Tensor<T> &tensor, int channel) const = 0;
  virtual void apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                                            const std::vector<T> &bias) const = 0;

  virtual void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const = 0;

  virtual std::string name() const = 0;
  virtual std::unique_ptr<ActivationFunction<T>> clone() const = 0;

  virtual void apply_spatial(Tensor<T> &tensor, int batch, int channel, int height,
                             int width) const {
    T &value = tensor(batch, channel, height, width);
    apply_single_value(value);
  }

protected:
  virtual void apply_single_value(T &value) const = 0;
  virtual T compute_single_gradient(T pre_activation_value) const = 0;
};
} // namespace tnn