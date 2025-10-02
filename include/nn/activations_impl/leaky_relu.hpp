/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once
#include "nn/activations_impl/base_activation.hpp"
#include "tensor/tensor.hpp"

namespace tnn {
template <typename T = float> class LeakyReLU : public ActivationFunction<T> {
private:
  T negative_slope_;

public:
  explicit LeakyReLU(T negative_slope = T(0.01));

  void apply(Tensor<T> &tensor) const override;
  void apply_with_bias(Tensor<T> &tensor, const Tensor<T> &bias) const override;
  void apply_with_scalar_bias(Tensor<T> &tensor, T bias) const override;

  Tensor<T> compute_gradient(const Tensor<T> &pre_activation_values,
                             const Tensor<T> *upstream_gradient = nullptr) const override;
  void compute_gradient_inplace(const Tensor<T> &pre_activation_values,
                                Tensor<T> &upstream_gradient) const override;

  void apply_channel_wise(Tensor<T> &tensor, int channel) const override;
  void apply_channel_wise_with_bias(Tensor<T> &tensor, int channel,
                                    const std::vector<T> &bias) const override;
  void apply_batch_wise(Tensor<T> &tensor, int batch_idx) const override;

  std::string name() const override;
  std::unique_ptr<ActivationFunction<T>> clone() const override;

protected:
  void apply_single_value(T &value) const override;
  T compute_single_gradient(T pre_activation_value) const override;
};

} // namespace tnn