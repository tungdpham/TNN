/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "../activations.hpp"
#include "../optimizers.hpp"
#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

template <typename T = float> class DenseLayer : public ParameterizedLayer<T> {
private:
  size_t input_features_;
  size_t output_features_;
  bool use_bias_;
  std::unique_ptr<ActivationFunction<T>> activation_;
  Tensor<T> weights_;
  Tensor<T> bias_;
  Tensor<T> weight_gradients_;
  Tensor<T> bias_gradients_;

  std::unordered_map<size_t, Tensor<T>> micro_batch_inputs_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_pre_activations_;

  void compute_dense_forward(const T *input_data, const T *weight_data,
                             T *output_data, const size_t batch_size,
                             const size_t input_features,
                             const size_t output_features) const;

  void compute_weight_gradients(const T *input_data, const T *gradient_data,
                                T *weight_grad_data, const size_t batch_size,
                                const size_t input_features,
                                const size_t output_features) const;

  void compute_input_gradients(const T *gradient_data, const T *weight_data,
                               T *grad_input_data, const size_t batch_size,
                               const size_t input_features,
                               const size_t output_features) const;

  void compute_bias_gradients(const T *current_grad_data, T *bias_gradient_data,
                              const size_t batch_size, const size_t output_features) const;

  void add_bias_vector(T *output_data, const T *bias_data, const size_t batch_size,
                       const size_t output_features) const;

public:
  DenseLayer(size_t input_features, size_t output_features,
             std::unique_ptr<ActivationFunction<T>> activation = nullptr,
             bool use_bias = true, const std::string &name = "dense");

  Tensor<T> forward(const Tensor<T> &input, size_t micro_batch_id = 0) override;
  Tensor<T> backward(const Tensor<T> &gradient,
                     size_t micro_batch_id = 0) override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override;

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override;
  void collect_gradients(std::vector<Tensor<T> *> &grads) override;
  void clear_gradients() override;

public:
  static std::unique_ptr<Layer<T>>
  create_from_config(const LayerConfig &config);
};

} // namespace tnn

#include "dense_layer.tpp"