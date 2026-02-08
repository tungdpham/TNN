/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "device/task.hpp"
#include "parameterized_layer.hpp"

namespace tnn {

class LegacyDenseLayer : public ParameterizedLayer {
private:
  size_t input_features_;
  size_t output_features_;
  bool use_bias_;

  Tensor weights_;
  Tensor bias_;
  Tensor weight_gradients_;
  Tensor bias_gradients_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_dense_forward(const ConstTensor &input, const ConstTensor &weights,
                                              const Tensor &output, size_t batch_size,
                                              size_t input_features, size_t output_features,
                                              flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_weight_gradients(const ConstTensor &input,
                                                 const ConstTensor &gradient,
                                                 const Tensor &weight_grad, size_t batch_size,
                                                 size_t input_features, size_t output_features,
                                                 flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_input_gradients(const ConstTensor &gradient,
                                                const ConstTensor &weights,
                                                const Tensor &grad_input, size_t batch_size,
                                                size_t input_features, size_t output_features,
                                                flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_bias_gradients(const ConstTensor &gradient,
                                               const Tensor &bias_gradient, size_t batch_size,
                                               size_t output_features, flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> add_bias_vector(const Tensor &output, const ConstTensor &bias,
                                        size_t batch_size, size_t output_features,
                                        flowHandle_t handle) const;

  void init_params() override;
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &gradient, const Tensor &grad_input,
                     size_t mb_id = 0) override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;

public:
  LegacyDenseLayer(size_t input_features, size_t output_features, bool use_bias = true,
                   const std::string &name = "legacy_dense");

  static constexpr const char *TYPE_NAME = "legacy_dense";

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

  static std::unique_ptr<LegacyDenseLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
