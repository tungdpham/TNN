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
                                                 const ConstTensor &grad_output,
                                                 const Tensor &weight_grad, size_t batch_size,
                                                 size_t input_features, size_t output_features,
                                                 flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_input_gradients(const ConstTensor &grad_output,
                                                const ConstTensor &weights,
                                                const Tensor &grad_input, size_t batch_size,
                                                size_t input_features, size_t output_features,
                                                flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_bias_gradients(const ConstTensor &grad_output,
                                               const Tensor &bias_gradient, size_t batch_size,
                                               size_t output_features, flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> add_bias_vector(const Tensor &output, const ConstTensor &bias,
                                        size_t batch_size, size_t output_features,
                                        flowHandle_t handle) const;

  std::vector<ParamDescriptor> param_descriptors() override {
    std::vector<ParamDescriptor> descriptors;
    auto weight_desc = ParamDescriptor{
        param_dtype_,
        {output_features_, input_features_},
        &weights_,
        &weight_gradients_,
    };
    descriptors.push_back(weight_desc);
    if (use_bias_) {
      auto bias_desc = ParamDescriptor{
          param_dtype_,
          {output_features_},
          &bias_,
          &bias_gradients_,
      };
      descriptors.push_back(bias_desc);
    }
    return descriptors;
  }

  void init_impl() override;
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  LegacyDenseLayer(size_t input_features, size_t output_features, bool use_bias = true,
                   const std::string &name = "legacy_dense");

  static constexpr const char *TYPE_NAME = "legacy_dense";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

  static std::unique_ptr<LegacyDenseLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
