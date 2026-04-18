/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "parameterized_layer.hpp"

namespace tnn {

class LegacyBatchNormLayer : public ParameterizedLayer {
private:
  size_t num_features_;
  float epsilon_;
  float momentum_;
  bool affine_;

  Tensor gamma_;
  Tensor beta_;
  Tensor gamma_gradients_;
  Tensor beta_gradients_;

  Tensor running_mean_;
  Tensor running_var_;
  Tensor dummy_mean_gradients_;
  Tensor dummy_var_gradients_;

  std::unique_ptr<Task> forward_task_;
  std::unique_ptr<Task> backward_task_;

  Tensor def_forward(const ConstTensor &input, size_t mb_id);
  Tensor def_backward(const ConstTensor &grad_output, size_t mb_id);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_inference_impl(const ConstTensor &input, const Tensor &output,
                                           size_t batch_size, size_t channels, size_t spatial_size,
                                           flowHandle_t handle = defaultFlowHandle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_inference(const ConstTensor &input, const Tensor &output,
                                      size_t batch_size, size_t channels, size_t spatial_size,
                                      flowHandle_t handle = defaultFlowHandle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_forward(const ConstTensor &input, const Tensor &batch_mean,
                                    const Tensor &batch_inv_std, const Tensor &running_mean,
                                    const Tensor &running_var, const ConstTensor &gamma,
                                    const ConstTensor &beta, const Tensor &output,
                                    const Tensor &norm, size_t batch_size, size_t channels,
                                    size_t spatial_size, flowHandle_t handle = defaultFlowHandle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_backward(const ConstTensor &grad_output, const ConstTensor &norm_input,
                                     const ConstTensor &inv_std, const ConstTensor &gamma,
                                     const Tensor &d_gamma, const Tensor &d_beta,
                                     const Tensor &grad_input, size_t batch_size, size_t channels,
                                     size_t spatial_size, flowHandle_t handle = defaultFlowHandle);

  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    auto gamma_desc = ParamDescriptor{
        param_dtype_,
        {num_features_},
        &gamma_,
        &gamma_gradients_,
    };
    descriptors.push_back(gamma_desc);
    auto beta_desc = ParamDescriptor{
        param_dtype_,
        {num_features_},
        &beta_,
        &beta_gradients_,
    };
    descriptors.push_back(beta_desc);
    auto running_mean_desc = ParamDescriptor{
        param_dtype_,
        {num_features_},
        &running_mean_,
        &dummy_mean_gradients_,
    };
    descriptors.push_back(running_mean_desc);
    auto running_var_desc = ParamDescriptor{
        param_dtype_,
        {num_features_},
        &running_var_,
        &dummy_var_gradients_,
    };
    descriptors.push_back(running_var_desc);
    return descriptors;
  }

  void init_impl() override;
  Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) override;

public:
  explicit LegacyBatchNormLayer(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                                bool affine = true, const std::string &name = "batchnorm");

  static constexpr const char *TYPE_NAME = "legacy_batchnorm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  static std::unique_ptr<LegacyBatchNormLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
