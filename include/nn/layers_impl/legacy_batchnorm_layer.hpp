/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/siso_layer.hpp"

namespace synet {

class LegacyBatchNormLayerImpl : public SISOLayerImpl {
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

  Tensor def_forward(const Tensor &input, Residuals &residuals);
  Tensor def_backward(const Tensor &grad_output, Residuals &residuals);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_inference_impl(const Tensor &input, Tensor &output, size_t batch_size,
                                           size_t channels, size_t spatial_size,
                                           flowHandle_t handle = defaultFlowHandle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_inference(const Tensor &input, Tensor &output, size_t batch_size,
                                      size_t channels, size_t spatial_size,
                                      flowHandle_t handle = defaultFlowHandle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_forward(const Tensor &input, Tensor &batch_mean, Tensor &batch_inv_std,
                                    Tensor &running_mean, Tensor &running_var, const Tensor &gamma,
                                    const Tensor &beta, Tensor &output, Tensor &norm,
                                    size_t batch_size, size_t channels, size_t spatial_size,
                                    flowHandle_t handle = defaultFlowHandle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_backward(const Tensor &grad_output, const Tensor &norm_input,
                                     const Tensor &inv_std, const Tensor &gamma, Tensor &d_gamma,
                                     Tensor &d_beta, Tensor &grad_input, size_t batch_size,
                                     size_t channels, size_t spatial_size,
                                     flowHandle_t handle = defaultFlowHandle);

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit LegacyBatchNormLayerImpl(size_t num_features, float epsilon = 1e-5f,
                                    float momentum = 0.1f, bool affine = true,
                                    const std::string &name = "batchnorm");

  static constexpr const char *TYPE_NAME = "legacy_batchnorm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;

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

  static std::shared_ptr<LegacyBatchNormLayerImpl> create_from_config(const LayerConfig &config);
};

class LegacyBatchNormLayer : public LayerRef<LegacyBatchNormLayerImpl> {
public:
  explicit LegacyBatchNormLayer(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                                bool affine = true, const std::string &name = "batchnorm")
      : LayerRef(std::make_shared<LegacyBatchNormLayerImpl>(num_features, epsilon, momentum, affine,
                                                            name)) {}

  using LayerRef<LegacyBatchNormLayerImpl>::LayerRef;
};

}  // namespace synet
