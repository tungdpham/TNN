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
#include "tensor/tensor.hpp"

namespace synet {

class GroupNormLayerImpl : public SISOLayerImpl {
private:
  size_t num_groups_;
  size_t num_channels_;
  float epsilon_;
  bool affine_;

  Tensor gamma_;
  Tensor beta_;
  Tensor gamma_gradients_;
  Tensor beta_gradients_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_forward(const Tensor &input, Tensor &group_mean, Tensor &group_inv_std,
                                    const Tensor &gamma, const Tensor &beta, Tensor &output,
                                    Tensor &norm_cache, size_t batch_size, size_t channels,
                                    size_t spatial_size,
                                    flowHandle_t handle = defaultFlowHandle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_backward(const Tensor &grad_output, const Tensor &norm_input,
                                     const Tensor &inv_std, const Tensor &gamma, Tensor &d_gamma,
                                     Tensor &d_beta, Tensor &grad_input, size_t batch_size,
                                     size_t channels, size_t spatial_size,
                                     flowHandle_t handle = defaultFlowHandle) const;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  GroupNormLayerImpl(size_t num_groups, size_t num_channels, float epsilon = 1e-5f,
                     bool affine = true, const std::string &name = "groupnorm");

  static constexpr const char *TYPE_NAME = "groupnorm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;

  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    if (affine_) {
      auto gamma_desc = ParamDescriptor{
          param_dtype_,
          {num_channels_},
          &gamma_,
          &gamma_gradients_,
      };
      descriptors.push_back(gamma_desc);
      auto beta_desc = ParamDescriptor{
          param_dtype_,
          {num_channels_},
          &beta_,
          &beta_gradients_,
      };
      descriptors.push_back(beta_desc);
    }
    return descriptors;
  }
  static std::shared_ptr<GroupNormLayerImpl> create_from_config(const LayerConfig &config);
};

class GroupNormLayer : public LayerRef<GroupNormLayerImpl> {
public:
  GroupNormLayer(size_t num_groups, size_t num_channels, float epsilon = 1e-5f, bool affine = true,
                 const std::string &name = "groupnorm")
      : LayerRef(std::make_shared<GroupNormLayerImpl>(num_groups, num_channels, epsilon, affine,
                                                      name)) {}

  using LayerRef<GroupNormLayerImpl>::LayerRef;
};

}  // namespace synet
