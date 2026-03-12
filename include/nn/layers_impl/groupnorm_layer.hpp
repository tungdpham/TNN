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

#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class GroupNormLayer : public ParameterizedLayer {
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
  std::unique_ptr<Task> run_forward_fused(const ConstTensor &input, const Tensor &group_mean,
                                          const Tensor &group_inv_std, const ConstTensor &gamma,
                                          const ConstTensor &beta, const Tensor &output,
                                          const Tensor &norm_cache, size_t batch_size,
                                          size_t channels, size_t spatial_size,
                                          flowHandle_t handle = defaultFlowHandle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_backward_fused(const ConstTensor &grad_output,
                                           const ConstTensor &norm_input,
                                           const ConstTensor &inv_std, const ConstTensor &gamma,
                                           const Tensor &d_gamma, const Tensor &d_beta,
                                           const Tensor &grad_input, size_t batch_size,
                                           size_t channels, size_t spatial_size,
                                           flowHandle_t handle = defaultFlowHandle) const;

  std::vector<ParamDescriptor> param_descriptors() override {
    std::vector<ParamDescriptor> descriptors;
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

  void init_impl() override;
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  GroupNormLayer(size_t num_groups, size_t num_channels, float epsilon = 1e-5f, bool affine = true,
                 const std::string &name = "groupnorm");

  static constexpr const char *TYPE_NAME = "groupnorm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  static std::unique_ptr<GroupNormLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
