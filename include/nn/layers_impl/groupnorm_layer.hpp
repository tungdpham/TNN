/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"

#include <memory>
#include <string>
#include <vector>

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
  std::unique_ptr<Task> run_forward_fused(const Tensor &input, Tensor &group_mean,
                                          Tensor &group_inv_std, const Tensor &gamma,
                                          const Tensor &beta, Tensor &output, Tensor &norm_cache,
                                          size_t batch_size, size_t channels, size_t spatial_size,
                                          const std::string &flow_id = "default") const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_backward_fused(const Tensor &grad_output, const Tensor &norm_input,
                                           const Tensor &inv_std, const Tensor &gamma,
                                           Tensor &d_gamma, Tensor &d_beta, Tensor &grad_input,
                                           size_t batch_size, size_t channels, size_t spatial_size,
                                           const std::string &flow_id = "default") const;

  void init_params() override;
  void forward_impl(const Tensor &input, Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id = 0) override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;

public:
  GroupNormLayer(size_t num_groups, size_t num_channels, float epsilon = 1e-5f, bool affine = true,
                 const std::string &name = "groupnorm");

  static constexpr const char *TYPE_NAME = "groupnorm";

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  static std::unique_ptr<GroupNormLayer> create_from_config(const LayerConfig &config);
};

} // namespace tnn
