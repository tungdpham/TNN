/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "nn/layers_impl/common/batchnorm.hpp"
#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"

#ifdef USE_CUDNN
#include <cudnn.h>

#include "cuda/cudnn_batchnorm_ops.hpp"
#endif

namespace tnn {

class BatchNormLayer : public ParameterizedLayer {
private:
  size_t num_features_;
  float epsilon_;
  float momentum_;
  bool affine_;
  bool use_relu_;

  Tensor gamma_;
  Tensor beta_;
  Tensor gamma_gradients_;
  Tensor beta_gradients_;

  Tensor running_mean_;
  Tensor running_var_;
  Tensor dummy_mean_gradients_;
  Tensor dummy_var_gradients_;

  mutable std::unordered_map<size_t, BatchNormStats> stats_cache;

#ifdef USE_CUDNN
  void build_graph(const Vec<size_t> &input_shape) const;

  mutable std::unordered_map<size_t, cuda::cudnn_batchnorm::feHandle_t *> fe_handle_cache;
  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> forward_training_task(
      cuda::cudnn_batchnorm::feHandle_t *fe_handle, BatchNormStats &stats, const ConstTensor &input,
      const Tensor &output, const ConstTensor &gamma, const ConstTensor &beta,
      const Tensor &prev_running_mean, const Tensor &prev_running_var,
      const Tensor &next_running_mean, const Tensor &next_running_var, const Tensor &batch_mean,
      const Tensor &batch_invar, const Tensor &relu_mask, const Tensor &workspace,
      flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> forward_inference_task(cuda::cudnn_batchnorm::feHandle_t *fe_handle,
                                               BatchNormStats &stats, const ConstTensor &input,
                                               const Tensor &output, const ConstTensor &gamma,
                                               const ConstTensor &beta,
                                               const ConstTensor &saved_mean,
                                               const ConstTensor &saved_var,
                                               const Tensor &workspace, flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> backward_task(cuda::cudnn_batchnorm::feHandle_t *fe_handle,
                                      BatchNormStats &stats, const ConstTensor &grad_output,
                                      const ConstTensor &relu_mask, const ConstTensor &input,
                                      const Tensor &grad_input, const ConstTensor &gamma,
                                      const Tensor &gamma_gradients, const Tensor &beta_gradients,
                                      const ConstTensor &batch_mean, const ConstTensor &batch_var,
                                      const Tensor &workspace, flowHandle_t handle);

  Tensor cudnn_forward(const ConstTensor &input, size_t mb_id);
  Tensor cudnn_backward(const ConstTensor &grad_output, size_t mb_id);
#endif

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
  explicit BatchNormLayer(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                          bool affine = true, bool use_relu = false,
                          const std::string &name = "batchnorm");
  ~BatchNormLayer() override;

  static constexpr const char *TYPE_NAME = "batchnorm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  static std::unique_ptr<BatchNormLayer> create_from_config(const LayerConfig &config);

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
};

}  // namespace tnn
