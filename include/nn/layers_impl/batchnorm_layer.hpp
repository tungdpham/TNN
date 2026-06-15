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
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

#ifdef USE_CUDNN
#include <cudnn.h>

#include "cuda/cudnn_batchnorm_ops.hpp"
#endif
#ifdef USE_DNNL
#include "nn/layers_impl/cpu/dnnl_batchnorm_ops.hpp"
#endif

namespace synet {

class BatchNormLayerImpl : public SISOLayerImpl {
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

#ifdef USE_DNNL
  void build_dnnl_handle(const Vec<size_t> &input_shape) const;
  Tensor dnnl_forward(const Tensor &input, Residuals &residuals);
  Tensor dnnl_backward(const Tensor &grad_output, Residuals &residuals);

  mutable std::unordered_map<size_t, cpu::dnnl_batchnorm::dnnlBNHandle_t *> dnnl_handle_cache;
  mutable std::unordered_map<size_t, BatchNormStats> dnnl_stats_cache;
#endif

#ifdef USE_CUDNN
  void build_graph(const Vec<size_t> &input_shape) const;

  mutable std::unordered_map<size_t, cuda::cudnn_batchnorm::feHandle_t *> fe_handle_cache;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> forward_training_task(
      cuda::cudnn_batchnorm::feHandle_t *fe_handle, BatchNormStats &stats, const Tensor &input,
      Tensor &output, const Tensor &gamma, const Tensor &beta, const Tensor &prev_running_mean,
      const Tensor &prev_running_var, Tensor &next_running_mean, Tensor &next_running_var,
      Tensor &batch_mean, Tensor &batch_invar, Tensor &relu_mask, Tensor &workspace,
      flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> forward_inference_task(cuda::cudnn_batchnorm::feHandle_t *fe_handle,
                                               BatchNormStats &stats, const Tensor &input,
                                               Tensor &output, const Tensor &gamma,
                                               const Tensor &beta, const Tensor &saved_mean,
                                               const Tensor &saved_var, Tensor &workspace,
                                               flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> backward_task(cuda::cudnn_batchnorm::feHandle_t *fe_handle,
                                      BatchNormStats &stats, const Tensor &grad_output,
                                      const Tensor &relu_mask, const Tensor &input,
                                      Tensor &grad_input, const Tensor &gamma,
                                      Tensor &gamma_gradients, Tensor &beta_gradients,
                                      const Tensor &batch_mean, const Tensor &batch_invar,
                                      Tensor &workspace, flowHandle_t handle);

  Tensor cudnn_forward(const Tensor &input, Residuals &residuals);
  Tensor cudnn_backward(const Tensor &grad_output, Residuals &residuals);
#endif

  Tensor def_forward(const Tensor &input, Residuals &residuals);
  Tensor def_backward(const Tensor &grad_output, Residuals &residuals);

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit BatchNormLayerImpl(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                              bool affine = true, bool use_relu = false,
                              const std::string &name = "batchnorm");
  ~BatchNormLayerImpl() override;

  static constexpr const char *TYPE_NAME = "batchnorm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::shared_ptr<BatchNormLayerImpl> create_from_config(const LayerConfig &config);
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
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
};

class BatchNormLayer : public LayerRef<BatchNormLayerImpl> {
public:
  BatchNormLayer(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                 bool affine = true, bool use_relu = false, const std::string &name = "batchnorm")
      : LayerRef(std::make_shared<BatchNormLayerImpl>(num_features, epsilon, momentum, affine,
                                                      use_relu, name)) {}

  using LayerRef<BatchNormLayerImpl>::LayerRef;
};

}  // namespace synet
