/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "nn/layers_impl/common/layer_norm.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDNN
#include "cuda/cudnn_layer_norm_ops.hpp"
#include "device/task.hpp"
#endif

namespace synet {

class LayerNormLayerImpl : public SISOLayerImpl {
private:
  size_t normalized_shape_;  // Size of C (channels)
  float epsilon_;
  bool affine_;  // Whether to use learnable affine parameters

  Tensor gamma_;
  Tensor beta_;
  Tensor gamma_gradients_;
  Tensor beta_gradients_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_forward(const Tensor &input, Tensor &output, const Tensor &gamma,
                                    const Tensor &beta, size_t batch_size, size_t channels,
                                    flowHandle_t handle = defaultFlowHandle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_backward(const Tensor &grad_output, const Tensor &input,
                                     const Tensor &gamma, Tensor &grad_input,
                                     Tensor &gamma_gradients, Tensor &beta_gradients,
                                     size_t batch_size, size_t channels,
                                     flowHandle_t handle = defaultFlowHandle) const;

#ifdef USE_CUDNN
  void build_graph(const Vec<size_t> &input_shape) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_run_forward(cuda::cudnn_layer_norm::feHandle_t *fe_handle,
                                          LayerNormStats &stats, const Tensor &input,
                                          Tensor &output, const Tensor &gamma, const Tensor &beta,
                                          Tensor &mean, Tensor &inv_variance, Tensor &workspace,
                                          size_t batch_size, size_t channels,
                                          flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_run_backward(cuda::cudnn_layer_norm::feHandle_t *fe_handle,
                                           LayerNormStats &stats, const Tensor &grad_output,
                                           const Tensor &input, const Tensor &gamma,
                                           Tensor &grad_input, Tensor &gamma_gradients,
                                           Tensor &beta_gradients, const Tensor &mean,
                                           const Tensor &inv_variance, Tensor &workspace,
                                           size_t batch_size, size_t channels,
                                           flowHandle_t handle) const;

  Tensor cudnn_forward(const Tensor &input, Residuals &residuals);
  Tensor cudnn_backward(const Tensor &grad_output, Residuals &residuals);

  mutable std::unordered_map<size_t, cuda::cudnn_layer_norm::feHandle_t *> fe_handle_cache;
#endif
  mutable std::unordered_map<size_t, LayerNormStats> stats_cache;

  Tensor def_forward(const Tensor &input, Residuals &residuals);
  Tensor def_backward(const Tensor &grad_output, Residuals &residuals);

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  explicit LayerNormLayerImpl(size_t normalized_shape, float epsilon = 1e-5f, bool affine = true,
                              const std::string &name = "layer_norm");

  ~LayerNormLayerImpl();

  static constexpr const char *TYPE_NAME = "layer_norm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override {
    return input_shape;
  }

  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    if (affine_) {
      auto gamma_desc = ParamDescriptor{
          param_dtype_,
          {normalized_shape_},
          &gamma_,
          &gamma_gradients_,
      };
      descriptors.push_back(gamma_desc);
      auto beta_desc = ParamDescriptor{
          param_dtype_,
          {normalized_shape_},
          &beta_,
          &beta_gradients_,
      };
      descriptors.push_back(beta_desc);
    }
    return descriptors;
  }

  static std::shared_ptr<LayerNormLayerImpl> create_from_config(const LayerConfig &config);
};

class LayerNormLayer : public LayerRef<LayerNormLayerImpl> {
public:
  explicit LayerNormLayer(size_t normalized_shape, float epsilon = 1e-5f, bool affine = true,
                          const std::string &name = "layer_norm")
      : LayerRef(std::make_shared<LayerNormLayerImpl>(normalized_shape, epsilon, affine, name)) {}

  using LayerRef<LayerNormLayerImpl>::LayerRef;
};

}  // namespace synet
