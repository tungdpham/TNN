/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/layers_impl/common/layer_norm.hpp"
#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDNN
#include "cuda/cudnn_layer_norm_ops.hpp"
#include "device/task.hpp"
#endif
#include <unordered_map>

namespace tnn {

class LayerNormLayer : public ParameterizedLayer {
private:
  size_t normalized_shape_;  // Size of C (channels)
  float epsilon_;
  bool affine_;  // Whether to use learnable affine parameters

  Tensor gamma_;
  Tensor beta_;
  Tensor gamma_gradients_;
  Tensor beta_gradients_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> layer_norm_forward(const ConstTensor &input, const Tensor &output,
                                           const ConstTensor &gamma, const ConstTensor &beta,
                                           size_t batch_size, size_t channels,
                                           flowHandle_t handle = defaultFlowHandle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> layer_norm_backward(const ConstTensor &grad_output,
                                            const ConstTensor &input, const ConstTensor &gamma,
                                            const Tensor &grad_input, const Tensor &gamma_gradients,
                                            const Tensor &beta_gradients, size_t batch_size,
                                            size_t channels,
                                            flowHandle_t handle = defaultFlowHandle) const;

#ifdef USE_CUDNN
  void build_graph(const Vec<size_t> &input_shape) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_layer_norm_forward(cuda::cudnn_layer_norm::feHandle_t *fe_handle,
                                                 LayerNormStats &stats, const ConstTensor &input,
                                                 const Tensor &output, const ConstTensor &gamma,
                                                 const ConstTensor &beta, const Tensor &mean,
                                                 const Tensor &inv_variance,
                                                 const Tensor &workspace, size_t batch_size,
                                                 size_t channels, flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_layer_norm_backward(
      cuda::cudnn_layer_norm::feHandle_t *fe_handle, LayerNormStats &stats,
      const ConstTensor &grad_output, const ConstTensor &input, const ConstTensor &gamma,
      const Tensor &grad_input, const Tensor &gamma_gradients, const Tensor &beta_gradients,
      const ConstTensor &mean, const ConstTensor &inv_variance, const Tensor &workspace,
      size_t batch_size, size_t channels, flowHandle_t handle) const;

  void cudnn_forward(const ConstTensor &input, const Tensor &output, size_t mb_id);
  void cudnn_backward(const ConstTensor &grad_output, const Tensor &grad_input, size_t mb_id);

  mutable std::unordered_map<size_t, cuda::cudnn_layer_norm::feHandle_t *> fe_handle_cache;
#endif
  mutable std::unordered_map<size_t, LayerNormStats> stats_cache;

  void def_forward(const ConstTensor &input, const Tensor &output, size_t mb_id = 0);
  void def_backward(const ConstTensor &grad_output, const Tensor &grad_input, size_t mb_id = 0);

  std::vector<ParamDescriptor> param_descriptors() override {
    std::vector<ParamDescriptor> descriptors;
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

  void init_impl() override;
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  explicit LayerNormLayer(size_t normalized_shape, float epsilon = 1e-5f, bool affine = true,
                          const std::string &name = "layer_norm");

  ~LayerNormLayer();

  static constexpr const char *TYPE_NAME = "layer_norm";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override {
    return input_shape;
  }
  size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  static std::unique_ptr<LayerNormLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
