/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layer.hpp"
#include "nn/layers_impl/common/conv2d.hpp"
#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDNN
#include "cuda/cudnn_conv2d_ops.hpp"
#include "device/task.hpp"
#endif
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

// [N, H, W, C] input
class Conv2DLayer : public ParameterizedLayer {
private:
  size_t in_channels_;
  size_t out_channels_;
  size_t kernel_h_;
  size_t kernel_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;
  bool use_bias_;

  Tensor weights_;
  Tensor bias_;
  Tensor weight_gradients_;
  Tensor bias_gradients_;

#ifdef USE_CUDNN
  void build_graph(const Vec<size_t> &input_shape) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> conv2d_forward_task(cuda::cudnn_conv2d::feHandle_t *fe_handle,
                                            ConvolutionStats &stats, const ConstTensor &input,
                                            const Tensor &output, const ConstTensor &weights,
                                            const ConstTensor &bias, const Tensor &workspace,
                                            size_t batch_size, size_t input_h, size_t input_w,
                                            size_t output_h, size_t output_w,
                                            flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> conv2d_backward_weights_and_bias_task(
      cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats, const ConstTensor &input,
      const ConstTensor &grad_output, const Tensor &weight_gradients, const Tensor &bias_gradients,
      const Tensor &workspace, size_t batch_size, size_t input_h, size_t input_w, size_t output_h,
      size_t output_w, flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> conv2d_backward_data_task(
      cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats,
      const ConstTensor &grad_output, const ConstTensor &weights, const Tensor &grad_input,
      const Tensor &workspace, size_t batch_size, size_t input_h, size_t input_w, size_t output_h,
      size_t output_w, flowHandle_t handle) const;

  void cudnn_forward(const ConstTensor &input, const Tensor &output, size_t mb_id);
  void cudnn_backward(const ConstTensor &current_gradient, const Tensor &grad_input, size_t mb_id);

  mutable std::unordered_map<size_t, cuda::cudnn_conv2d::feHandle_t *> fe_handle_cache;
  mutable std::unordered_map<size_t, ConvolutionStats> stats_cache;
#endif

  void def_forward(const ConstTensor &input, const Tensor &output, size_t mb_id);
  void def_backward(const ConstTensor &grad_output, const Tensor &grad_input, size_t mb_id);

  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    auto weight_desc = ParamDescriptor{
        param_dtype_,
        {out_channels_, kernel_h_, kernel_w_, in_channels_},
        &weights_,
        &weight_gradients_,
    };
    descriptors.push_back(weight_desc);
    if (use_bias_) {
      auto bias_desc = ParamDescriptor{
          param_dtype_,
          {out_channels_},
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
  static constexpr const char *TYPE_NAME = "conv2d";

  Conv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
              size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
              bool use_bias = true, const std::string &name = "conv2d");

  ~Conv2DLayer();

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;

  static std::unique_ptr<Conv2DLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
