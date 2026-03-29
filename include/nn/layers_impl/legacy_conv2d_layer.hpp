/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#ifdef USE_CUDNN
#include "nn/layers_impl/cuda/cudnn_conv2d_nchw_ops.hpp"
#endif
#include <memory>
#include <string>
#include <unordered_map>

#include "nn/layers_impl/common/conv2d.hpp"
#include "parameterized_layer.hpp"

namespace tnn {

class LegacyConv2DLayer : public ParameterizedLayer {
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

  Tensor def_forward(const ConstTensor &input, size_t mb_id);
  Tensor def_backward(const ConstTensor &current_gradient, size_t mb_id);

#ifdef USE_CUDNN
  Tensor cudnn_forward(const ConstTensor &input, size_t mb_id);
  Tensor cudnn_backward(const ConstTensor &grad_output, size_t mb_id);
#endif

  std::unordered_map<size_t, Vec<size_t>> micro_batch_input_shapes_;
  std::unordered_map<size_t, Tensor> micro_batch_col_buffers_;

#ifdef USE_CUDNN
  void build_graph(const Vec<size_t> &input_shape) const;

  mutable std::unordered_map<size_t, cuda::cudnn_conv2d::ConvolutionHandle *>
      convolution_handle_cache;
  mutable std::unordered_map<size_t, ConvolutionStats> stats_cache;
#endif

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_forward(const ConstTensor &col_data, const ConstTensor &weight_data,
                                    const Tensor &output_data, const size_t output_size,
                                    const size_t kernel_size, const size_t out_channels,
                                    flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> add_bias(const Tensor &output_data, const ConstTensor &bias_data,
                                 const size_t batch_size, const size_t output_h,
                                 const size_t output_w, const size_t out_channels,
                                 flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_wgrad(const ConstTensor &col_data, const ConstTensor &gradient_data,
                                  const Tensor &weight_grad_data, const size_t output_size,
                                  const size_t kernel_size, const size_t out_channels,
                                  flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_dgrad(const ConstTensor &gradient_data, const ConstTensor &weight_data,
                                  const Tensor &col_grad_data, const size_t output_size,
                                  const size_t kernel_size, const size_t out_channels,
                                  flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_bgrad(const ConstTensor &gradient_data, const Tensor &bias_grad_data,
                                  const size_t batch_size, const size_t output_h,
                                  const size_t output_w, const size_t out_channels,
                                  flowHandle_t handle);

  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    auto weight_desc = ParamDescriptor{
        param_dtype_,
        {in_channels_, out_channels_, kernel_h_, kernel_w_},
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

#ifdef USE_CUDNN
  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_run_forward(const ConstTensor &input, const ConstTensor &weight,
                                          const ConstTensor bias, const Tensor &output,
                                          size_t batch_size, size_t input_h, size_t input_w,
                                          size_t output_h, size_t output_w,
                                          const Tensor &cudnn_workspace, flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_run_dgrad(const ConstTensor &grad_output, const ConstTensor &weight,
                                        const Tensor &input_grad, size_t batch_size, size_t input_h,
                                        size_t input_w, size_t output_h, size_t output_w,
                                        const Tensor &cudnn_workspace, flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_run_wgrad(const ConstTensor &input, const ConstTensor &grad_output,
                                        const Tensor &weight_grad, size_t batch_size,
                                        size_t input_h, size_t input_w, size_t output_h,
                                        size_t output_w, const Tensor &cudnn_workspace,
                                        flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_run_bgrad(const ConstTensor &grad_output, const Tensor &bias_grad,
                                        size_t batch_size, size_t output_h, size_t output_w,
                                        size_t out_channels, flowHandle_t handle);
#endif

  Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) override;

public:
  LegacyConv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
                    size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                    bool use_bias = true, const std::string &name = "legacy_conv2d");

  ~LegacyConv2DLayer();

  static constexpr const char *TYPE_NAME = "legacy_conv2d";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;

  static std::unique_ptr<LegacyConv2DLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
