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
#include "nn/siso_layer.hpp"

namespace synet {

class LegacyConv2DLayerImpl : public SISOLayerImpl {
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

  Tensor def_forward(const Tensor &input, Residuals &residuals);
  Tensor def_backward(const Tensor &current_gradient, Residuals &residuals);

#ifdef USE_CUDNN
  Tensor cudnn_forward(const Tensor &input, Residuals &residuals);
  Tensor cudnn_backward(const Tensor &grad_output, Residuals &residuals);
#endif

  std::unordered_map<size_t, Vec<size_t>> micro_batch_input_shapes_;

#ifdef USE_CUDNN
  void build_graph(const Vec<size_t> &input_shape) const;

  mutable std::unordered_map<size_t, cuda::cudnn_conv2d::ConvolutionHandle *>
      convolution_handle_cache;
  mutable std::unordered_map<size_t, ConvolutionStats> stats_cache;
#endif

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_forward(const Tensor &col_data, const Tensor &weight_data,
                                    Tensor &output_data, size_t output_size, size_t kernel_size,
                                    size_t out_channels, flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> add_bias(Tensor &output_data, const Tensor &bias_data, size_t batch_size,
                                 size_t output_h, size_t output_w, size_t out_channels,
                                 flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_wgrad(const Tensor &col_data, const Tensor &gradient_data,
                                  Tensor &weight_grad_data, size_t output_size, size_t kernel_size,
                                  size_t out_channels, flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_dgrad(const Tensor &gradient_data, const Tensor &weight_data,
                                  Tensor &col_grad_data, size_t output_size, size_t kernel_size,
                                  size_t out_channels, flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_bgrad(const Tensor &gradient_data, Tensor &bias_grad_data,
                                  size_t batch_size, size_t output_h, size_t output_w,
                                  size_t out_channels, flowHandle_t handle);

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
  std::unique_ptr<Task> cudnn_run_forward(const Tensor &input, const Tensor &weight,
                                          const Tensor bias, Tensor &output, size_t batch_size,
                                          size_t input_h, size_t input_w, size_t output_h,
                                          size_t output_w, Tensor &cudnn_workspace,
                                          flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_run_dgrad(const Tensor &grad_output, const Tensor &weight,
                                        Tensor &input_grad, size_t batch_size, size_t input_h,
                                        size_t input_w, size_t output_h, size_t output_w,
                                        Tensor &cudnn_workspace, flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_run_wgrad(const Tensor &input, const Tensor &grad_output,
                                        Tensor &weight_grad, size_t batch_size, size_t input_h,
                                        size_t input_w, size_t output_h, size_t output_w,
                                        Tensor &cudnn_workspace, flowHandle_t handle);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_run_bgrad(const Tensor &grad_output, Tensor &bias_grad,
                                        size_t batch_size, size_t output_h, size_t output_w,
                                        size_t out_channels, flowHandle_t handle);
#endif

  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  LegacyConv2DLayerImpl(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
                        size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0,
                        size_t pad_w = 0, bool use_bias = true,
                        const std::string &name = "legacy_conv2d");

  ~LegacyConv2DLayerImpl();

  static constexpr const char *TYPE_NAME = "legacy_conv2d";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  static std::shared_ptr<LegacyConv2DLayerImpl> create_from_config(const LayerConfig &config);
};

class LegacyConv2DLayer : public LayerRef<LegacyConv2DLayerImpl> {
public:
  LegacyConv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
                    size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                    bool use_bias = true, const std::string &name = "legacy_conv2d")
      : LayerRef(std::make_shared<LegacyConv2DLayerImpl>(in_channels, out_channels, kernel_h,
                                                         kernel_w, stride_h, stride_w, pad_h, pad_w,
                                                         use_bias, name)) {}

  using LayerRef<LegacyConv2DLayerImpl>::LayerRef;
};

}  // namespace synet
