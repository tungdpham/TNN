/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/common/conv2d.hpp"
#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_CUDNN
#include "cuda/cudnn_conv2d_ops.hpp"
#include "device/task.hpp"
#endif
#ifdef USE_DNNL
#include "nn/layers_impl/cpu/dnnl_conv2d_ops.hpp"
#endif
#include <memory>
#include <string>
#include <unordered_map>

namespace synet {

// [N, H, W, C] input
class Conv2DLayerImpl : public SISOLayerImpl {
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
                                            ConvolutionStats &stats, const Tensor &input,
                                            Tensor &output, const Tensor &weights,
                                            const Tensor &bias, Tensor &workspace,
                                            size_t batch_size, size_t input_h, size_t input_w,
                                            size_t output_h, size_t output_w,
                                            flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> conv2d_backward_weights_and_bias_task(
      cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats, const Tensor &input,
      const Tensor &grad_output, Tensor &weight_gradients, Tensor &bias_gradients,
      Tensor &workspace, size_t batch_size, size_t input_h, size_t input_w, size_t output_h,
      size_t output_w, flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> conv2d_backward_data_task(
      cuda::cudnn_conv2d::feHandle_t *fe_handle, ConvolutionStats &stats, const Tensor &grad_output,
      const Tensor &weights, Tensor &grad_input, Tensor &workspace, size_t batch_size,
      size_t input_h, size_t input_w, size_t output_h, size_t output_w, flowHandle_t handle) const;

  Tensor cudnn_forward(const Tensor &input, Residuals &residuals);
  Tensor cudnn_backward(const Tensor &current_gradient, Residuals &residuals);

  mutable std::unordered_map<size_t, cuda::cudnn_conv2d::feHandle_t *> fe_handle_cache;
  mutable std::unordered_map<size_t, ConvolutionStats> stats_cache;
#endif

#ifdef USE_DNNL
  void build_dnnl_handle(const Vec<size_t> &input_shape) const;
  Tensor dnnl_forward(const Tensor &input, Residuals &residuals);
  Tensor dnnl_backward(const Tensor &grad_output, Residuals &residuals);

  mutable std::unordered_map<size_t, cpu::dnnl_conv2d::dnnlHandle_t *> dnnl_handle_cache;
  mutable std::unordered_map<size_t, ConvolutionStats> dnnl_stats_cache;
#endif

  Tensor def_forward(const Tensor &input, Residuals &residuals);
  Tensor def_backward(const Tensor &grad_output, Residuals &residuals);

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, Residuals &residuals) override;
  Tensor backward_impl(const Tensor &grad_output, Residuals &residuals) override;

public:
  static constexpr const char *TYPE_NAME = "conv2d";

  Conv2DLayerImpl(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
                  size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                  bool use_bias = true, const std::string &name = "conv2d");

  ~Conv2DLayerImpl();

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
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
  static std::shared_ptr<Conv2DLayerImpl> create_from_config(const LayerConfig &config);
};

class Conv2DLayer : public LayerRef<Conv2DLayerImpl> {
public:
  Conv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
              size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
              bool use_bias = true, const std::string &name = "conv2d")
      : LayerRef(std::make_shared<Conv2DLayerImpl>(in_channels, out_channels, kernel_h, kernel_w,
                                                   stride_h, stride_w, pad_h, pad_w, use_bias,
                                                   name)) {}

  using LayerRef<Conv2DLayerImpl>::LayerRef;
};

}  // namespace synet
