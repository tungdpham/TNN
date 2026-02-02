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
#include <vector>

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

  void def_forward(const ConstTensor &input, Tensor &output, size_t mb_id);
  void def_backward(const ConstTensor &current_gradient, Tensor &grad_input, size_t mb_id);

#ifdef USE_CUDNN
  void cudnn_forward(const ConstTensor &input, Tensor &output, size_t mb_id);
  void cudnn_backward(const ConstTensor &gradient, Tensor &grad_input, size_t mb_id);
#endif

  std::unordered_map<size_t, std::vector<size_t>> micro_batch_input_shapes_;
  std::unordered_map<size_t, Tensor> micro_batch_col_buffers_;

  Tensor temp_output_buffer_;
  Tensor temp_gradient_buffer_;
  Tensor temp_col_grad_matrix_buffer_;

  ConvolutionStats stats_;
#ifdef USE_CUDNN
  cuda::cudnn_conv2d::ConvolutionHandle *convolution_handle_ = nullptr;
  size_t max_workspace_ = 0;
#endif

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_conv_forward_impl(const ConstTensor &col_data,
                                                  const ConstTensor &weight_data,
                                                  Tensor &output_data, const size_t output_size,
                                                  const size_t kernel_size,
                                                  const size_t out_channels,
                                                  const std::string &flow_id);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> add_bias_to_output_impl(Tensor &output_data, const ConstTensor &bias_data,
                                                const size_t batch_size, const size_t output_h,
                                                const size_t output_w, const size_t out_channels,
                                                const std::string &flow_id) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_weight_gradients_impl(
      const ConstTensor &col_data, const ConstTensor &gradient_data, Tensor &weight_grad_data,
      const size_t output_size, const size_t kernel_size, const size_t out_channels,
      const std::string &flow_id);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_input_gradients_impl(
      const ConstTensor &gradient_data, const ConstTensor &weight_data, Tensor &col_grad_data,
      const size_t output_size, const size_t kernel_size, const size_t out_channels,
      const std::string &flow_id) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_bias_gradients_impl(const ConstTensor &gradient_data,
                                                    Tensor &bias_grad_data, const size_t batch_size,
                                                    const size_t output_h, const size_t output_w,
                                                    const size_t out_channels,
                                                    const std::string &flow_id);

#ifdef USE_CUDNN
  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_compute_fwd(const ConstTensor &input, const ConstTensor &weight,
                                          const ConstTensor bias, Tensor &output, size_t batch_size,
                                          size_t input_h, size_t input_w, size_t output_h,
                                          size_t output_w, Tensor &cudnn_workspace,
                                          const std::string &flow_id);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_backward_data(const ConstTensor &gradient, const ConstTensor &weight,
                                            Tensor &input_grad, size_t batch_size, size_t input_h,
                                            size_t input_w, size_t output_h, size_t output_w,
                                            Tensor &cudnn_workspace, const std::string &flow_id);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_backward_filter(const ConstTensor &input, const ConstTensor &gradient,
                                              Tensor &weight_grad, size_t batch_size,
                                              size_t input_h, size_t input_w, size_t output_h,
                                              size_t output_w, Tensor &cudnn_workspace,
                                              const std::string &flow_id);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> cudnn_backward_bias(const ConstTensor &gradient, Tensor &bias_grad,
                                            size_t batch_size, size_t output_h, size_t output_w,
                                            size_t out_channels, const std::string &flow_id);
#endif

  void init_params() override;
  void forward_impl(const ConstTensor &input, Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &gradient, Tensor &grad_input, size_t mb_id = 0) override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;

public:
  LegacyConv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
                    size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                    bool use_bias = true, const std::string &name = "legacy_conv2d");

  ~LegacyConv2DLayer();

  static constexpr const char *TYPE_NAME = "legacy_conv2d";

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

  static std::unique_ptr<LegacyConv2DLayer> create_from_config(const LayerConfig &config);

  size_t cached_memory_bytes() const override;
};

}  // namespace tnn
