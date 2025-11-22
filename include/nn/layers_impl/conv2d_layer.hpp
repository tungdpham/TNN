/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device_ptr.hpp"
#include "device/task.hpp"
#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

template <typename T = float> class Conv2DLayer : public ParameterizedLayer<T> {
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

  Tensor<T> weights_;
  Tensor<T> bias_;
  Tensor<T> weight_gradients_;
  Tensor<T> bias_gradients_;

  mutable std::unordered_map<size_t, std::vector<size_t>> micro_batch_input_shapes_;
  mutable std::unordered_map<size_t, Tensor<T>> micro_batch_pre_activations_;
  mutable std::unordered_map<size_t, device_ptr<T[]>> micro_batch_col_buffers_;

  std::unique_ptr<Task> im2col_task_;
  std::unique_ptr<Task> forward_task_;
  std::unique_ptr<Task> cnhw_to_nchw_task_;
  std::unique_ptr<Task> add_bias_task_;

  std::unique_ptr<Task> nchw_to_cnhw_task_;
  std::unique_ptr<Task> weight_grad_task_;
  std::unique_ptr<Task> input_grad_task_;
  std::unique_ptr<Task> col2im_task_;
  std::unique_ptr<Task> bias_grad_task_;

  // Reusable temporary buffers to avoid allocation overhead
  mutable device_ptr<T[]> temp_output_buffer_;
  mutable device_ptr<T[]> temp_gradient_buffer_;
  mutable device_ptr<T[]> temp_col_grad_matrix_buffer_;

  std::unique_ptr<Task> compute_conv_forward(const device_ptr<T[]> &col_data,
                                             const device_ptr<T[]> &weight_data,
                                             device_ptr<T[]> &output_data, const size_t output_size,
                                             const size_t kernel_size, const size_t out_channels,
                                             const std::string &flow_id) const;

  std::unique_ptr<Task> compute_weight_gradients(const device_ptr<T[]> &col_data,
                                                 const device_ptr<T[]> &gradient_data,
                                                 device_ptr<T[]> &weight_grad_data,
                                                 const size_t output_size, const size_t kernel_size,
                                                 const size_t out_channels,
                                                 const std::string &flow_id) const;

  std::unique_ptr<Task> compute_input_gradients(const device_ptr<T[]> &gradient_data,
                                                const device_ptr<T[]> &weight_data,
                                                device_ptr<T[]> &col_grad_data,
                                                const size_t output_size, const size_t kernel_size,
                                                const size_t out_channels,
                                                const std::string &flow_id) const;

  std::unique_ptr<Task> compute_bias_gradients(const device_ptr<T[]> &gradient_data,
                                               device_ptr<T[]> &bias_grad_data,
                                               const size_t batch_size, const size_t output_h,
                                               const size_t output_w, const size_t out_channels,
                                               const std::string &flow_id) const;

  std::unique_ptr<Task> add_bias_to_output(device_ptr<T[]> &output_data,
                                           const device_ptr<T[]> &bias_data,
                                           const size_t batch_size, const size_t output_h,
                                           const size_t output_w, const size_t out_channels,
                                           const std::string &flow_id) const;

public:
  Conv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
              size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
              bool use_bias = true, const std::string &name = "conv2d");

  const Tensor<T> &forward(const Tensor<T> &input, size_t micro_batch_id = 0) override;
  const Tensor<T> &backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) override;

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override;

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

  static std::unique_ptr<Layer<T>> create_from_config(const LayerConfig &config);

protected:
  void initialize_params() override;
  void collect_parameters(std::vector<Tensor<T> *> &params) override;
  void collect_gradients(std::vector<Tensor<T> *> &grads) override;
  void clear_gradients() override;
};

} // namespace tnn

#include "nn/layers_impl/conv2d_layer.tpp"