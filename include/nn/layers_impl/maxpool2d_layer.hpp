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

#include "device/task.hpp"
#include "nn/layers_impl/common/maxpool.hpp"
#include "stateless_layer.hpp"
#include "tensor/tensor.hpp"
#ifdef USE_DNNL
#include "nn/layers_impl/cpu/dnnl_maxpool_ops.hpp"
#endif

namespace tnn {

class MaxPool2DLayer : public StatelessLayer {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  std::unordered_map<size_t, Vec<size_t>> micro_batch_input_shapes_;

  std::unique_ptr<Task> forward_task_;
  std::unique_ptr<Task> backward_task_;

#ifdef USE_DNNL
  void build_dnnl_handle(const Vec<size_t> &input_shape) const;
  Tensor dnnl_forward(const ConstTensor &input, size_t mb_id);
  Tensor dnnl_backward(const ConstTensor &grad_output, size_t mb_id);

  mutable std::unordered_map<size_t, cpu::dnnl_maxpool::dnnlMaxPoolHandle_t *> dnnl_handle_cache;
  mutable std::unordered_map<size_t, MaxPoolStats> dnnl_stats_cache;
#endif

  template <typename IO_T>
  std::unique_ptr<Task> run_forward(const ConstTensor &input_data, const Tensor &output_data,
                                    size_t batch_size, size_t height, size_t width, size_t channels,
                                    size_t output_h, size_t output_w, const Tensor &mask_indices,
                                    flowHandle_t handle) const;

  std::unique_ptr<Task> run_forward(const ConstTensor &input_data, const Tensor &output_data,
                                    size_t batch_size, size_t height, size_t width, size_t channels,
                                    size_t output_h, size_t output_w, const Tensor &mask_indices,
                                    flowHandle_t handle) const;

  template <typename IO_T>
  std::unique_ptr<Task> run_backward(const ConstTensor &gradient_data,
                                     const Tensor &grad_input_data, size_t batch_size,
                                     size_t channels, size_t output_h, size_t output_w,
                                     const ConstTensor &mask_indices, flowHandle_t handle) const;

  std::unique_ptr<Task> run_backward(const ConstTensor &gradient_data,
                                     const Tensor &grad_input_data, size_t batch_size,
                                     size_t channels, size_t output_h, size_t output_w,
                                     const ConstTensor &mask_indices, flowHandle_t handle) const;

  Tensor forward_impl(const ConstTensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const ConstTensor &grad_output, size_t mb_id = 0) override;

public:
  MaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h = 1, size_t stride_w = 1,
                 size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "maxpool2d");
  ~MaxPool2DLayer();

  static constexpr const char *TYPE_NAME = "maxpool2d";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;

  static std::unique_ptr<MaxPool2DLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
