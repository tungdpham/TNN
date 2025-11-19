/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device_ptr.hpp"
#include "stateless_layer.hpp"
#include "tensor/tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

template <typename T = float> class MaxPool2DLayer : public StatelessLayer<T> {
private:
  size_t pool_h_;
  size_t pool_w_;
  size_t stride_h_;
  size_t stride_w_;
  size_t pad_h_;
  size_t pad_w_;

  std::unordered_map<size_t, device_ptr<size_t[]>> micro_batch_mask_indices_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_inputs_;

  mutable size_t input_stride_n_, input_stride_c_, input_stride_h_, input_stride_w_;
  mutable size_t output_stride_n_, output_stride_c_, output_stride_h_, output_stride_w_;

  void compute_max_pool_forward(const device_ptr<T[]> &input_data, device_ptr<T[]> &output_data,
                                size_t batch_size, size_t channels, size_t input_h, size_t input_w,
                                size_t output_h, size_t output_w,
                                device_ptr<size_t[]> &mask_indices) const;

  void compute_max_pool_backward(const device_ptr<T[]> &gradient_data,
                                 device_ptr<T[]> &grad_input_data, size_t batch_size,
                                 size_t channels, size_t output_h, size_t output_w,
                                 const device_ptr<size_t[]> &mask_indices) const;

public:
  MaxPool2DLayer(size_t pool_h, size_t pool_w, size_t stride_h = 0, size_t stride_w = 0,
                 size_t pad_h = 0, size_t pad_w = 0, const std::string &name = "maxpool2d");

  Tensor<T> forward(const Tensor<T> &input, size_t micro_batch_id = 0) override;
  Tensor<T> backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) override;

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override;

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

  static std::unique_ptr<Layer<T>> create_from_config(const LayerConfig &config);
};

} // namespace tnn

#include "nn/layers_impl/maxpool2d_layer.tpp"