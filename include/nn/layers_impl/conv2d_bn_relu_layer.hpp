/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#ifdef USE_CUDNN
#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

class Conv2DBNReLULayer : public ParameterizedLayer {
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
  Tensor gamma_;
  Tensor beta_;
  Tensor running_mean_;
  Tensor running_var_;

  Tensor weight_gradients_;
  Tensor bias_gradients_;
  Tensor grad_gamma_;
  Tensor grad_beta_;

  std::unordered_map<size_t, Tensor> micro_batch_inputs_cache_;

  void cudnn_forward(const Tensor &input, Tensor &output, size_t micro_batch_id);
  void cudnn_backward(const Tensor &current_gradient, Tensor &grad_input, size_t micro_batch_id);

  void init_params() override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;
  void forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input,
                     size_t micro_batch_id = 0) override;

public:
  Conv2DBNReLULayer(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w,
                    size_t stride_h = 1, size_t stride_w = 1, size_t pad_h = 0, size_t pad_w = 0,
                    bool use_bias = true, const std::string &name = "conv2d");

  ~Conv2DBNReLULayer();

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

  static std::unique_ptr<Layer> create_from_config(const LayerConfig &config);

  size_t cached_memory_bytes() const override;
};

} // namespace tnn

#endif