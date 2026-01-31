/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"

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
  std::unique_ptr<Task> layer_norm_forward(const Tensor &input, Tensor &output, const Tensor &gamma,
                                           const Tensor &beta, size_t batch_size, size_t channels,
                                           const std::string &flow_id = "default") const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> layer_norm_backward(const Tensor &gradient, const Tensor &input,
                                            const Tensor &gamma, Tensor &grad_input,
                                            Tensor &gamma_gradients, Tensor &beta_gradients,
                                            size_t batch_size, size_t channels,
                                            const std::string &flow_id = "default") const;

  void forward_impl(const Tensor &input, Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id = 0) override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;

public:
  explicit LayerNormLayer(size_t normalized_shape, float epsilon = 1e-5f, bool affine = true,
                          const std::string &name = "layer_norm");

  static constexpr const char *TYPE_NAME = "layer_norm";

  void init_params() override;
  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;
  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override {
    return input_shape;
  }

public:
  static std::unique_ptr<LayerNormLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
