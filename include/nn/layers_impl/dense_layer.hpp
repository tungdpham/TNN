/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

class DenseLayer : public ParameterizedLayer {
private:
  size_t input_features_;
  size_t output_features_;
  bool use_bias_;
  Tensor weights_;
  Tensor bias_;
  Tensor weight_gradients_;
  Tensor bias_gradients_;

  std::unordered_map<size_t, Tensor> micro_batch_inputs_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_dense_forward(const Tensor &input, const Tensor &weights,
                                              Tensor &output, size_t batch_size,
                                              size_t input_features, size_t output_features,
                                              const std::string &flow_id) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_weight_gradients(const Tensor &input, const Tensor &gradient,
                                                 Tensor &weight_grad, size_t batch_size,
                                                 size_t input_features, size_t output_features,
                                                 const std::string &flow_id) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_input_gradients(const Tensor &gradient, const Tensor &weights,
                                                Tensor &grad_input, size_t batch_size,
                                                size_t input_features, size_t output_features,
                                                const std::string &flow_id) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_bias_gradients(const Tensor &gradient, Tensor &bias_gradient,
                                               size_t batch_size, size_t output_features,
                                               const std::string &flow_id) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> add_bias_vector(Tensor &output, const Tensor &bias, size_t batch_size,
                                        size_t output_features, const std::string &flow_id) const;

  void init_params() override;
  void forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input,
                     size_t micro_batch_id = 0) override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;

public:
  DenseLayer(size_t input_features, size_t output_features, bool use_bias = true,
             const std::string &name = "dense");

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
