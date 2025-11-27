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

template <typename T = float> class DenseLayer : public ParameterizedLayer<T> {
private:
  size_t input_features_;
  size_t output_features_;
  bool use_bias_;
  Tensor<T> weights_;
  Tensor<T> bias_;
  Tensor<T> weight_gradients_;
  Tensor<T> bias_gradients_;

  std::unordered_map<size_t, Tensor<T>> micro_batch_inputs_;

  std::unique_ptr<Task> forward_task_;
  std::unique_ptr<Task> add_bias_task_;
  std::unique_ptr<Task> activation_task_;

  std::unique_ptr<Task> weight_grad_task_;
  std::unique_ptr<Task> input_grad_task_;
  std::unique_ptr<Task> bias_grad_task_;

  std::unique_ptr<Task> compute_dense_forward(const device_ptr<T[]> &input_data,
                                              const device_ptr<T[]> &weight_data,
                                              device_ptr<T[]> &output_data, const size_t batch_size,
                                              const size_t input_features,
                                              const size_t output_features,
                                              const std::string &flow_id) const;

  std::unique_ptr<Task>
  compute_weight_gradients(const device_ptr<T[]> &input_data, const device_ptr<T[]> &gradient_data,
                           device_ptr<T[]> &weight_grad_data, const size_t batch_size,
                           const size_t input_features, const size_t output_features,
                           const std::string &flow_id) const;

  std::unique_ptr<Task>
  compute_input_gradients(const device_ptr<T[]> &gradient_data, const device_ptr<T[]> &weight_data,
                          device_ptr<T[]> &grad_input_data, const size_t batch_size,
                          const size_t input_features, const size_t output_features,
                          const std::string &flow_id) const;

  std::unique_ptr<Task> compute_bias_gradients(const device_ptr<T[]> &current_grad_data,
                                               device_ptr<T[]> &bias_gradient_data,
                                               const size_t batch_size,
                                               const size_t output_features,
                                               const std::string &flow_id) const;

  std::unique_ptr<Task> add_bias_vector(device_ptr<T[]> &output_data,
                                        const device_ptr<T[]> &bias_data, const size_t batch_size,
                                        const size_t output_features,
                                        const std::string &flow_id) const;

public:
  DenseLayer(size_t input_features, size_t output_features, bool use_bias = true,
             const std::string &name = "dense");

  void forward(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override;
  void backward(const Tensor<T> &gradient, Tensor<T> &grad_input,
                size_t micro_batch_id = 0) override;

  uint64_t forward_complexity(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_complexity(const std::vector<size_t> &input_shape) const override;

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

  static std::unique_ptr<Layer<T>> create_from_config(const LayerConfig &config);

  size_t cached_memory_bytes() const override;

protected:
  void initialize_params() override;
  void collect_parameters(std::vector<Tensor<T> *> &params) override;
  void collect_gradients(std::vector<Tensor<T> *> &grads) override;
  void clear_gradients() override;
};

} // namespace tnn

#include "nn/layers_impl/dense_layer.tpp"