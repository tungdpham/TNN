/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device.hpp"
#include "device/device_ptr.hpp"
#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

template <typename T = float> class BatchNormLayer : public ParameterizedLayer<T> {
private:
  size_t num_features_;
  T epsilon_;
  T momentum_;
  bool affine_;

  Tensor<T> gamma_;
  Tensor<T> beta_;
  Tensor<T> gamma_gradients_;
  Tensor<T> beta_gradients_;

  Tensor<T> running_mean_;
  Tensor<T> running_var_;
  Tensor<T> running_mean_gradients_; // dummy gradient
  Tensor<T> running_var_gradients_;  // dummy gradient

  std::unordered_map<size_t, Tensor<T>> micro_batch_inputs_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_gradients_;
  std::unordered_map<size_t, device_ptr<T[]>> micro_batch_normalized_;
  std::unordered_map<size_t, device_ptr<T[]>> micro_batch_inv_std_;
  std::unordered_map<size_t, device_ptr<T[]>> batch_mean_fixed_;

  // Workspace buffers for backward pass (GPU optimization)
  std::unordered_map<size_t, device_ptr<T[]>> workspace_sum_grad_normalized_;
  std::unordered_map<size_t, device_ptr<T[]>> workspace_sum_grad_norm_times_norm_;

  void extract_tensor_dimensions(const Tensor<T> &input, size_t &batch_size, size_t &channels,
                                 size_t &height, size_t &width, size_t &spatial_size);

  std::unique_ptr<Task> compute_inference_output(const Tensor<T> &input, Tensor<T> &output,
                                                 size_t batch_size, size_t channels,
                                                 size_t spatial_size,
                                                 const std::string &flow_id = "batchnorm_infer");

public:
  explicit BatchNormLayer(size_t num_features, T epsilon = T(1e-5), T momentum = T(0.1),
                          bool affine = true, const std::string &name = "batchnorm");

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

#include "nn/layers_impl/batchnorm_layer.tpp"