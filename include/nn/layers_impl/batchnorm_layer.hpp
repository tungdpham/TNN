/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

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
  std::unordered_map<size_t, Tensor<T>> micro_batch_normalized_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_std_;

  void normalize_and_scale_optimized(const device_ptr<T[]> &input_data,
                                     const device_ptr<T[]> &mean_data,
                                     const device_ptr<T[]> &std_data,
                                     const device_ptr<T[]> &gamma_data,
                                     const device_ptr<T[]> &beta_data, device_ptr<T[]> &output_data,
                                     device_ptr<T[]> &normalized_data, size_t batch_size,
                                     size_t channels, size_t spatial_size, bool affine);

  void compute_batch_std(const Tensor<T> &batch_var, Tensor<T> &batch_std, size_t channels);
  void update_running_stats(const Tensor<T> &batch_mean, const Tensor<T> &batch_var,
                            size_t channels);
  void compute_inference_output(const Tensor<T> &input, Tensor<T> &output, size_t batch_size,
                                size_t channels, size_t spatial_size);
  void extract_tensor_dimensions(const Tensor<T> &input, size_t &batch_size, size_t &channels,
                                 size_t &height, size_t &width, size_t &spatial_size);

  // Device-specific helper functions
  void compute_mean_variance_fused(const Tensor<T> &input, Tensor<T> &batch_mean,
                                   Tensor<T> &batch_var, size_t batch_size, size_t channels,
                                   size_t spatial_size);
  void compute_batchnorm_backward_fused(const Tensor<T> &gradient, const Tensor<T> &normalized,
                                        const Tensor<T> &std_val, Tensor<T> &dummy_gamma,
                                        Tensor<T> &grad_input, size_t batch_size, size_t channels,
                                        size_t spatial_size);

  // Device-agnostic wrapper functions
  void compute_batch_std_wrapper(const device_ptr<T[]> &batch_var_data,
                                 device_ptr<T[]> &batch_std_data, size_t channels, T epsilon);
  void update_running_stats_wrapper(device_ptr<T[]> &running_mean_data,
                                    device_ptr<T[]> &running_var_data,
                                    const device_ptr<T[]> &batch_mean_data,
                                    const device_ptr<T[]> &batch_var_data, size_t channels,
                                    T momentum);
  void compute_inference_output_wrapper(
      const device_ptr<T[]> &input_data, const device_ptr<T[]> &running_mean_data,
      const device_ptr<T[]> &running_var_data, const device_ptr<T[]> &gamma_data,
      const device_ptr<T[]> &beta_data, device_ptr<T[]> &output_data, size_t batch_size,
      size_t channels, size_t spatial_size, T epsilon, bool affine);

public:
  explicit BatchNormLayer(size_t num_features, T epsilon = T(1e-5), T momentum = T(0.1),
                          bool affine = true, const std::string &name = "batchnorm");

  Tensor<T> forward(const Tensor<T> &input, size_t micro_batch_id = 0) override;
  Tensor<T> backward(const Tensor<T> &gradient, size_t micro_batch_id = 0) override;

  void forward_inplace(Tensor<T> &input, size_t micro_batch_id = 0) override;

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