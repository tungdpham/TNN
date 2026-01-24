/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

class LegacyBatchNormLayer : public ParameterizedLayer {
private:
  size_t num_features_;
  float epsilon_;
  float momentum_;
  bool affine_;

  Tensor gamma_;
  Tensor beta_;
  Tensor gamma_gradients_;
  Tensor beta_gradients_;

  Tensor running_mean_;
  Tensor running_var_;

  std::unordered_map<size_t, Tensor> micro_batch_normalized_;
  std::unordered_map<size_t, Tensor> micro_batch_inv_std_;
  std::unordered_map<size_t, Tensor> batch_mean_fixed_;
  std::unordered_map<size_t, Tensor> micro_batch_inputs_cache_;

  std::unique_ptr<Task> forward_task_;
  std::unique_ptr<Task> backward_task_;

  void def_forward(const Tensor &input, Tensor &output, size_t micro_batch_id);
  void def_backward(const Tensor &gradient, Tensor &grad_input, size_t micro_batch_id);

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_inference_output_impl(const Tensor &input, Tensor &output,
                                                      size_t batch_size, size_t channels,
                                                      size_t spatial_size,
                                                      const std::string &flow_id = "default");

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task>
  compute_inference_output(const Tensor &input, Tensor &output, size_t batch_size, size_t channels,
                           size_t spatial_size, const std::string &flow_id = "default");

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task>
  run_forward_fused(const Tensor &input, Tensor &batch_mean_fixed, Tensor &batch_inv_std,
                    Tensor &running_mean, Tensor &running_var, const Tensor &gamma,
                    const Tensor &beta, Tensor &output, Tensor &norm_cache, size_t batch_size,
                    size_t channels, size_t spatial_size, const std::string &flow_id = "default");

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> run_backward_fused(const Tensor &grad_output, const Tensor &norm_input,
                                           const Tensor &inv_std, const Tensor &gamma,
                                           Tensor &d_gamma, Tensor &d_beta, Tensor &grad_input,
                                           size_t batch_size, size_t channels, size_t spatial_size,
                                           const std::string &flow_id = "default");

  void init_params() override;
  void forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input,
                     size_t micro_batch_id = 0) override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;

public:
  static constexpr const char *TYPE_NAME = "legacy_batchnorm";

  explicit LegacyBatchNormLayer(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                                bool affine = true, const std::string &name = "batchnorm");

  void save_state(std::ofstream &file) override;
  void load_state(std::ifstream &file) override;

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  static std::unique_ptr<LegacyBatchNormLayer> create_from_config(const LayerConfig &config);

  size_t cached_memory_bytes() const override;
};

} // namespace tnn
