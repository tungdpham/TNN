/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/common/batchnorm.hpp"
#include "parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDNN
#include "cuda/cudnn_batchnorm_ops.hpp"
#include <cudnn.h>
#endif

namespace tnn {

class BatchNormLayer : public ParameterizedLayer {
private:
  size_t num_features_;
  float epsilon_;
  float momentum_;
  bool affine_;
  bool use_relu_;

  Tensor gamma_;
  Tensor beta_;
  Tensor gamma_gradients_;
  Tensor beta_gradients_;

  Tensor running_mean_;
  Tensor running_var_;
  Tensor dummy_mean_gradients_;
  Tensor dummy_var_gradients_;

  std::unordered_map<size_t, BatchNormStats> stats_cache;
  size_t get_shape_hash(size_t n, size_t c, size_t h, size_t w) const;

#ifdef USE_CUDNN
  std::unordered_map<size_t, cuda::cudnn_batchnorm::feHandle_t *> fe_handle_cache;
  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task>
  forward_training_task(cuda::cudnn_batchnorm::feHandle_t *fe_handle, BatchNormStats &stats,
                        const Tensor &input, Tensor &output, const Tensor &gamma,
                        const Tensor &beta, Tensor &prev_running_mean, Tensor &prev_running_var,
                        Tensor &next_running_mean, Tensor &next_running_var, Tensor &batch_mean,
                        Tensor &batch_invar, Tensor &workspace, const std::string &flow_id) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task>
  forward_inference_task(cuda::cudnn_batchnorm::feHandle_t *fe_handle, BatchNormStats &stats,
                         const Tensor &input, Tensor &output, const Tensor &gamma,
                         const Tensor &beta, const Tensor &saved_mean, const Tensor &saved_invar,
                         Tensor &workspace, const std::string &flow_id) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> backward_task(cuda::cudnn_batchnorm::feHandle_t *fe_handle,
                                      BatchNormStats &stats, const Tensor &gradient,
                                      const Tensor &input, Tensor &grad_input, const Tensor &gamma,
                                      Tensor &gamma_gradients, Tensor &beta_gradients,
                                      const Tensor &batch_mean, const Tensor &batch_var,
                                      Tensor &workspace, const std::string &flow_id) const;

  void cudnn_forward(const Tensor &input, Tensor &output, size_t mb_id);
  void cudnn_backward(const Tensor &gradient, Tensor &grad_input, size_t mb_id);
#endif

  void init_params() override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;
  void forward_impl(const Tensor &input, Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input, size_t mb_id = 0) override;

public:
  explicit BatchNormLayer(size_t num_features, float epsilon = 1e-5f, float momentum = 0.1f,
                          bool affine = true, bool use_relu = false,
                          const std::string &name = "batchnorm");
  ~BatchNormLayer() override;

  static constexpr const char *TYPE_NAME = "batchnorm";

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  static std::unique_ptr<BatchNormLayer> create_from_config(const LayerConfig &config);
  std::unique_ptr<Layer> clone() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
};

} // namespace tnn
