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

#ifdef USE_CUDNN
namespace tnn {
namespace cuda {
namespace cudnn_batchnorm {
struct BatchNormHandle;
}
} // namespace cuda
} // namespace tnn
#endif

namespace tnn {

template <typename T = float> class BatchNormLayer : public ParameterizedLayer<T> {
private:
  size_t num_features_;
  T epsilon_;
  T momentum_;
  bool affine_;

#ifdef USE_CUDNN
  cuda::cudnn_batchnorm::BatchNormHandle *cudnn_handle_ = nullptr;
  size_t cached_batch_size_ = 0;
  size_t cached_input_spatial_size_ = 0;
#endif

  Tensor<T> gamma_;
  Tensor<T> beta_;
  Tensor<T> gamma_gradients_;
  Tensor<T> beta_gradients_;

  Tensor<T> running_mean_;
  Tensor<T> running_var_;

  std::unordered_map<size_t, device_ptr<T[]>> micro_batch_normalized_;
  std::unordered_map<size_t, device_ptr<T[]>> micro_batch_inv_std_;
  std::unordered_map<size_t, device_ptr<T[]>> batch_mean_fixed_;
  std::unordered_map<size_t, Tensor<T>> micro_batch_inputs_cache_;

  std::unique_ptr<Task> forward_task_;
  std::unique_ptr<Task> backward_task_;

  void def_forward(const Tensor<T> *current, Tensor<T> &output, size_t micro_batch_id);
  void def_backward(const Tensor<T> *current_gradient, Tensor<T> &grad_input,
                    size_t micro_batch_id);

#ifdef USE_CUDNN
  void cudnn_forward(const Tensor<T> *current, Tensor<T> &output, size_t micro_batch_id);
  void cudnn_backward(const Tensor<T> *current_gradient, Tensor<T> &grad_input,
                      size_t micro_batch_id);
#endif

  std::unique_ptr<Task> compute_inference_output(const Tensor<T> &input, Tensor<T> &output,
                                                 size_t batch_size, size_t channels,
                                                 size_t spatial_size,
                                                 const std::string &flow_id = "default");

  std::unique_ptr<Task>
  run_forward_fused(const device_ptr<T[]> &input, device_ptr<T[]> &batch_mean_fixed,
                    device_ptr<T[]> &batch_inv_std, device_ptr<T[]> &running_mean,
                    device_ptr<T[]> &running_var, const device_ptr<T[]> &gamma,
                    const device_ptr<T[]> &beta, device_ptr<T[]> &output,
                    device_ptr<T[]> &norm_cache, size_t batch_size, size_t channels,
                    size_t spatial_size, const std::string &flow_id = "default");

  std::unique_ptr<Task> run_backward_fused(const device_ptr<T[]> &grad_output,
                                           const device_ptr<T[]> &norm_input,
                                           const device_ptr<T[]> &inv_std,
                                           const device_ptr<T[]> &gamma, device_ptr<T[]> &d_gamma,
                                           device_ptr<T[]> &d_beta, device_ptr<T[]> &grad_input,
                                           size_t batch_size, size_t channels, size_t spatial_size,
                                           const std::string &flow_id = "default");

public:
  explicit BatchNormLayer(size_t num_features, T epsilon = T(1e-5), T momentum = T(0.1),
                          bool affine = true, const std::string &name = "batchnorm");
  ~BatchNormLayer() override;

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

  const Tensor<T> &running_mean() const { return running_mean_; }
  const Tensor<T> &running_var() const { return running_var_; }

  size_t cached_memory_bytes() const override;

protected:
  void initialize_params() override;
  void collect_parameters(std::vector<Tensor<T> *> &params) override;
  void collect_gradients(std::vector<Tensor<T> *> &grads) override;
  void clear_gradients() override;
};

} // namespace tnn

#include "nn/layers_impl/batchnorm_layer.tpp"