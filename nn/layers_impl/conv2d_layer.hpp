#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../matrix/matrix.hpp"
#include "../../tensor/tensor.hpp"
#include "../activations.hpp"
#include "../optimizers.hpp"
#include "parameterized_layer.hpp"

namespace tnn {

// 2D Convolutional Layer using im2col + GEMM
template <typename T = float> class Conv2DLayer : public ParameterizedLayer<T> {
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
  std::unique_ptr<ActivationFunction<T>> activation_;

  Tensor<T> weights_; // [out_channels, in_channels, kernel_h, kernel_w]
  Tensor<T> bias_;    // [out_channels, 1, 1, 1]
  Tensor<T> weight_gradients_;
  Tensor<T> bias_gradients_;

  // Per-micro-batch state
  std::unordered_map<int, Tensor<T>> micro_batch_inputs_;
  std::unordered_map<int, Tensor<T>> micro_batch_pre_activations_;
  mutable std::unordered_map<int, Matrix<T>> micro_batch_im2col_matrices_;

  // Helper functions for convolution
  void conv_gemm_forward(const T *col_data, const T *weight_data,
                         T *output_data, size_t output_size, size_t kernel_size,
                         size_t out_channels) const;

  void conv_gemm_weight_gradients(const T *col_data, const T *grad_output_data,
                                  T *weight_grad_data, size_t output_size,
                                  size_t kernel_size,
                                  size_t out_channels) const;

  void conv_gemm_input_gradients(const T *grad_output_data,
                                 const T *weight_data, T *col_grad_data,
                                 size_t output_size, size_t kernel_size,
                                 size_t out_channels) const;

  void conv_gemm_forward_impl(const T *col_data, const T *weight_data,
                              T *output_data, const size_t output_size,
                              const size_t kernel_size,
                              const size_t out_channels) const;

  void conv_gemm_weight_gradients_impl(const T *col_data,
                                       const T *grad_output_data,
                                       T *weight_grad_data,
                                       const size_t output_size,
                                       const size_t kernel_size,
                                       const size_t out_channels) const;

  void conv_gemm_input_gradients_impl(const T *grad_output_data,
                                      const T *weight_data, T *col_grad_data,
                                      const size_t output_size,
                                      const size_t kernel_size,
                                      const size_t out_channels) const;

public:
  Conv2DLayer(size_t in_channels, size_t out_channels, size_t kernel_h,
              size_t kernel_w, size_t stride_h = 1, size_t stride_w = 1,
              size_t pad_h = 0, size_t pad_w = 0, bool use_bias = true,
              std::unique_ptr<ActivationFunction<T>> activation = nullptr,
              const std::string &name = "conv2d");

  Tensor<T> forward(const Tensor<T> &input, int micro_batch_id = 0) override;
  Tensor<T> backward(const Tensor<T> &grad_output,
                     int micro_batch_id = 0) override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;

  std::vector<size_t>
  compute_output_shape(const std::vector<size_t> &input_shape) const override;

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override;
  void collect_gradients(std::vector<Tensor<T> *> &grads) override;
  void update_parameters_impl(Optimizer<T> &optimizer) override;
};

} // namespace tnn

#include "conv2d_layer.tpp"