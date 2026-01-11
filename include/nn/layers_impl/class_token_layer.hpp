/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/task.hpp"
#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"
#include <memory>
#include <string>
#include <vector>

namespace tnn {

template <typename T = float> class ClassTokenLayer : public ParameterizedLayer<T> {
private:
  size_t embed_dim_;
  Tensor<T> class_token_;
  Tensor<T> class_token_gradients_;

  void forward_impl(const Tensor<T> &input, Tensor<T> &output, size_t micro_batch_id = 0) override;
  void backward_impl(const Tensor<T> &gradient, Tensor<T> &grad_input,
                     size_t micro_batch_id = 0) override;

public:
  ClassTokenLayer(size_t embed_dim, const std::string &name = "class_token");

  void init_params() override;

  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override;
  LayerConfig get_config() const override;
  std::unique_ptr<Layer<T>> clone() const override;
  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

protected:
  void collect_parameters(std::vector<Tensor<T> *> &params) override;
  void collect_gradients(std::vector<Tensor<T> *> &grads) override;
};

} // namespace tnn

#include "nn/layers_impl/class_token_layer.tpp"
