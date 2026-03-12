/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class ClassTokenLayer : public ParameterizedLayer {
private:
  size_t embed_dim_;
  Tensor class_token_;
  Tensor class_token_gradients_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> forward_task(const ConstTensor &input, const Tensor &output,
                                     const ConstTensor &class_token, size_t batch_size,
                                     size_t seq_len, size_t embed_dim, flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> backward_task(const ConstTensor &grad_output, const Tensor &grad_input,
                                      const Tensor &class_token_gradients,
                                      const ConstTensor &class_token, size_t batch_size,
                                      size_t seq_len, size_t embed_dim, flowHandle_t handle) const;

  std::vector<ParamDescriptor> param_descriptors() override {
    std::vector<ParamDescriptor> descriptors;
    auto token_desc = ParamDescriptor{
        param_dtype_,
        {1, embed_dim_},
        &class_token_,
        &class_token_gradients_,
    };
    descriptors.push_back(token_desc);
    return descriptors;
  }

  void init_impl() override;
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  explicit ClassTokenLayer(size_t embed_dim, const std::string &name = "class_token");

  static constexpr const char *TYPE_NAME = "class_token";

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;

public:
  static std::unique_ptr<ClassTokenLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
