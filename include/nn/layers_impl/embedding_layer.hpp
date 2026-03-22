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

class EmbeddingLayer : public ParameterizedLayer {
private:
  size_t vocab_size_;
  size_t embed_dim_;
  size_t padding_idx_;
  Tensor weight_;
  Tensor weight_gradients_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_forward_impl(const ConstTensor &input, const ConstTensor &weight,
                                             const Tensor &output, size_t num_indices,
                                             size_t vocab_size, size_t embed_dim,
                                             size_t padding_idx, flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_backward_impl(const ConstTensor &input,
                                              const ConstTensor &grad_output,
                                              const Tensor &grad_weight, size_t num_indices,
                                              size_t vocab_size, size_t embed_dim,
                                              size_t padding_idx, flowHandle_t handle) const;

  Vec<ParamDescriptor> param_descriptors() override {
    Vec<ParamDescriptor> descriptors;
    auto weight_desc = ParamDescriptor{
        param_dtype_,
        {vocab_size_, embed_dim_},
        &weight_,
        &weight_gradients_,
    };
    descriptors.push_back(weight_desc);
    return descriptors;
  }

  void init_impl() override;
  void forward_impl(const ConstTensor &input, const Tensor &output, size_t mb_id = 0) override;
  void backward_impl(const ConstTensor &grad_output, const Tensor &grad_input,
                     size_t mb_id = 0) override;

public:
  EmbeddingLayer(size_t vocab_size, size_t embed_dim, const std::string &name = "embedding",
                 size_t padding_idx = static_cast<size_t>(-1));

  static constexpr const char *TYPE_NAME = "embedding";

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  size_t fwd_cache_bytes(const Vec<Vec<size_t>> &input_shapes) const override { return 0; }
  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override {
    auto output_shapes = this->output_shapes(input_shapes);
    return get_shapes_bytes(output_shapes, io_dtype_);
  }
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override {
    auto output_shapes = this->output_shapes(input_shapes);
    return get_shapes_bytes(output_shapes, io_dtype_);
  }
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override {
    return get_shapes_bytes(input_shapes, io_dtype_);
  }

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

  static std::unique_ptr<EmbeddingLayer> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
