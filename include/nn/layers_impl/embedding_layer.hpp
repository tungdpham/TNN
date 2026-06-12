/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>
#include <string>

#include "nn/siso_layer.hpp"
#include "tensor/tensor.hpp"

namespace synet {

class EmbeddingLayerImpl : public SISOLayerImpl {
private:
  size_t vocab_size_;
  size_t embed_dim_;
  size_t padding_idx_;
  Tensor weight_;
  Tensor weight_gradients_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_forward_impl(const Tensor &input, const Tensor &weight,
                                             Tensor &output, size_t num_indices, size_t vocab_size,
                                             size_t embed_dim, size_t padding_idx,
                                             flowHandle_t handle) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_backward_impl(const Tensor &input, const Tensor &grad_output,
                                              Tensor &grad_weight, size_t num_indices,
                                              size_t vocab_size, size_t embed_dim,
                                              size_t padding_idx, flowHandle_t handle) const;

  void init_impl() override;
  Tensor forward_impl(const Tensor &input, size_t mb_id = 0) override;
  Tensor backward_impl(const Tensor &grad_output, size_t mb_id = 0) override;

public:
  EmbeddingLayerImpl(size_t vocab_size, size_t embed_dim, const std::string &name = "embedding",
                     size_t padding_idx = static_cast<size_t>(-1));

  static constexpr const char *TYPE_NAME = "embedding";

  Vec<size_t> compute_output_shape(const Vec<size_t> &input_shape) const override;
  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;

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

  static std::shared_ptr<EmbeddingLayerImpl> create_from_config(const LayerConfig &config);
};

class EmbeddingLayer : public LayerRef<EmbeddingLayerImpl> {
public:
  EmbeddingLayer(size_t vocab_size, size_t embed_dim, const std::string &name = "embedding",
                 size_t padding_idx = static_cast<size_t>(-1))
      : LayerRef(std::make_shared<EmbeddingLayerImpl>(vocab_size, embed_dim, name, padding_idx)) {}

  using LayerRef<EmbeddingLayerImpl>::LayerRef;
};

}  // namespace synet
