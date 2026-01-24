/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/layers_impl/parameterized_layer.hpp"
#include "tensor/tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tnn {

class EmbeddingLayer : public ParameterizedLayer {
private:
  size_t vocab_size_;
  size_t embed_dim_;
  size_t padding_idx_;
  Tensor weight_;
  Tensor grad_weight_;
  std::unordered_map<size_t, Tensor> micro_batch_inputs_;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_forward_impl(const Tensor &input, const Tensor &weight,
                                             Tensor &output, size_t num_indices, size_t vocab_size,
                                             size_t embed_dim, size_t padding_idx,
                                             const std::string &flow_id) const;

  template <typename IO_T, typename Param_T, typename Compute_T>
  std::unique_ptr<Task> compute_backward_impl(const Tensor &input, const Tensor &gradient,
                                              Tensor &grad_weight, size_t num_indices,
                                              size_t vocab_size, size_t embed_dim,
                                              size_t padding_idx, const std::string &flow_id) const;

  void init_params() override;
  void forward_impl(const Tensor &input, Tensor &output, size_t micro_batch_id = 0) override;
  void backward_impl(const Tensor &gradient, Tensor &grad_input,
                     size_t micro_batch_id = 0) override;
  void collect_parameters(std::vector<Tensor> &params) override;
  void collect_gradients(std::vector<Tensor> &grads) override;

public:
  static constexpr const char *TYPE_NAME = "embedding";

  EmbeddingLayer(size_t vocab_size, size_t embed_dim, const std::string &name = "embedding",
                 size_t padding_idx = static_cast<size_t>(-1));

  std::vector<size_t> compute_output_shape(const std::vector<size_t> &input_shape) const override;
  uint64_t forward_flops(const std::vector<size_t> &input_shape) const override;
  uint64_t backward_flops(const std::vector<size_t> &input_shape) const override;

  std::string type() const override { return TYPE_NAME; }
  LayerConfig get_config() const override;
  std::unique_ptr<Layer> clone() const override;

  size_t cached_memory_bytes() const override;

public:
  static std::unique_ptr<EmbeddingLayer> create_from_config(const LayerConfig &config);
};

} // namespace tnn
