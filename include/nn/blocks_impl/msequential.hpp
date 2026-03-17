/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fmt/core.h>

#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "nn/block.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

/**
 * MSequential: Multi-Input Single-Output (MISO) Block
 *
 * Implements the joining architecture described in Section 3.1.2 of the paper.
 * Multiple independent Sequential branches converge into a single join layer.
 *
 * Memory scheduling follows Algorithm 2 (Minimum Space for m joining SISO sequences)
 * to optimize buffer reuse by executing branches in order of decreasing (M_b,i - O_i).
 */
class MSequential : public Block {
private:
  std::vector<std::unique_ptr<Sequential>> sequences_;
  std::unique_ptr<Layer> join_layer_;

  // Cache for memory planning
  struct SequenceMemInfo {
    size_t cycling_cost;  // M_b,i: peak buffer requirement for sequence i
    size_t output_size;   // O_i: terminal output size of sequence i
    int priority;         // M_b,i - O_i: used for scheduling order
    size_t index;         // original index in sequences_ vector
  };

  // Cached execution order (sorted by priority, descending)
  mutable std::vector<size_t> execution_order_;
  mutable bool execution_order_cached_ = false;

  // Cache input shapes for backward pass
  std::unordered_map<size_t, Vec<Vec<size_t>>> input_shapes_cache_;

  // Compute optimal execution order using Algorithm 2
  std::vector<size_t> compute_execution_order(const Vec<Vec<size_t>> &input_shapes) const;

  // Helper to compute individual sequence memory requirements
  SequenceMemInfo compute_sequence_memory(size_t seq_idx, const Vec<size_t> &input_shape) const;

protected:
  std::vector<Layer *> layers() override {
    std::vector<Layer *> layers;
    for (auto &seq : sequences_) {
      auto seq_layers = seq->get_layers();
      layers.insert(layers.end(), seq_layers.begin(), seq_layers.end());
    }
    if (join_layer_) {
      layers.push_back(join_layer_.get());
    }
    return layers;
  }

  void forward_impl(const Vec<ConstTensor> &inputs, const Vec<Tensor> &outputs,
                    size_t mb_id) override;
  void backward_impl(const Vec<ConstTensor> &grad_outputs, const Vec<Tensor> &grad_inputs,
                     size_t mb_id) override;

public:
  /**
   * Construct MSequential block
   *
   * @param sequences Vector of Sequential blocks (the parallel branches)
   * @param join_layer Layer that accepts multiple inputs and produces single output
   * @param name Block name
   */
  explicit MSequential(std::vector<std::unique_ptr<Sequential>> sequences,
                       std::unique_ptr<Layer> join_layer, const std::string &name = "msequential");

  static constexpr const char *TYPE_NAME = "msequential";

  std::string type() const override { return TYPE_NAME; }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;

  size_t fwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t inf_workspace(const Vec<Vec<size_t>> &input_shapes) const override;
  size_t bwd_workspace(const Vec<Vec<size_t>> &input_shapes) const override;

  void print_summary(const Vec<Vec<size_t>> &input_shapes) const;

  std::vector<Sequential *> get_sequences();
  Layer *get_join_layer();

  LayerConfig get_config() const override;
  static std::unique_ptr<MSequential> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
