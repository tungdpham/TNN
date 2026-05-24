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

#include "nn/block.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

class MSequential : public Block {
private:
  Vec<std::unique_ptr<Sequential>> sequences_;
  std::unique_ptr<Layer> join_layer_;

  // Cache for memory planning
  struct SequenceMemInfo {
    size_t cycling_cost;  // W_i: peak memory pressure during sequence execution (measured via hook)
    size_t output_size;  // b_i = O_i + R_i: retained memory after forward (output + residual cache)
    int priority;        // W_i - b_i: scheduling priority (descending = execute first)
    size_t index;        // original index in sequences_ vector
  };

  // Cached execution order (sorted by priority, descending)
  Vec<size_t> execution_order_;
  bool execution_order_cached_ = false;

  std::unordered_map<size_t, Vec<Vec<size_t>>> input_shapes_cache_;

  Vec<size_t> compute_execution_order(const Vec<ConstTensor> &inputs, size_t mb_id);

  SequenceMemInfo measure_sequence_memory(size_t seq_idx, ConstTensor input, size_t mb_id);

protected:
  Vec<Layer *> layers() override {
    Vec<Layer *> layers;
    for (auto &seq : sequences_) {
      layers.push_back(seq.get());
    }
    if (join_layer_) {
      layers.push_back(join_layer_.get());
    }
    return layers;
  }

  Vec<Tensor> forward_impl(const Vec<ConstTensor> &inputs, size_t mb_id) override;
  Vec<Tensor> backward_impl(const Vec<ConstTensor> &grad_outputs, size_t mb_id) override;

public:
  /**
   * Construct MSequential block
   *
   * @param sequences Vector of Sequential blocks (the parallel branches)
   * @param join_layer Layer that accepts multiple inputs and produces single output
   * @param name Block name
   */
  explicit MSequential(Vec<std::unique_ptr<Sequential>> sequences,
                       std::unique_ptr<Layer> join_layer, const std::string &name = "msequential");

  static constexpr const char *TYPE_NAME = "msequential";

  std::string type() const override { return TYPE_NAME; }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;

  void print_summary(const Vec<Vec<size_t>> &input_shapes) const;

  Vec<Sequential *> get_sequences();
  Layer *get_join_layer();

  LayerConfig get_config() const override;
  static std::unique_ptr<MSequential> create_from_config(const LayerConfig &config);
};

}  // namespace tnn
