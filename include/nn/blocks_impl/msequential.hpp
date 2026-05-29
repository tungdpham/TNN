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

class MSequentialImpl : public Block {
private:
  Vec<Sequential> sequences_;
  Layer join_layer_;

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
  Vec<LayerImpl *> layers() override {
    Vec<LayerImpl *> layers;
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
   * @param join_layer LayerImpl that accepts multiple inputs and produces single output
   * @param name Block name
   */
  explicit MSequentialImpl(Vec<Sequential> sequences, Layer join_layer,
                           const std::string &name = "msequential");

  static constexpr const char *TYPE_NAME = "msequential";

  std::string type() const override { return TYPE_NAME; }

  Vec<Vec<size_t>> output_shapes(const Vec<Vec<size_t>> &input_shapes) const override;

  void print_summary(const Vec<Vec<size_t>> &input_shapes) const;

  Vec<SequentialImpl *> get_sequences();
  LayerImpl *get_join_layer();

  LayerConfig get_config() const override;
  static std::shared_ptr<MSequentialImpl> create_from_config(const LayerConfig &config);

  Node operator()(const Vec<Node> &inputs) {
    if (inputs.empty()) {
      throw std::runtime_error("Input nodes are empty");
    }
    Graph *graph = inputs[0]->graph();
    Node output = graph->make_node();

    std::shared_ptr<LayerImpl> self = shared_from_this();

    graph->add_edge(self, inputs, {output});
    return output;
  }
};

class MSequential : public LayerRef<MSequentialImpl> {
public:
  explicit MSequential(Vec<Sequential> sequences, Layer join_layer,
                       const std::string &name = "msequential")
      : LayerRef(
            std::make_shared<MSequentialImpl>(std::move(sequences), std::move(join_layer), name)) {}

  using LayerRef<MSequentialImpl>::LayerRef;
};

}  // namespace tnn
