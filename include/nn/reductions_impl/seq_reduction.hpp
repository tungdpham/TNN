#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "nn/blocks_impl/sequential.hpp"
#include "nn/graph_builder.hpp"
#include "nn/reduction_strategy.hpp"

namespace tnn {

class SeqReduction : public ReductionStrategy {
public:
  void reduce(GraphBuilder &builder) override {
    auto &edges = builder.edges();
    if (edges.size() <= 1) return;

    // Build a mapping from IONode* to edge indices that produce it
    std::unordered_map<const IONode *, size_t> producer_map;
    for (size_t i = 0; i < edges.size(); ++i) {
      for (const IONode *consumer : edges[i].consumers()) {
        producer_map[consumer] = i;
      }
    }

    // Find chains of sequential operations
    std::unordered_set<size_t> processed;
    Vec<Vec<size_t>> chains;

    for (size_t i = 0; i < edges.size(); ++i) {
      if (processed.count(i)) continue;

      const auto &edge = edges[i];

      // Only consider edges with exactly 1 producer and 1 consumer for fusion
      if (edge.producers().size() != 1 || edge.consumers().size() != 1) {
        continue;
      }

      // Build chain starting from this edge
      Vec<size_t> chain;
      chain.push_back(i);
      processed.insert(i);

      size_t current_idx = i;

      // Follow the chain of single producer/consumer edges
      while (true) {
        const IONode *consumer = edges[current_idx].consumers()[0];
        auto it = producer_map.find(consumer);
        if (it == producer_map.end()) break;

        size_t next_idx = it->second;
        if (processed.count(next_idx)) break;

        const auto &next_edge = edges[next_idx];
        if (next_edge.producers().size() != 1 || next_edge.consumers().size() != 1) {
          break;
        }

        chain.push_back(next_idx);
        processed.insert(next_idx);
        current_idx = next_idx;
      }

      // Only record chains of length > 1 (worthwhile to fuse)
      if (chain.size() > 1) {
        chains.push_back(chain);
      }
    }

    if (chains.empty()) return;

    // Collect indices to skip
    std::unordered_set<size_t> skip_indices;
    for (const auto &chain : chains) {
      for (size_t idx : chain) {
        skip_indices.insert(idx);
      }
    }

    // Build new edges list
    Vec<Edge> new_edges;

    // Add non-fused edges
    for (size_t i = 0; i < edges.size(); ++i) {
      if (!skip_indices.count(i)) {
        new_edges.push_back(std::move(edges[i]));
      }
    }

    // Add fused edges for each chain
    for (const auto &chain : chains) {
      Vec<std::unique_ptr<Layer>> composite_layers;

      // Collect layers from the chain
      for (size_t idx : chain) {
        composite_layers.push_back(edges[idx].op_node().release_layer());
      }

      // Create the composite sequential layer
      auto composite = std::make_unique<Sequential>(std::move(composite_layers));

      // Get producers from first edge and consumers from last edge
      const auto &first_edge = edges[chain.front()];
      const auto &last_edge = edges[chain.back()];

      auto producers = first_edge.producers();
      auto consumers = last_edge.consumers();

      // Add composite layer and get reference to its OpNode
      OpNode &fused_node = builder.add_layer(std::move(composite));

      // Create fused edge
      new_edges.push_back(Edge(producers, consumers, fused_node));
    }

    edges = std::move(new_edges);
  }
};
}  // namespace tnn