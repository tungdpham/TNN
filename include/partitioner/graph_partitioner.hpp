#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "nn/graph_api.hpp"

namespace tnn {

struct GraphPartition {
  Graph graph;
  std::vector<std::string> input_uids;
  std::vector<std::string> output_uids;
  size_t start_layer = 0;
  size_t layer_count = 0;
};

class GraphPartitioner {
public:
  explicit GraphPartitioner(std::vector<size_t> layer_counts);

  std::vector<GraphPartition> partition(const Graph &graph) const;
  const std::vector<size_t> &layer_counts() const { return layer_counts_; }

private:
  void validate_partitioning(size_t total_layers) const;

  std::vector<size_t> layer_counts_;
};

}  // namespace tnn