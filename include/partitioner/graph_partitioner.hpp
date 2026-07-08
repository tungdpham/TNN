#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "nn/graph.hpp"

namespace tunx {

struct GraphPartition {
  Graph graph;
  std::vector<std::string> input_uids;
  std::vector<std::string> output_uids;
  size_t start_layer = 0;
  size_t layer_count = 0;
};

class GraphPartitioner {
public:
  explicit GraphPartitioner(std::vector<double> layer_ratios);
  explicit GraphPartitioner(std::initializer_list<double> layer_ratios)
      : GraphPartitioner(std::vector<double>(layer_ratios)) {}
  explicit GraphPartitioner(std::vector<size_t> layer_ratios);

  std::vector<GraphPartition> partition(const Graph &graph) const;
  const std::vector<double> &layer_ratios() const { return layer_ratios_; }

private:
  std::vector<size_t> resolve_layer_counts(size_t total_layers) const;

  std::vector<double> layer_ratios_;
};

}  // namespace tunx