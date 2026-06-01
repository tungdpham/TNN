#include "partitioner/graph_partitioner.hpp"

#include <cmath>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace tnn {
namespace {

using WeakNode = std::weak_ptr<NodeImpl>;
using WeakNodeSet = std::set<WeakNode, std::owner_less<WeakNode>>;
using EdgeMap = std::map<WeakNode, Vec<Edge>, std::owner_less<WeakNode>>;

Vec<Edge> topologically_sorted_edges(const Graph &graph) {
  const Vec<Edge> original_edges = graph.edges();

  WeakNodeSet produced_nodes;
  for (const auto &edge : original_edges) {
    for (const auto &consumer : edge->consumers()) {
      produced_nodes.insert(consumer);
    }
  }

  EdgeMap node_to_dependent_edges;
  for (const auto &edge : original_edges) {
    for (const auto &producer : edge->producers()) {
      node_to_dependent_edges[producer].push_back(edge);
    }
  }

  std::map<Edge, size_t> pending_dependencies;
  std::queue<Edge> ready_edges;

  for (const auto &edge : original_edges) {
    size_t dependency_count = 0;
    for (const auto &producer : edge->producers()) {
      if (produced_nodes.count(producer) > 0) {
        ++dependency_count;
      }
    }
    pending_dependencies[edge] = dependency_count;
    if (dependency_count == 0) {
      ready_edges.push(edge);
    }
  }

  Vec<Edge> sorted_edges;
  sorted_edges.reserve(original_edges.size());

  while (!ready_edges.empty()) {
    Edge edge = ready_edges.front();
    ready_edges.pop();
    sorted_edges.push_back(edge);

    for (const auto &consumer : edge->consumers()) {
      const auto it = node_to_dependent_edges.find(consumer);
      if (it == node_to_dependent_edges.end()) {
        continue;
      }

      for (const auto &dependent_edge : it->second) {
        size_t &remaining = pending_dependencies[dependent_edge];
        if (remaining == 0) {
          continue;
        }
        --remaining;
        if (remaining == 0) {
          ready_edges.push(dependent_edge);
        }
      }
    }
  }

  if (sorted_edges.size() != original_edges.size()) {
    throw std::runtime_error("Graph contains a cycle; partitioning requires a DAG");
  }

  return sorted_edges;
}

Node ensure_partition_node(Graph &graph, const std::string &uid,
                           std::unordered_map<std::string, Node> &uid_to_node,
                           std::vector<std::string> &node_order) {
  const auto it = uid_to_node.find(uid);
  if (it != uid_to_node.end()) {
    return it->second;
  }

  Node node = graph.make_node(uid);
  uid_to_node.emplace(uid, node);
  node_order.push_back(uid);
  return node;
}

GraphPartition build_partition(const Vec<Edge> &edges, size_t start_layer) {
  GraphPartition partition;
  partition.start_layer = start_layer;
  partition.layer_count = edges.size();

  std::unordered_map<std::string, Node> uid_to_node;
  std::unordered_map<std::string, size_t> in_degree;
  std::unordered_map<std::string, size_t> out_degree;
  std::vector<std::string> node_order;

  for (const auto &edge : edges) {
    Vec<Node> producers;
    producers.reserve(edge->producers().size());
    for (const auto &producer : edge->producers()) {
      const std::string &uid = producer->uid();
      in_degree.try_emplace(uid, 0);
      out_degree.try_emplace(uid, 0);
      producers.push_back(ensure_partition_node(partition.graph, uid, uid_to_node, node_order));
      ++out_degree[uid];
    }

    Vec<Node> consumers;
    consumers.reserve(edge->consumers().size());
    for (const auto &consumer : edge->consumers()) {
      const std::string &uid = consumer->uid();
      in_degree.try_emplace(uid, 0);
      out_degree.try_emplace(uid, 0);
      consumers.push_back(ensure_partition_node(partition.graph, uid, uid_to_node, node_order));
      ++in_degree[uid];
    }

    partition.graph.add_edge(edge->layer(), producers, consumers);
  }

  for (const auto &uid : node_order) {
    if (in_degree[uid] == 0) {
      partition.input_uids.push_back(uid);
    }
    if (out_degree[uid] == 0) {
      partition.output_uids.push_back(uid);
    }
  }

  return partition;
}

}  // namespace

GraphPartitioner::GraphPartitioner(std::vector<double> layer_ratios)
    : layer_ratios_(std::move(layer_ratios)) {}

GraphPartitioner::GraphPartitioner(std::vector<size_t> layer_ratios) {
  layer_ratios_.reserve(layer_ratios.size());
  for (const size_t layer_ratio : layer_ratios) {
    layer_ratios_.push_back(static_cast<double>(layer_ratio));
  }
}

std::vector<GraphPartition> GraphPartitioner::partition(const Graph &graph) const {
  const Vec<Edge> sorted_edges = topologically_sorted_edges(graph);
  const std::vector<size_t> layer_counts = resolve_layer_counts(sorted_edges.size());

  std::vector<GraphPartition> partitions;
  partitions.reserve(layer_counts.size());

  size_t offset = 0;
  for (const size_t layer_count : layer_counts) {
    Vec<Edge> partition_edges;
    partition_edges.reserve(layer_count);
    for (size_t i = 0; i < layer_count; ++i) {
      partition_edges.push_back(sorted_edges[offset + i]);
    }

    partitions.push_back(build_partition(partition_edges, offset));
    offset += layer_count;
  }

  return partitions;
}

std::vector<size_t> GraphPartitioner::resolve_layer_counts(size_t total_layers) const {
  if (layer_ratios_.empty()) {
    if (total_layers == 0) {
      return {};
    }
    throw std::runtime_error("GraphPartitioner requires at least one partition ratio");
  }

  if (total_layers == 0) {
    throw std::runtime_error("GraphPartitioner cannot apply partition ratios to an empty graph");
  }

  double ratio_sum = 0.0;
  std::vector<size_t> layer_counts(layer_ratios_.size(), 0);
  struct FractionalCount {
    size_t index;
    double remainder;
  };
  std::vector<FractionalCount> fractional_counts;
  fractional_counts.reserve(layer_ratios_.size());

  for (const double layer_ratio : layer_ratios_) {
    if (!std::isfinite(layer_ratio) || layer_ratio <= 0.0) {
      throw std::runtime_error(
          "GraphPartitioner partition ratios must be finite and greater than zero");
    }
    ratio_sum += layer_ratio;
  }

  for (size_t i = 0; i < layer_ratios_.size(); ++i) {
    const double exact_layer_count =
        (layer_ratios_[i] / ratio_sum) * static_cast<double>(total_layers);
    const size_t resolved_layer_count = static_cast<size_t>(std::floor(exact_layer_count));
    layer_counts[i] = resolved_layer_count;
    fractional_counts.push_back({i, exact_layer_count - static_cast<double>(resolved_layer_count)});
  }

  size_t assigned_layers = std::accumulate(layer_counts.begin(), layer_counts.end(), size_t{0});
  std::sort(fractional_counts.begin(), fractional_counts.end(),
            [](const FractionalCount &lhs, const FractionalCount &rhs) {
              if (lhs.remainder == rhs.remainder) {
                return lhs.index < rhs.index;
              }
              return lhs.remainder > rhs.remainder;
            });

  for (size_t i = 0; i < total_layers - assigned_layers; ++i) {
    ++layer_counts[fractional_counts[i].index];
  }

  for (const size_t layer_count : layer_counts) {
    if (layer_count == 0) {
      throw std::runtime_error(
          "GraphPartitioner partition ratios resolve to an empty partition for this graph");
    }
  }

  return layer_counts;
}

}  // namespace tnn