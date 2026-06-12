/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <array>
#include <cstdint>
#include <istream>
#include <ostream>
#include <unordered_map>

#include "nn/graph.hpp"
#include "nn/layers.hpp"
#include "tensor/tensor_factory.hpp"

namespace synet {
namespace {

constexpr std::array<char, 4> kGraphStateMagic{'T', 'N', 'N', 'G'};
constexpr std::uint32_t kGraphStateVersion = 2;

template <typename T>
void write_binary(std::ostream &stream, const T &value) {
  stream.write(reinterpret_cast<const char *>(&value), sizeof(T));
  if (!stream) {
    throw std::runtime_error("Failed to write graph state");
  }
}

template <typename T>
T read_binary(std::istream &stream) {
  T value{};
  stream.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!stream) {
    throw std::runtime_error("Failed to read graph state");
  }
  return value;
}

void write_string(std::ostream &stream, const std::string &value) {
  const size_t size = value.size();
  write_binary(stream, size);
  stream.write(value.data(), static_cast<std::streamsize>(size));
  if (!stream) {
    throw std::runtime_error("Failed to write graph state string");
  }
}

std::string read_string(std::istream &stream) {
  const size_t size = read_binary<size_t>(stream);
  std::string value(size, '\0');
  stream.read(value.data(), static_cast<std::streamsize>(size));
  if (!stream) {
    throw std::runtime_error("Failed to read graph state string");
  }
  return value;
}

Layer load_layer_config(std::istream &stream) {
  nlohmann::json config_json = nlohmann::json::parse(read_string(stream));
  LayerConfig config = LayerConfig::from_json(config_json);
  LayerFactory::register_defaults();
  return LayerFactory::create(config);
}

void save_layer_config(std::ostream &stream, const std::shared_ptr<LayerImpl> &layer) {
  nlohmann::json config_json = layer->get_config().to_json();
  write_string(stream, config_json.dump());
}

void save_node_index_set(std::ostream &stream, const std::set<Node> &nodes,
                         const std::unordered_map<NodeImpl *, size_t> &node_indices,
                         const char *context) {
  write_binary(stream, nodes.size());
  for (const auto &node : nodes) {
    const auto it = node_indices.find(node.get());
    if (it == node_indices.end()) {
      throw std::runtime_error(std::string("Internal error while saving graph ") + context +
                               " node state");
    }
    write_binary(stream, it->second);
  }
}

void load_node_index_set(std::istream &stream, Graph &graph, const Vec<Node> &nodes,
                         const char *context, const std::function<void(const Node &)> &mark_node) {
  const size_t count = read_binary<size_t>(stream);
  for (size_t i = 0; i < count; ++i) {
    const size_t node_index = read_binary<size_t>(stream);
    if (node_index >= nodes.size()) {
      throw std::runtime_error(std::string("Graph state references an invalid ") + context +
                               " node index");
    }
    mark_node(nodes[node_index]);
  }
}

}  // namespace

void Graph::save_state(std::ostream &stream) const {
  if (!stream) {
    throw std::runtime_error("Stream is not ready for writing");
  }

  std::unordered_map<NodeImpl *, size_t> node_indices;
  node_indices.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i) {
    node_indices.emplace(nodes_[i].get(), i);
  }

  std::unordered_map<LayerImpl *, size_t> layer_indices;
  layer_indices.reserve(edges_.size());
  Vec<std::shared_ptr<LayerImpl>> unique_layers;
  unique_layers.reserve(edges_.size());
  for (const auto &edge : edges_) {
    LayerImpl *layer_ptr = edge->layer().get();
    if (layer_indices.count(layer_ptr) == 0) {
      layer_indices.emplace(layer_ptr, unique_layers.size());
      unique_layers.push_back(edge->layer());
    }
  }

  stream.write(kGraphStateMagic.data(), static_cast<std::streamsize>(kGraphStateMagic.size()));
  if (!stream) {
    throw std::runtime_error("Failed to write graph state header");
  }
  write_binary(stream, kGraphStateVersion);

  write_binary(stream, nodes_.size());
  for (const auto &node : nodes_) {
    write_string(stream, node->uid());
  }

  save_node_index_set(stream, input_nodes_, node_indices, "input");
  save_node_index_set(stream, output_nodes_, node_indices, "output");

  write_binary(stream, unique_layers.size());
  for (const auto &layer : unique_layers) {
    save_layer_config(stream, layer);
  }

  write_binary(stream, edges_.size());
  for (const auto &edge : edges_) {
    auto layer_it = layer_indices.find(edge->layer().get());
    if (layer_it == layer_indices.end()) {
      throw std::runtime_error("Internal error while saving graph state");
    }
    write_binary(stream, layer_it->second);

    write_binary(stream, edge->producers().size());
    for (const auto &producer : edge->producers()) {
      auto node_it = node_indices.find(producer.get());
      if (node_it == node_indices.end()) {
        throw std::runtime_error("Graph edge producer is not registered as a node");
      }
      write_binary(stream, node_it->second);
    }

    write_binary(stream, edge->consumers().size());
    for (const auto &consumer : edge->consumers()) {
      auto node_it = node_indices.find(consumer.get());
      if (node_it == node_indices.end()) {
        throw std::runtime_error("Graph edge consumer is not registered as a node");
      }
      write_binary(stream, node_it->second);
    }
  }

  write_binary(stream, unique_layers.size());
  for (const auto &layer : unique_layers) {
    Vec<Tensor> params = layer->parameters();
    write_binary(stream, params.size());
    for (const auto &param : params) {
      if (!param) {
        throw std::runtime_error("Cannot save uninitialized layer parameter");
      }
      param->save(stream);
    }
  }
}

Graph Graph::load_state(std::istream &stream, IAllocator &allocator) {
  if (!stream) {
    throw std::runtime_error("Stream is not ready for reading");
  }

  std::array<char, 4> magic{};
  stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!stream) {
    throw std::runtime_error("Failed to read graph state header");
  }
  if (magic != kGraphStateMagic) {
    throw std::runtime_error("Invalid graph state file header");
  }

  const std::uint32_t version = read_binary<std::uint32_t>(stream);
  if (version != 1 && version != kGraphStateVersion) {
    throw std::runtime_error("Unsupported graph state version: " + std::to_string(version));
  }

  Graph graph;

  const size_t node_count = read_binary<size_t>(stream);
  Vec<Node> nodes;
  nodes.reserve(node_count);
  for (size_t i = 0; i < node_count; ++i) {
    nodes.push_back(graph.make_node(read_string(stream)));
  }

  if (version >= 2) {
    load_node_index_set(stream, graph, nodes, "input",
                        [&graph](const Node &node) { graph.set_input(node); });
    load_node_index_set(stream, graph, nodes, "output",
                        [&graph](const Node &node) { graph.set_output(node); });
  }

  const size_t layer_count = read_binary<size_t>(stream);
  Vec<std::shared_ptr<LayerImpl>> layers;
  layers.reserve(layer_count);
  for (size_t i = 0; i < layer_count; ++i) {
    Layer layer = load_layer_config(stream);
    layers.push_back(static_cast<std::shared_ptr<LayerImpl>>(layer));
  }

  const size_t edge_count = read_binary<size_t>(stream);
  for (size_t i = 0; i < edge_count; ++i) {
    const size_t layer_index = read_binary<size_t>(stream);
    if (layer_index >= layers.size()) {
      throw std::runtime_error("Graph state references an invalid layer index");
    }

    const size_t producer_count = read_binary<size_t>(stream);
    Vec<Node> producers;
    producers.reserve(producer_count);
    for (size_t j = 0; j < producer_count; ++j) {
      const size_t node_index = read_binary<size_t>(stream);
      if (node_index >= nodes.size()) {
        throw std::runtime_error("Graph state references an invalid producer node index");
      }
      producers.push_back(nodes[node_index]);
    }

    const size_t consumer_count = read_binary<size_t>(stream);
    Vec<Node> consumers;
    consumers.reserve(consumer_count);
    for (size_t j = 0; j < consumer_count; ++j) {
      const size_t node_index = read_binary<size_t>(stream);
      if (node_index >= nodes.size()) {
        throw std::runtime_error("Graph state references an invalid consumer node index");
      }
      consumers.push_back(nodes[node_index]);
    }

    graph.add_edge(layers[layer_index], producers, consumers);
  }

  if (version == 1) {
    for (const auto &node : nodes) {
      if (graph.node_in_degree(node) == 0) {
        graph.set_input(node);
      }
      if (graph.node_out_degree(node) == 0) {
        graph.set_output(node);
      }
    }
  }

  graph.compile(allocator);

  const size_t param_layer_count = read_binary<size_t>(stream);
  if (param_layer_count != layers.size()) {
    throw std::runtime_error("Graph state parameter section does not match layer count");
  }

  for (size_t i = 0; i < layers.size(); ++i) {
    Vec<Tensor> params = layers[i]->parameters();
    const size_t param_count = read_binary<size_t>(stream);
    if (param_count != params.size()) {
      throw std::runtime_error("Graph state parameter count does not match layer definition");
    }
    for (auto &param : params) {
      load_into(stream, param);
    }
  }

  return graph;
}

}  // namespace synet