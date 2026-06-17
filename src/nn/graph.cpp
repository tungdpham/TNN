/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/graph.hpp"

#include <array>
#include <cstdint>
#include <istream>
#include <ostream>
#include <queue>
#include <unordered_map>

#include "nn/layer.hpp"
#include "nn/layers.hpp"
#include "tensor/tensor_ops.hpp"

namespace synet {

void Graph::compile(IAllocator &allocator) {
  sort();
  GraphContextDescriptor ctx_desc;
  std::set<LayerImpl *> unique_layers;
  for (const auto &edge : edges_) {
    LayerImpl *layer_ptr = edge->layer().get();
    if (unique_layers.count(layer_ptr) == 0) {
      unique_layers.insert(layer_ptr);
    }
  }
  for (LayerImpl *layer_ptr : unique_layers) {
    for (const auto &param_desc : layer_ptr->param_descriptors()) {
      ctx_desc.register_desc(param_desc);
    }
  }
  context_ = std::make_unique<GraphContext>(allocator, ctx_desc);
  workspace_allocator_ = DELAllocatorV2::create(context_->device(), defaultFlowHandle);
  for (LayerImpl *layer_ptr : unique_layers) {
    layer_ptr->set_engine_type(allocator.device().get_engine());
    layer_ptr->set_allocator(*workspace_allocator_);
    layer_ptr->init();
  }

  // sanity check: ensure there are output nodes
  if (output_nodes_.empty()) {
    throw std::runtime_error("Graph must have at least one output node");
  }
}

Vec<std::string> Graph::input_uids() const {
  Vec<std::string> uids;
  uids.reserve(input_nodes_.size());
  for (const auto &node : nodes_) {
    if (is_input(node)) {
      uids.push_back(node->uid());
    }
  }
  return uids;
}

Vec<std::string> Graph::output_uids() const {
  Vec<std::string> uids;
  uids.reserve(output_nodes_.size());
  for (const auto &node : nodes_) {
    if (is_output(node)) {
      uids.push_back(node->uid());
    }
  }
  return uids;
}

void Graph::add_edge(std::shared_ptr<LayerImpl> layer, const Vec<Node> &producers,
                     const Vec<Node> &consumers) {
  Edge edge = std::make_shared<EdgeImpl>(layer, producers, consumers);
  edges_.push_back(edge);
  on_add_edge(edge);
}

void Graph::add_edge(std::shared_ptr<LayerImpl> layer, std::initializer_list<Node> producers,
                     std::initializer_list<Node> consumers) {
  Edge edge = std::make_shared<EdgeImpl>(layer, producers, consumers);
  edges_.push_back(edge);
  on_add_edge(edge);
}

void Graph::sort() {
  std::set<std::weak_ptr<NodeImpl>, std::owner_less<std::weak_ptr<NodeImpl>>> produced_nodes;
  for (const auto &edge : edges_) {
    for (const auto &consumer : edge->consumers()) {
      produced_nodes.insert(consumer);
    }
  }

  std::map<std::weak_ptr<NodeImpl>, Vec<Edge>, std::owner_less<std::weak_ptr<NodeImpl>>>
      node_to_dependent_edges;
  for (const auto &edge : edges_) {
    for (const auto &producer : edge->producers()) {
      node_to_dependent_edges[producer].push_back(edge);
    }
  }

  std::map<Edge, int> pending;
  std::queue<Edge> ready;

  for (const auto &edge : edges_) {
    int count = 0;
    for (const auto &producer : edge->producers()) {
      if (produced_nodes.count(producer)) {
        ++count;
      }
    }
    pending[edge] = count;
    if (count == 0) {
      ready.push(edge);
    }
  }

  Vec<Edge> sorted;
  sorted.reserve(edges_.size());

  while (!ready.empty()) {
    Edge e = ready.front();
    ready.pop();
    sorted.push_back(e);

    for (const auto &consumer : e->consumers()) {
      auto it = node_to_dependent_edges.find(consumer);
      if (it != node_to_dependent_edges.end()) {
        for (const auto &dep_edge : it->second) {
          if (--pending[dep_edge] == 0) {
            ready.push(dep_edge);
          }
        }
      }
    }
  }

  if (sorted.size() != edges_.size()) {
    throw std::runtime_error("Graph contains a cycle; topological sort failed");
  }

  edges_ = std::move(sorted);

  std::set<NodeImpl *> placed;
  Vec<Node> sorted_nodes;
  sorted_nodes.reserve(nodes_.size());

  for (const auto &node : nodes_) {
    if (!produced_nodes.count(node)) {
      sorted_nodes.push_back(node);
      placed.insert(node.get());
    }
  }

  for (const auto &edge : edges_) {
    for (const auto &consumer : edge->consumers()) {
      if (placed.insert(consumer.get()).second) {
        sorted_nodes.push_back(consumer);
      }
    }
  }

  nodes_ = std::move(sorted_nodes);
}
TensorBundle Graph::forward(TensorBundle &input_map, size_t pid) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : nodes_) {
    uid_to_node[node->uid()] = node;
  }
  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Input UID not found in graph: " + uid);
    }
    auto node = it->second;
    Tensor device_tensor = tensor;
    if (tensor.device() != context_->device()) {
      device_tensor = tensor.to_device(context_->device());
    }
    it->second->set_data(pid, device_tensor, out_degree_[node]);
  }

  size_t hook_id = 0;
  bool hook_registered = false;
  size_t edge_peak_usage = workspace_allocator_ ? workspace_allocator_->total_allocated() : 0;
  if (workspace_allocator_) {
    hook_id = workspace_allocator_->add_allocation_hook([&edge_peak_usage](size_t current_usage) {
      edge_peak_usage = std::max(edge_peak_usage, current_usage);  // Hook to track peak memory
    });
    hook_registered = true;
  }

  TensorBundle output_map;
  for (size_t edge_index = 0; edge_index < edges_.size(); ++edge_index) {
    size_t usage_before = workspace_allocator_ ? workspace_allocator_->total_allocated() : 0;
    edge_peak_usage = usage_before;
    forward_edge(edges_[edge_index], pid);

    for (const auto &consumer : edges_[edge_index]->consumers()) {
      if (is_output(consumer)) {
        output_map.set(consumer->uid(), consumer->data(pid));
      }
    }
  }

  if (hook_registered) {
    workspace_allocator_->remove_allocation_hook(hook_id);  // Unregister hook
  }

  // clean up boundary node data
  for (Node &node : nodes_) {
    if (node->data_ref_count(pid) == 0) {
      node->clear_data(pid);
    }
  }

  return output_map;
}

TensorBundle Graph::backward(TensorBundle &output_grad_map, size_t pid) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : nodes_) {
    uid_to_node[node->uid()] = node;
  }
  for (const auto &[uid, tensor] : output_grad_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Output UID not found in graph: " + uid);
    }
    auto node = it->second;
    Tensor device_tensor = tensor;
    if (tensor.device() != context_->device()) {
      device_tensor = tensor.to_device(context_->device());
    }
    it->second->set_grad(pid, device_tensor, in_degree_[node]);
  }
  TensorBundle input_grad_map;
  for (auto it = edges_.rbegin(); it != edges_.rend(); ++it) {
    Edge &edge = *it;
    backward(edge, pid);
    for (auto &producer : edge->producers()) {
      if (is_input(producer)) {
        input_grad_map.set(producer->uid(), producer->grad(pid));
      }
    }
  }

  // clean up boundary node grads
  for (Node &node : nodes_) {
    if (node->grad_ref_count(pid) == 0) {
      node->clear_grad(pid);
    }
  }
  return input_grad_map;
}

Node Graph::make_node(std::string uid) {
  if (uid.empty()) {
    uid = generate_uid();
  } else if (used_uids_.count(uid) > 0) {
    throw std::runtime_error("Duplicate node UID: " + uid);
  } else {
    used_uids_.insert(uid);
  }
  Node node = std::make_shared<NodeImpl>(this, uid);
  nodes_.push_back(node);
  return node;
}

void Graph::set_mode(ExecutionMode mode) {
  mode_ = mode;
  for (const auto &edge : edges_) {
    edge->layer()->set_training(mode == ExecutionMode::TRAIN);
  }
}

void Graph::set_io_dtype(DType_t dtype) {
  for (const auto &edge : edges_) {
    edge->layer()->set_io_dtype(dtype);
  }
}

void Graph::set_param_dtype(DType_t dtype) {
  for (const auto &edge : edges_) {
    edge->layer()->set_param_dtype(dtype);
  }
}

void Graph::set_compute_dtype(DType_t dtype) {
  for (const auto &edge : edges_) {
    edge->layer()->set_compute_dtype(dtype);
  }
}

void Graph::set_input(Node node) {
  if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
    throw std::runtime_error("Input node does not belong to graph");
  }
  input_nodes_.insert(node);
}

void Graph::set_output(Node node) {
  if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
    throw std::runtime_error("Output node does not belong to graph");
  }
  output_nodes_.insert(node);
}

Node Graph::input(const std::string &uid) {
  Node node = make_node(uid);
  set_input(node);
  return node;
}

void Graph::zero_grads() {
  for (const auto &node : nodes_) {
    node->zero_grads();
  }
  context_->zero_grads();
}

Vec<Tensor *> Graph::parameters() { return context_->parameters(); }
Vec<Tensor *> Graph::gradients() { return context_->gradients(); }

int Graph::node_in_degree(const Node &node) const {
  auto it = in_degree_.find(node);
  return it == in_degree_.end() ? 0 : it->second;
}

int Graph::node_out_degree(const Node &node) const {
  auto it = out_degree_.find(node);
  return it == out_degree_.end() ? 0 : it->second;
}

std::string Graph::generate_uid() {
  std::string uid;
  do {
    uid = "node_" + std::to_string(node_count_++);
  } while (used_uids_.count(uid) > 0);
  used_uids_.insert(uid);
  return uid;
}

Vec<Node> Graph::inputs() { return Vec<Node>(input_nodes_.begin(), input_nodes_.end()); }

Vec<Node> Graph::outputs() { return Vec<Node>(output_nodes_.begin(), output_nodes_.end()); }

void Graph::on_add_edge(const Edge &edge) {
  for (const auto &producer : edge->producers()) {
    out_degree_[producer]++;
  }
  for (const auto &consumer : edge->consumers()) {
    in_degree_[consumer]++;
  }
}

void Graph::forward_edge(Edge &edge, size_t pid) {
  Vec<Tensor> input_data;
  for (const auto &producer : edge->producers()) {
    if (!producer->data(pid)) {
      throw std::runtime_error("Null input data while forwarding graph");
    }
    input_data.push_back(producer->data(pid));
    producer->decrement_data_ref_count(pid);
  }
  Residuals residuals;  // can be used to store intermediate results for reuse within the same
                        // forward pass
  Vec<Tensor> output_data = edge->layer()->forward(input_data, residuals);

  edge->set_residuals(pid, std::move(residuals));

  for (size_t i = 0; i < edge->consumers().size(); ++i) {
    Node consumer = edge->consumers()[i];
    consumer->set_data(pid, output_data[i], out_degree_[consumer]);
  }
}

void Graph::backward(Edge &edge, size_t pid) {
  Vec<Tensor> output_grads;
  for (const auto &consumer : edge->consumers()) {
    if (!consumer->grad(pid)) {
      throw std::runtime_error("Null output gradient while backwarding graph");
    }
    output_grads.push_back(consumer->grad(pid));
    consumer->decrement_grad_ref_count(pid);
  }
  Residuals &residuals = edge->residuals(pid);
  Vec<Tensor> input_grads = edge->layer()->backward(output_grads, residuals);

  edge->clear_residuals(pid);
  for (size_t i = 0; i < edge->producers().size(); ++i) {
    Node producer = edge->producers()[i];
    producer->accumulate_grad(pid, input_grads[i], in_degree_[producer]);
  }
}

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
    Vec<ParamDescriptor> descriptors = layer->param_descriptors();
    write_binary(stream, descriptors.size());
    for (const ParamDescriptor &descriptor : descriptors) {
      if (!descriptor.data_ptr) {
        throw std::runtime_error("Cannot save uninitialized layer parameter");
      }
      ops::save_tensor(*descriptor.data_ptr, stream);
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
    Vec<ParamDescriptor> descriptors = layers[i]->param_descriptors();
    const size_t param_count = read_binary<size_t>(stream);
    if (param_count != descriptors.size()) {
      throw std::runtime_error("Graph state parameter count does not match layer definition");
    }
    for (auto &descriptor : descriptors) {
      ops::load_tensor(*descriptor.data_ptr, stream);
    }
  }

  return graph;
}

}  // namespace synet