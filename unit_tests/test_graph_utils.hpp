#pragma once

#include <string>

#include "nn/graph_api.hpp"

namespace tnn::test {

template <typename LayerRefT>
Graph compile_single_layer(LayerRefT &layer, IAllocator &allocator,
                           const std::string &input_uid = "input",
                           const std::string &output_uid = "output") {
  Graph graph;
  Node input = graph.make_node(input_uid);
  Node output = layer(input);
  output->set_uid(output_uid);
  graph.compile(allocator);
  return graph;
}

template <typename FirstLayerRefT, typename SecondLayerRefT>
Graph compile_two_layer_chain(FirstLayerRefT &first_layer, SecondLayerRefT &second_layer,
                              IAllocator &allocator, const std::string &input_uid = "input",
                              const std::string &output_uid = "output") {
  Graph graph;
  Node input = graph.make_node(input_uid);
  Node output = second_layer(first_layer(input));
  output->set_uid(output_uid);
  graph.compile(allocator);
  return graph;
}

}  // namespace tnn::test