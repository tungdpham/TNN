#pragma once

#include <string>

#include "nn/graph.hpp"

namespace synet::test {

template <typename LayerRefT>
Graph compile_single_layer(LayerRefT &layer, IAllocator &allocator,
                           const std::string &input_uid = "input",
                           const std::string &output_uid = "output") {
  Graph graph;
  Node input = graph.make_node(input_uid);
  graph.set_input(input);
  Node output = layer(input);
  output->set_uid(output_uid);
  graph.set_output(output);
  graph.compile(allocator);
  return graph;
}

template <typename FirstLayerRefT, typename SecondLayerRefT>
Graph compile_two_layer_chain(FirstLayerRefT &first_layer, SecondLayerRefT &second_layer,
                              IAllocator &allocator, const std::string &input_uid = "input",
                              const std::string &output_uid = "output") {
  Graph graph;
  Node input = graph.make_node(input_uid);
  graph.set_input(input);
  Node output = second_layer(first_layer(input));
  output->set_uid(output_uid);
  graph.set_output(output);
  graph.compile(allocator);
  return graph;
}

}  // namespace synet::test