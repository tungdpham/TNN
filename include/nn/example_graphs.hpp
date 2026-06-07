/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "nn/graph.hpp"

namespace tnn {

class ExampleGraphs {
private:
  static std::unordered_map<std::string, std::function<Graph(IAllocator &)>> creators_;

public:
  static void register_graph(const std::string &name, std::function<Graph(IAllocator &)> creator) {
    creators_[name] = std::move(creator);
  }

  static Graph create(const std::string &name, IAllocator &allocator) {
    auto it = creators_.find(name);
    if (it != creators_.end()) {
      return it->second(allocator);
    }
    throw std::invalid_argument("Unknown graph: " + name);
  }

  static Vec<std::string> available_graphs() {
    Vec<std::string> graphs;
    for (const auto &pair : creators_) {
      graphs.push_back(pair.first);
    }
    return graphs;
  }

  static void register_defaults();
};

inline Graph load_or_create_graph(const std::string &graph_name, const std::string &graph_path,
                                  IAllocator &allocator) {
  if (!graph_path.empty()) {
    std::cout << "Loading graph from: " << graph_path << std::endl;
    std::ifstream file(graph_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open graph file");
    }
    auto graph = Graph::load_state(file, allocator);
    file.close();
    return graph;
  }
  try {
    auto graph = ExampleGraphs::create(graph_name, allocator);
    std::cout << "Created graph: " << graph_name << std::endl;
    return graph;
  } catch (const std::exception &e) {
    std::cerr << "Error creating graph: " << e.what() << std::endl;
    std::cout << "Available graphs are: ";
    for (const auto &name : ExampleGraphs::available_graphs()) {
      std::cout << name << "\n";
    }
    std::cout << std::endl;
    throw std::runtime_error("Failed to create graph");
  }
}

}  // namespace tnn