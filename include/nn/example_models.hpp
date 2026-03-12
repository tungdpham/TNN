/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <iosfwd>

#include "nn/blocks_impl/sequential.hpp"
#include "nn/graph.hpp"
#include "nn/graph_builder.hpp"
#include "nn/io_node.hpp"
#include "nn/layers.hpp"
#include "nn/op_node.hpp"
namespace tnn {

class ExampleModels {
private:
  static std::unordered_map<std::string, std::function<Sequential(DType_t)>> creators_;

public:
  static void register_model(std::function<Sequential(DType_t)> creator) {
    Sequential model = creator(DType_t::FP32);
    std::string model_name = model.name();
    creators_[model_name] = [creator](DType_t io_dtype) { return creator(io_dtype); };
  }

  static Sequential create(const std::string &name, DType_t io_dtype_ = DType_t::FP32) {
    auto it = creators_.find(name);
    if (it != creators_.end()) {
      return it->second(io_dtype_);
    }
    throw std::invalid_argument("Unknown model: " + name);
  }

  static std::vector<std::string> available_models() {
    std::vector<std::string> models;
    for (const auto &pair : creators_) {
      models.push_back(pair.first);
    }
    return models;
  }

  static void register_defaults();
};

inline Graph load_or_create_model(const std::string &model_name, const std::string &model_path,
                                  IAllocator &allocator) {
  GraphBuilder builder;
  std::unique_ptr<Sequential> model;
  if (!model_path.empty()) {
    std::cout << "Loading model from: " << model_path << std::endl;
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open model file");
    }
    auto model = Graph::load_state(file, allocator);
    file.close();
    return model;
  } else {
    try {
      auto model = std::make_unique<Sequential>(ExampleModels::create(model_name));
      std::string model_name = model->name();
      IONode &input_node = builder.input("input");
      OpNode &op_node = builder.add_layer(std::move(model));
      builder.output(op_node, input_node, "output");
      Graph graph = builder.compile(allocator);
      graph.set_name(model_name);
      std::cout << "Created model: " << model_name << std::endl;
      return graph;
    } catch (const std::exception &e) {
      std::cerr << "Error creating model: " << e.what() << std::endl;
      std::cout << "Available models are: ";
      for (const auto &name : ExampleModels::available_models()) {
        std::cout << name << "\n";
      }
      std::cout << std::endl;
      throw std::runtime_error("Failed to create model");
    }
  }
}

}  // namespace tnn