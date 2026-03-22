#pragma once

#include "nn/layer.hpp"
#include "nn/layers.hpp"

namespace tnn {

struct SeqPartition {
  size_t start_offset;
  size_t length;

  SeqPartition(size_t start, size_t length)
      : start_offset(start),
        length(length) {}
};

struct InputPartition {
  size_t start_offset;
  size_t length;  // exclusive

  InputPartition(size_t start, size_t length)
      : start_offset(start),
        length(length) {}
};

inline Vec<Vec<std::unique_ptr<Layer>>> split(Vec<Layer *> layers, Vec<SeqPartition> &partitions) {
  if (partitions.empty()) {
    throw std::invalid_argument("Partitions vector is empty");
  }
  Vec<Vec<std::unique_ptr<Layer>>> stages;
  stages.reserve(partitions.size());
  for (const auto &part : partitions) {
    if (part.start_offset >= layers.size() || part.start_offset + part.length > layers.size() ||
        part.length == 0) {
      throw std::out_of_range("Invalid partition range");
    }

    auto partition_layers = Vec<std::unique_ptr<Layer>>();
    LayerFactory::register_defaults();
    for (size_t i = part.start_offset; i < part.start_offset + part.length; ++i) {
      auto layer_config = layers[i]->get_config();
      partition_layers.push_back(LayerFactory::create(layer_config));
    }
    stages.push_back(std::move(partition_layers));
  }
  return stages;
}

class Partitioner {
public:
  Partitioner(size_t num_partitions = 1)
      : num_partitions_(num_partitions){};
  virtual ~Partitioner() = default;

  // splits the sequential model into partitions corresponding to each stage
  virtual Vec<SeqPartition> partition_model(const Vec<Layer *> &layers) = 0;

  // splits the input data for each stage.
  virtual Vec<InputPartition> partition_input(const ConstTensor &input,
                                              const ConstTensor &labels) = 0;

protected:
  size_t num_partitions_;
};

}  // namespace tnn