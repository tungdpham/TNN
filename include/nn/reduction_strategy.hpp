#pragma once

#include "nn/graph_builder.hpp"

namespace tnn {
class ReductionStrategy {
public:
  virtual ~ReductionStrategy() = default;

  // takes a sorted graph and eliminates redundant nodes/layers, e.g. consecutive linear layers can
  // be merged into one
  virtual void reduce(GraphBuilder& builder) = 0;
};
}  // namespace tnn