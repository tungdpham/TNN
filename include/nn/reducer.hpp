#pragma once

#include "nn/graph_builder.hpp"
#include "nn/reduction_strategy.hpp"
#include "nn/reductions_impl/seq_reduction.hpp"
#include "type/type.hpp"

namespace tnn {
class Reducer {
public:
  Reducer(Vec<std::unique_ptr<ReductionStrategy>> reduction_strategies)
      : reduction_strategies_(std::move(reduction_strategies)) {}

  // takes a sorted graph and eliminates redundant nodes/layers, e.g. consecutive linear layers can
  // be merged into one
  void reduce(GraphBuilder& builder) {
    for (auto& strategy : reduction_strategies_) {
      strategy->reduce(builder);
    }
  }

private:
  Vec<std::unique_ptr<ReductionStrategy>> reduction_strategies_;
};

class ReducerBuilder {
  Vec<std::unique_ptr<ReductionStrategy>> reduction_strategies_;

public:
  ReducerBuilder() = default;

  ReducerBuilder& add_strategy(std::unique_ptr<ReductionStrategy> strategy) {
    reduction_strategies_.push_back(std::move(strategy));
    return *this;
  }

  ReducerBuilder& seq_reduction() { return add_strategy(std::make_unique<SeqReduction>()); }

  Reducer build() { return Reducer(std::move(reduction_strategies_)); }
};

}  // namespace tnn