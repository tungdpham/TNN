#pragma once

#include "nn/layer.hpp"
#include "nn/node.hpp"

namespace synet {

class EdgeImpl {
public:
  EdgeImpl(std::shared_ptr<LayerImpl> layer, const Vec<Node> &inputs, const Vec<Node> &outputs)
      : layer_(layer),
        producers_(inputs),
        consumers_(outputs) {}

  EdgeImpl(std::shared_ptr<LayerImpl> layer, std::initializer_list<Node> inputs,
           std::initializer_list<Node> outputs)
      : layer_(layer),
        producers_(inputs),
        consumers_(outputs) {}

  const Vec<Node> &producers() const { return producers_; }
  const Vec<Node> &consumers() const { return consumers_; }
  std::shared_ptr<LayerImpl> layer() const { return layer_; }

private:
  std::shared_ptr<LayerImpl> layer_;
  Vec<Node> producers_;
  Vec<Node> consumers_;
};

using Edge = std::shared_ptr<EdgeImpl>;

}  // namespace synet