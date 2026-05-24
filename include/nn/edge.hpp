#include <initializer_list>

#include "nn/io_node.hpp"
#include "nn/op_node.hpp"

namespace tnn {
class Edge {
public:
  Edge(Vec<const IONode*> producers, Vec<const IONode*> consumers, OpNode& op_node)
      : producers_(std::move(producers)),
        consumers_(std::move(consumers)),
        op_node_(op_node) {}

  Edge(std::initializer_list<const IONode*> producers,
       std::initializer_list<const IONode*> consumers, OpNode& op_node)
      : producers_(producers),
        consumers_(consumers),
        op_node_(op_node) {}

  const Vec<const IONode*>& producers() const { return producers_; }
  const Vec<const IONode*>& consumers() const { return consumers_; }
  OpNode& op_node() const { return op_node_; }

private:
  Vec<const IONode*> producers_;
  Vec<const IONode*> consumers_;
  OpNode& op_node_;
};
}  // namespace tnn