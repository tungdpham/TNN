#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "nn/graph_api.hpp"
#include "nn/layers.hpp"
#include "partitioner/graph_partitioner.hpp"

using namespace tnn;

namespace {

Graph build_linear_graph() {
  Graph graph;

  Node input = graph.make_node("input");
  Node hidden_a = DenseLayer(8, 8, false, "dense_a")(input);
  hidden_a->set_uid("hidden_a");
  Node hidden_b = DenseLayer(8, 8, false, "dense_b")(hidden_a);
  hidden_b->set_uid("hidden_b");
  Node hidden_c = DenseLayer(8, 8, false, "dense_c")(hidden_b);
  hidden_c->set_uid("hidden_c");
  Node output = DenseLayer(8, 8, false, "dense_d")(hidden_c);
  output->set_uid("output");

  return graph;
}

Graph build_branched_graph() {
  Graph graph;

  Node input = graph.make_node("input");
  Node left = DenseLayer(8, 8, false, "left_dense")(input);
  left->set_uid("left");
  Node right = DenseLayer(8, 8, false, "right_dense")(input);
  right->set_uid("right");
  Node merged = AddLayer("merge")(left, right);
  merged->set_uid("merged");
  Node output = DenseLayer(8, 8, false, "tail_dense")(merged);
  output->set_uid("output");

  return graph;
}

}  // namespace

TEST(GraphPartitionerTest, SplitsLinearGraphIntoRequestedLayerCounts) {
  Graph graph = build_linear_graph();
  GraphPartitioner partitioner({1, 3});

  std::vector<GraphPartition> partitions = partitioner.partition(graph);

  ASSERT_EQ(partitions.size(), 2);
  EXPECT_EQ(partitions[0].start_layer, 0);
  EXPECT_EQ(partitions[0].layer_count, 1);
  EXPECT_EQ(partitions[0].graph.edges().size(), 1);
  EXPECT_EQ(partitions[0].input_uids, (std::vector<std::string>{"input"}));
  EXPECT_EQ(partitions[0].output_uids, (std::vector<std::string>{"hidden_a"}));

  EXPECT_EQ(partitions[1].start_layer, 1);
  EXPECT_EQ(partitions[1].layer_count, 3);
  EXPECT_EQ(partitions[1].graph.edges().size(), 3);
  EXPECT_EQ(partitions[1].input_uids, (std::vector<std::string>{"hidden_a"}));
  EXPECT_EQ(partitions[1].output_uids, (std::vector<std::string>{"output"}));
}

TEST(GraphPartitionerTest, PreservesBoundaryNodesForBranchedGraph) {
  Graph graph = build_branched_graph();
  GraphPartitioner partitioner({2, 2});

  std::vector<GraphPartition> partitions = partitioner.partition(graph);

  ASSERT_EQ(partitions.size(), 2);
  EXPECT_EQ(partitions[0].graph.edges().size(), 2);
  EXPECT_EQ(partitions[0].input_uids, (std::vector<std::string>{"input"}));
  EXPECT_EQ(partitions[0].output_uids, (std::vector<std::string>{"left", "right"}));

  EXPECT_EQ(partitions[1].graph.edges().size(), 2);
  EXPECT_EQ(partitions[1].input_uids, (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(partitions[1].output_uids, (std::vector<std::string>{"output"}));
}

TEST(GraphPartitionerTest, RejectsPartitionSizesThatDoNotMatchLayerCount) {
  Graph graph = build_linear_graph();
  GraphPartitioner partitioner({2, 1});

  EXPECT_THROW(partitioner.partition(graph), std::runtime_error);
}