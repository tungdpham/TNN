#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/graph_api.hpp"
#include "nn/layers.hpp"
#include "partitioner/graph_partitioner.hpp"
#include "tensor/tensor.hpp"
#include "test_graph_utils.hpp"

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

class GraphPlannerStateTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }
};

TEST(GraphPartitionerTest, SplitsLinearGraphIntoRequestedLayerCounts) {
  Graph graph = build_linear_graph();
  GraphPartitioner partitioner({0.25, 0.75});

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
  GraphPartitioner partitioner({0.5, 0.5});

  std::vector<GraphPartition> partitions = partitioner.partition(graph);

  ASSERT_EQ(partitions.size(), 2);
  EXPECT_EQ(partitions[0].graph.edges().size(), 2);
  EXPECT_EQ(partitions[0].input_uids, (std::vector<std::string>{"input"}));
  EXPECT_EQ(partitions[0].output_uids, (std::vector<std::string>{"left", "right"}));

  EXPECT_EQ(partitions[1].graph.edges().size(), 2);
  EXPECT_EQ(partitions[1].input_uids, (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(partitions[1].output_uids, (std::vector<std::string>{"output"}));
}

TEST(GraphPartitionerTest, RejectsPartitionRatiosThatResolveToEmptyPartitions) {
  Graph graph = build_linear_graph();
  GraphPartitioner partitioner({0.95, 0.05, 0.01, 0.01, 0.01});

  EXPECT_THROW(partitioner.partition(graph), std::runtime_error);
}

TEST(GraphPartitionerTest, ClassifiesLinearGraphAsSingleSPSCRegion) {
  Graph graph = build_linear_graph();

  GraphRegionSummary regions = graph.classify_regions();

  ASSERT_EQ(regions.mpsc_regions.size(), 0u);
  ASSERT_EQ(regions.spsc_regions.size(), 1u);
  ASSERT_TRUE(regions.unsupported_edge_indices.empty());

  const SPSCRegion &region = regions.spsc_regions.front();
  ASSERT_EQ(region.edge_indices.size(), 4u);
  EXPECT_EQ(graph.edges()[region.edge_indices.front()]->layer()->name(), "dense_a");
  EXPECT_EQ(graph.edges()[region.edge_indices.back()]->layer()->name(), "dense_d");
}

TEST(GraphPartitionerTest, ClassifiesBranchedGraphAsSingleJoinMPSCPlusTailSPSC) {
  Graph graph = build_branched_graph();

  GraphRegionSummary regions = graph.classify_regions();

  ASSERT_EQ(regions.mpsc_regions.size(), 1u);
  ASSERT_EQ(regions.spsc_regions.size(), 1u);
  ASSERT_TRUE(regions.unsupported_edge_indices.empty());

  const MPSCRegion &mpsc = regions.mpsc_regions.front();
  EXPECT_EQ(graph.edges()[mpsc.join_edge_index]->layer()->name(), "merge");
  ASSERT_EQ(mpsc.branch_edge_indices.size(), 2u);
  EXPECT_EQ(mpsc.branch_edge_indices[0].size(), 1u);
  EXPECT_EQ(mpsc.branch_edge_indices[1].size(), 1u);
  EXPECT_EQ(graph.edges()[mpsc.branch_edge_indices[0].front()]->layer()->name(), "left_dense");
  EXPECT_EQ(graph.edges()[mpsc.branch_edge_indices[1].front()]->layer()->name(), "right_dense");

  const SPSCRegion &tail = regions.spsc_regions.front();
  ASSERT_EQ(tail.edge_indices.size(), 1u);
  EXPECT_EQ(graph.edges()[tail.edge_indices.front()]->layer()->name(), "tail_dense");
}

TEST_F(GraphPlannerStateTest, CompileUsesGraphLocalWorkspaceAllocator) {
  auto layer_a = DenseLayer(4, 2, false, "dense_a");
  auto layer_b = DenseLayer(4, 2, false, "dense_b");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);

  Graph graph_a = test::compile_single_layer(layer_a, allocator);
  Graph graph_b = test::compile_single_layer(layer_b, allocator);

  ASSERT_NE(graph_a.workspace_allocator(), nullptr);
  ASSERT_NE(graph_b.workspace_allocator(), nullptr);
  EXPECT_NE(graph_a.workspace_allocator(), graph_b.workspace_allocator());
  EXPECT_EQ(graph_a.cached_forward_plan_count(), 0u);
  EXPECT_EQ(graph_b.cached_forward_plan_count(), 0u);
}

TEST_F(GraphPlannerStateTest, ForwardPlanCacheKeysOnModeAndShape) {
  auto dense = DenseLayer(4, 2, false, "planner_dense");
  auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
  Graph graph = test::compile_single_layer(dense, allocator);

  Tensor input = make_tensor<float>({1, 4}, getHost());
  input->fill(1.0f);
  InputMap eval_inputs({{"input", input}});

  graph.set_mode(ExecutionMode::EVAL);
  graph.forward(eval_inputs);

  EXPECT_TRUE(graph.has_cached_forward_plan(eval_inputs));
  EXPECT_TRUE(graph.cached_forward_plan_profiled(eval_inputs));
  EXPECT_EQ(graph.cached_forward_plan_spsc_region_count(eval_inputs), 1u);
  EXPECT_EQ(graph.cached_forward_plan_mpsc_region_count(eval_inputs), 0u);
  EXPECT_EQ(graph.cached_forward_plan_profiled_edge_count(eval_inputs), 1u);
  EXPECT_EQ(graph.cached_forward_plan_execution_count(eval_inputs), 1u);
  EXPECT_EQ(graph.cached_forward_plan_count(), 1u);

  graph.forward(eval_inputs);
  EXPECT_EQ(graph.cached_forward_plan_execution_count(eval_inputs), 2u);

  Tensor wider_batch = make_tensor<float>({2, 4}, getHost());
  wider_batch->fill(2.0f);
  InputMap wider_inputs({{"input", wider_batch}});

  graph.forward(wider_inputs);
  EXPECT_TRUE(graph.has_cached_forward_plan(wider_inputs));
  EXPECT_EQ(graph.cached_forward_plan_count(), 2u);

  graph.set_mode(ExecutionMode::TRAIN);
  graph.forward(eval_inputs);
  EXPECT_TRUE(graph.has_cached_forward_plan(eval_inputs));
  EXPECT_EQ(graph.cached_forward_plan_execution_count(eval_inputs), 1u);
  EXPECT_EQ(graph.cached_forward_plan_count(), 3u);

  graph.set_mode(ExecutionMode::EVAL);
  EXPECT_EQ(graph.cached_forward_plan_execution_count(eval_inputs), 2u);
}