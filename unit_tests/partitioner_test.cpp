#include <gtest/gtest.h>

#include "nn/layers.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "partitioner/weighted_partitioner.hpp"
#include "tensor/tensor_factory.hpp"
#include "type/type.hpp"

using namespace tnn;
using namespace std;

/**
 * Test fixture for Partitioner validation tests.
 * These tests verify the correctness of different partitioning strategies
 * including naive and weighted partitioning.
 */
class PartitionerTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { initializeDefaultDevices(); }

  void SetUp() override {}
};

TEST_F(PartitionerTest, NaivePipelineModelPartition) {
  vector<std::unique_ptr<Layer>> layers = LayerBuilder()
                                              .input({224, 224, 3})
                                              .conv2d(64, 3, 3, 1, 1, 0, 0)
                                              .batchnorm(1e-5, 0.1f, true, SBool::TRUE)
                                              .maxpool2d(2, 2, 2, 2, 0, 0)
                                              .basic_residual_block(64, 128)
                                              .basic_residual_block(128, 256)
                                              .basic_residual_block(256, 256)
                                              .maxpool2d(2, 2, 2, 2, 0, 0)
                                              .conv2d(512, 3, 3, 1, 1)
                                              .batchnorm()
                                              .conv2d(512, 3, 3, 1, 1)
                                              .batchnorm()
                                              .maxpool2d(2, 2, 2, 2, 0, 0)
                                              .basic_residual_block(512, 512)
                                              .basic_residual_block(512, 512)
                                              .maxpool2d(2, 2, 2, 2, 0, 0)
                                              .avgpool2d(7, 7, 1, 1, 0, 0)
                                              .flatten()
                                              .dense(4096)
                                              .dense(1000)
                                              .build();

  NaivePartitionerConfig config{{4, 3, 2}};
  NaivePipelinePartitioner partitioner(config);

  size_t num_partitions = 3;
  vector<Layer *> layer_ptrs;
  for (const auto &layer : layers) {
    layer_ptrs.push_back(layer.get());
  }

  auto partitions = partitioner.partition_model(layer_ptrs);

  size_t current_layer = 0;
  for (const auto &part : partitions) {
    EXPECT_EQ(part.start_offset, current_layer);
    current_layer += part.length;
  }
  EXPECT_EQ(current_layer, layers.size());

  EXPECT_EQ(partitions.size(), num_partitions);
}

TEST_F(PartitionerTest, NaivePipelineInputPartition) {
  NaivePartitionerConfig config{{1, 1}};
  NaivePipelinePartitioner partitioner(config);

  auto input = make_tensor(DType_t::FP32, {32, 224, 224, 3});
  auto labels = make_tensor(DType_t::FP32, {32, 1000});

  auto input_partitions = partitioner.partition_input(input, labels);

  EXPECT_EQ(input_partitions.size(), 1);
  EXPECT_EQ(input_partitions[0].start_offset, 0);
  EXPECT_EQ(input_partitions[0].length, 32);
}

TEST_F(PartitionerTest, NaiveDataModelPartition) {
  vector<std::unique_ptr<Layer>> layers = LayerBuilder()
                                              .input({224, 224, 3})
                                              .conv2d(64, 3, 3, 1, 1)
                                              .batchnorm()
                                              .maxpool2d(2, 2, 2, 2, 0, 0)
                                              .dense(1000)
                                              .build();

  NaivePartitionerConfig config{{1, 1, 1}};
  NaiveDataPartitioner partitioner(config);

  vector<Layer *> layer_ptrs;
  for (const auto &layer : layers) {
    layer_ptrs.push_back(layer.get());
  }

  auto partitions = partitioner.partition_model(layer_ptrs);

  EXPECT_EQ(partitions.size(), 1);
  EXPECT_EQ(partitions[0].start_offset, 0);
  EXPECT_EQ(partitions[0].length, layers.size());
}

TEST_F(PartitionerTest, NaiveDataInputPartition) {
  NaivePartitionerConfig config{{2, 3, 1}};
  NaiveDataPartitioner partitioner(config);

  auto input = make_tensor(DType_t::FP32, {60, 224, 224, 3});
  auto labels = make_tensor(DType_t::FP32, {60, 1000});

  auto input_partitions = partitioner.partition_input(input, labels);

  EXPECT_EQ(input_partitions.size(), 3);

  size_t current_idx = 0;
  for (const auto &part : input_partitions) {
    EXPECT_EQ(part.start_offset, current_idx);
    current_idx += part.length;
  }
  EXPECT_EQ(current_idx, 60);

  EXPECT_EQ(input_partitions[0].length, 20);
  EXPECT_EQ(input_partitions[1].length, 30);
  EXPECT_EQ(input_partitions[2].length, 10);
}

TEST_F(PartitionerTest, NaiveDataInputPartitionUneven) {
  NaivePartitionerConfig config{{1, 1, 1}};
  NaiveDataPartitioner partitioner(config);

  auto input = make_tensor(DType_t::FP32, {10, 224, 224, 3});
  auto labels = make_tensor(DType_t::FP32, {10, 1000});

  auto input_partitions = partitioner.partition_input(input, labels);

  EXPECT_EQ(input_partitions.size(), 3);

  size_t total = 0;
  for (const auto &part : input_partitions) {
    total += part.length;
  }
  EXPECT_EQ(total, 10);
}

TEST_F(PartitionerTest, WeightedPartitionerModelPartition) {
  vector<std::unique_ptr<Layer>> layers = LayerBuilder()
                                              .input({32, 32, 3})
                                              .conv2d(16, 3, 3, 1, 1)
                                              .batchnorm()
                                              .maxpool2d(2, 2, 2, 2, 0, 0)
                                              .conv2d(32, 3, 3, 1, 1)
                                              .batchnorm()
                                              .maxpool2d(2, 2, 2, 2, 0, 0)
                                              .flatten()
                                              .dense(128)
                                              .dense(10)
                                              .build();

  WeightedPartitionerConfig config{{2, 1}, {64, 32, 32, 3}};
  WeightedPipelinePartitioner partitioner(config);

  vector<Layer *> layer_ptrs;
  for (const auto &layer : layers) {
    layer_ptrs.push_back(layer.get());
  }

  auto partitions = partitioner.partition_model(layer_ptrs);

  EXPECT_EQ(partitions.size(), 2);

  size_t current_layer = 0;
  for (const auto &part : partitions) {
    EXPECT_EQ(part.start_offset, current_layer);
    current_layer += part.length;
  }
  EXPECT_EQ(current_layer, layers.size());
  EXPECT_EQ(partitions.size(), 2);
}

TEST_F(PartitionerTest, WeightedPartitionerEqualWeights) {
  vector<std::unique_ptr<Layer>> layers = LayerBuilder()
                                              .input({28, 28, 1})
                                              .conv2d(8, 3, 3, 1, 1)
                                              .conv2d(16, 3, 3, 1, 1)
                                              .conv2d(32, 3, 3, 1, 1)
                                              .flatten()
                                              .dense(10)
                                              .build();

  WeightedPartitionerConfig config{{1, 1, 1}, {64, 28, 28, 1}};
  WeightedPipelinePartitioner partitioner(config);

  vector<Layer *> layer_ptrs;
  for (const auto &layer : layers) {
    layer_ptrs.push_back(layer.get());
  }

  auto partitions = partitioner.partition_model(layer_ptrs);

  EXPECT_EQ(partitions.size(), 3);

  size_t total_layers = 0;
  for (const auto &part : partitions) {
    total_layers += part.length;
  }
  EXPECT_EQ(total_layers, layers.size());
}

TEST_F(PartitionerTest, WeightedPartitionerManyWorkers) {
  vector<std::unique_ptr<Layer>> layers = LayerBuilder()
                                              .input({32, 32, 3})
                                              .conv2d(16, 3, 3, 1, 1)
                                              .conv2d(16, 3, 3, 1, 1)
                                              .conv2d(32, 3, 3, 1, 1)
                                              .conv2d(32, 3, 3, 1, 1)
                                              .conv2d(64, 3, 3, 1, 1)
                                              .flatten()
                                              .dense(10)
                                              .build();

  WeightedPartitionerConfig config{{4, 2, 2, 1}, {64, 32, 32, 3}};
  WeightedPipelinePartitioner partitioner(config);

  vector<Layer *> layer_ptrs;
  for (const auto &layer : layers) {
    layer_ptrs.push_back(layer.get());
  }

  auto partitions = partitioner.partition_model(layer_ptrs);

  EXPECT_EQ(partitions.size(), 4);

  size_t current_layer = 0;
  for (const auto &part : partitions) {
    EXPECT_EQ(part.start_offset, current_layer);
    EXPECT_GT(part.length, 0);
    current_layer += part.length;
  }
  EXPECT_EQ(current_layer, layers.size());
}

TEST_F(PartitionerTest, NaivePipelineEqualProportions) {
  vector<std::unique_ptr<Layer>> layers = LayerBuilder()
                                              .input({28, 28, 1})
                                              .conv2d(16, 3, 3, 1, 1)
                                              .conv2d(16, 3, 3, 1, 1)
                                              .conv2d(16, 3, 3, 1, 1)
                                              .flatten()
                                              .dense(10)
                                              .build();

  NaivePartitionerConfig config{{1, 1}};
  NaivePipelinePartitioner partitioner(config);

  vector<Layer *> layer_ptrs;
  for (const auto &layer : layers) {
    layer_ptrs.push_back(layer.get());
  }

  auto partitions = partitioner.partition_model(layer_ptrs);

  EXPECT_EQ(partitions.size(), 2);

  size_t total = 0;
  for (const auto &part : partitions) {
    total += part.length;
  }
  EXPECT_EQ(total, layers.size());
}

TEST_F(PartitionerTest, NaivePipelineSinglePartition) {
  vector<std::unique_ptr<Layer>> layers =
      LayerBuilder().input({28, 28, 1}).conv2d(16, 3, 3, 1, 1).flatten().dense(10).build();

  NaivePartitionerConfig config{{1}};
  NaivePipelinePartitioner partitioner(config);

  vector<Layer *> layer_ptrs;
  for (const auto &layer : layers) {
    layer_ptrs.push_back(layer.get());
  }

  auto partitions = partitioner.partition_model(layer_ptrs);

  EXPECT_EQ(partitions.size(), 1);
  EXPECT_EQ(partitions[0].start_offset, 0);
  EXPECT_EQ(partitions[0].length, layers.size());
}

TEST_F(PartitionerTest, NaiveDataSinglePartition) {
  NaivePartitionerConfig config{{1}};
  NaiveDataPartitioner partitioner(config);

  auto input = make_tensor(DType_t::FP32, {64, 224, 224, 3});
  auto labels = make_tensor(DType_t::FP32, {64, 1000});

  auto input_partitions = partitioner.partition_input(input, labels);

  EXPECT_EQ(input_partitions.size(), 1);
  EXPECT_EQ(input_partitions[0].start_offset, 0);
  EXPECT_EQ(input_partitions[0].length, 64);
}
