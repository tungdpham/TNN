/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/example_graphs.hpp"

#include "nn/layers.hpp"
#include "nn/node_ops.hpp"

namespace tnn {

using namespace tnn::graph_api_v2;

std::unordered_map<std::string, std::function<Graph(IAllocator &)>> ExampleGraphs::creators_;

namespace {

LayerRef<ResidualBlockImpl> make_bottleneck_residual_block(
    size_t in_channels, size_t mid_channels, size_t out_channels, size_t stride = 1,
    const std::string &name = "bottleneck_residual_block") {
  auto main_seq = Sequential(
      {Conv2DLayer(in_channels, mid_channels, 1, 1, 1, 1, 0, 0, false, name + "_conv1"),
       BatchNormLayer(mid_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn1"),
       Conv2DLayer(mid_channels, mid_channels, 3, 3, stride, stride, 1, 1, false, name + "_conv2"),
       BatchNormLayer(mid_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn2"),
       Conv2DLayer(mid_channels, out_channels, 1, 1, 1, 1, 0, 0, false, name + "_conv3"),
       BatchNormLayer(out_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn3")});

  Sequential shortcut_seq;
  if (stride != 1 || in_channels != out_channels) {
    shortcut_seq = Sequential({Conv2DLayer(in_channels, out_channels, 1, 1, stride, stride, 0, 0,
                                           false, name + "_shortcut_conv"),
                               BatchNormLayer(out_channels, dtype_eps(DType_t::FP32), 0.1f, true,
                                              true, name + "_shortcut_bn")});
  }

  return make_layer<ResidualBlockImpl>(std::move(main_seq), std::move(shortcut_seq), name);
}

Graph create_add_graph(IAllocator &allocator) {
  Graph graph;
  auto input1 = graph.make_node("input1");
  auto input2 = graph.make_node("input2");

  auto conv2d_1 = Conv2DLayer(3, 32, 3, 3, 1, 1, 1, 1, false, "test_conv2d");
  auto output1 = conv2d_1(input1);

  auto conv2d_2 = Conv2DLayer(3, 32, 3, 3, 1, 1, 1, 1, false, "test_conv2d_2");
  auto output2 = conv2d_2(input2);

  auto output = output1 + output2;
  output->set_uid("output");

  graph.compile(allocator);
  return graph;
}

Graph create_mnist_graph(IAllocator &allocator) {
  Graph graph;
  auto input = graph.make_node("input");

  auto conv2d_1 = Conv2DLayer(1, 8, 5, 5, 1, 1, 0, 0, false, "conv1");
  auto bn1 = BatchNormLayer(8, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn1");
  auto pool1 = MaxPool2DLayer(3, 3, 3, 3, 0, 0, "pool1");
  auto conv2d_2 = Conv2DLayer(8, 16, 1, 1, 1, 1, 0, 0, false, "conv2_1x1");
  auto bn2_1x1 = BatchNormLayer(16, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn2_1x1");
  auto conv2d_3 = Conv2DLayer(16, 48, 5, 5, 1, 1, 0, 0, false, "conv3");
  auto bn3 = BatchNormLayer(48, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn3");
  auto pool2 = MaxPool2DLayer(2, 2, 2, 2, 0, 0, "pool2");
  auto flatten = FlattenLayer(1, -1, "flatten");
  auto fc = DenseLayer(192, 10, false, "output");

  auto x = conv2d_1(input);
  x = bn1(x);
  x = pool1(x);
  x = conv2d_2(x);
  x = bn2_1x1(x);
  x = conv2d_3(x);
  x = bn3(x);
  x = pool2(x);
  x = flatten(x);
  auto output = fc(x);
  output->set_uid("output");

  graph.compile(allocator);
  return graph;
}

Graph create_resnet50_graph(IAllocator &allocator) {
  Graph graph;
  auto input = graph.make_node("input");

  auto conv1 = Conv2DLayer(3, 64, 7, 7, 2, 2, 3, 3, true, "conv1");
  auto bn1 = BatchNormLayer(64, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn1");
  auto maxpool = MaxPool2DLayer(3, 3, 2, 2, 1, 1, "maxpool");

  auto layer1_block1 = make_bottleneck_residual_block(64, 64, 256, 1, "layer1_block1");
  auto layer1_block2 = make_bottleneck_residual_block(256, 64, 256, 1, "layer1_block2");
  auto layer1_block3 = make_bottleneck_residual_block(256, 64, 256, 1, "layer1_block3");

  auto layer2_block1 = make_bottleneck_residual_block(256, 128, 512, 2, "layer2_block1");
  auto layer2_block2 = make_bottleneck_residual_block(512, 128, 512, 1, "layer2_block2");
  auto layer2_block3 = make_bottleneck_residual_block(512, 128, 512, 1, "layer2_block3");
  auto layer2_block4 = make_bottleneck_residual_block(512, 128, 512, 1, "layer2_block4");

  auto layer3_block1 = make_bottleneck_residual_block(512, 256, 1024, 2, "layer3_block1");
  auto layer3_block2 = make_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block2");
  auto layer3_block3 = make_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block3");
  auto layer3_block4 = make_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block4");
  auto layer3_block5 = make_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block5");
  auto layer3_block6 = make_bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block6");

  auto layer4_block1 = make_bottleneck_residual_block(1024, 512, 2048, 2, "layer4_block1");
  auto layer4_block2 = make_bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block2");
  auto layer4_block3 = make_bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block3");

  auto avgpool = AvgPool2DLayer(7, 7, 1, 1, 0, 0, "avgpool");
  auto flatten = FlattenLayer(1, -1, "flatten");
  auto fc = DenseLayer(2048, 1000, true, "fc");

  auto x = conv1(input);
  x = bn1(x);
  x = maxpool(x);

  x = layer1_block1(x);
  x = layer1_block2(x);
  x = layer1_block3(x);

  x = layer2_block1(x);
  x = layer2_block2(x);
  x = layer2_block3(x);
  x = layer2_block4(x);

  x = layer3_block1(x);
  x = layer3_block2(x);
  x = layer3_block3(x);
  x = layer3_block4(x);
  x = layer3_block5(x);
  x = layer3_block6(x);

  x = layer4_block1(x);
  x = layer4_block2(x);
  x = layer4_block3(x);

  x = avgpool(x);
  x = flatten(x);
  auto output = fc(x);
  output->set_uid("output");

  graph.compile(allocator);
  return graph;
}

}  // namespace

void ExampleGraphs::register_defaults() {
  register_graph("dual_input_add", create_add_graph);
  register_graph("mnist_cnn", create_mnist_graph);
  register_graph("imagenet_resnet50", create_resnet50_graph);
}

}  // namespace tnn