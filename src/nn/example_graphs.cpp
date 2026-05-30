/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/example_graphs.hpp"

#include "nn/layers.hpp"
#include "nn/layers_impl/relu_layer.hpp"
#include "nn/node_ops.hpp"

namespace tnn {

using namespace tnn::graph_api_v2;

std::unordered_map<std::string, std::function<Graph(IAllocator &)>> ExampleGraphs::creators_;

static Node bottleneck_residual_block(Node input, size_t in_channels, size_t mid_channels,
                                      size_t out_channels, size_t stride = 1,
                                      const std::string &name = "bottleneck_residual_block") {
  Node main_output = input;
  // main seq
  {
    auto conv1 = Conv2DLayer(in_channels, mid_channels, 1, 1, 1, 1, 0, 0, false, name + "_conv1");
    auto bn1 =
        BatchNormLayer(mid_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn1");
    auto conv2 =
        Conv2DLayer(mid_channels, mid_channels, 3, 3, stride, stride, 1, 1, false, name + "_conv2");
    auto bn2 =
        BatchNormLayer(mid_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn2");
    auto conv3 = Conv2DLayer(mid_channels, out_channels, 1, 1, 1, 1, 0, 0, false, name + "_conv3");
    auto bn3 =
        BatchNormLayer(out_channels, dtype_eps(DType_t::FP32), 0.1f, true, true, name + "_bn3");
    main_output = conv1(input);
    main_output = bn1(main_output);
    main_output = conv2(main_output);
    main_output = bn2(main_output);
    main_output = conv3(main_output);
    main_output = bn3(main_output);
  }

  Node shortcut_output = input;
  // shortcut seq
  if (stride != 1 || in_channels != out_channels) {
    auto shortcut_conv = Conv2DLayer(in_channels, out_channels, 1, 1, stride, stride, 0, 0, false,
                                     name + "_shortcut_conv");
    auto shortcut_bn = BatchNormLayer(out_channels, dtype_eps(DType_t::FP32), 0.1f, true, true,
                                      name + "_shortcut_bn");
    shortcut_output = shortcut_conv(input);
    shortcut_output = shortcut_bn(shortcut_output);
  }

  auto output = main_output + shortcut_output;
  auto relu = ReLULayer(name + "_relu");
  output = relu(output);
  return output;
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
  auto input = graph.input("input");

  auto conv1 = Conv2DLayer(3, 64, 7, 7, 2, 2, 3, 3, true, "conv1");
  auto bn1 = BatchNormLayer(64, dtype_eps(DType_t::FP32), 0.1f, true, true, "bn1");
  auto maxpool = MaxPool2DLayer(3, 3, 2, 2, 1, 1, "maxpool");

  auto avgpool = AvgPool2DLayer(7, 7, 1, 1, 0, 0, "avgpool");
  auto flatten = FlattenLayer(1, -1, "flatten");
  auto fc = DenseLayer(2048, 1000, true, "fc");

  auto x = conv1(input);
  x = bn1(x);
  x = maxpool(x);

  x = bottleneck_residual_block(x, 64, 64, 256, 1, "layer1_block1");
  x = bottleneck_residual_block(x, 256, 64, 256, 1, "layer1_block2");
  x = bottleneck_residual_block(x, 256, 64, 256, 1, "layer1_block3");

  x = bottleneck_residual_block(x, 256, 128, 512, 2, "layer2_block1");
  x = bottleneck_residual_block(x, 512, 128, 512, 1, "layer2_block2");
  x = bottleneck_residual_block(x, 512, 128, 512, 1, "layer2_block3");
  x = bottleneck_residual_block(x, 512, 128, 512, 1, "layer2_block4");

  x = bottleneck_residual_block(x, 512, 256, 1024, 2, "layer3_block1");
  x = bottleneck_residual_block(x, 1024, 256, 1024, 1, "layer3_block2");
  x = bottleneck_residual_block(x, 1024, 256, 1024, 1, "layer3_block3");
  x = bottleneck_residual_block(x, 1024, 256, 1024, 1, "layer3_block4");
  x = bottleneck_residual_block(x, 1024, 256, 1024, 1, "layer3_block5");
  x = bottleneck_residual_block(x, 1024, 256, 1024, 1, "layer3_block6");

  x = bottleneck_residual_block(x, 1024, 512, 2048, 2, "layer4_block1");
  x = bottleneck_residual_block(x, 2048, 512, 2048, 1, "layer4_block2");
  x = bottleneck_residual_block(x, 2048, 512, 2048, 1, "layer4_block3");

  x = avgpool(x);
  x = flatten(x);
  auto output = fc(x);
  output->set_uid("output");

  graph.compile(allocator);
  return graph;
}

void ExampleGraphs::register_defaults() {
  register_graph("mnist_cnn", create_mnist_graph);
  register_graph("resnet50_imagenet100", create_resnet50_graph);
}

}  // namespace tnn