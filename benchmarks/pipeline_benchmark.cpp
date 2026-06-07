#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/graph.hpp"
#include "nn/layers.hpp"
#include "nn/layers_impl/batchnorm_layer.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "nn/layers_impl/maxpool2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

signed main() {
  auto &device = getGPU();
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  Graph graph;
  auto input = graph.make_node("input");

  auto conv_layer = Conv2DLayer(3, 64, 3, 3, 1, 1, 1, 1, true, "conv2d_test");
  auto conv_output_node = conv_layer(input);

  auto bn_layer = BatchNormLayer(64, 1e-5f, 0.1, true, true, "batchnorm_test");
  auto bn_output_node = bn_layer(conv_output_node);

  auto maxpool_layer = MaxPool2DLayer(2, 2, 2, 2, 0, 0, "maxpool_test");
  auto maxpool_output_node = maxpool_layer(bn_output_node);

  graph.compile(allocator);

  Tensor input_data = make_tensor<float>({128, 224, 224, 3}, getGPU());
  input_data->fill_random_normal(0.5f, 0.2f, 676767);
  // cold pass
  auto conv2d_output = conv_layer->forward({input_data})[0];
  auto batchnorm_output = bn_layer->forward({conv2d_output})[0];
  auto maxpool_output = maxpool_layer->forward({batchnorm_output})[0];
  Flow *flow = getGPU().getFlow(defaultFlowHandle);
  flow->synchronize();

  // warm pass
  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    conv2d_output = conv_layer->forward({input_data})[0];
    batchnorm_output = bn_layer->forward({conv2d_output})[0];
    maxpool_output = maxpool_layer->forward({batchnorm_output})[0];
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();

    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Pass " << i + 1 << " took " << pass_duration.count() << " ms" << std::endl;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Average time per forward pass: " << duration.count() / passes << " ms" << std::endl;

  return 0;
}
