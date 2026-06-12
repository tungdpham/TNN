#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/graph.hpp"
#include "nn/layers.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "nn/layers_impl/legacy_conv2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace synet;
using namespace std;

signed main() {
  auto &device = getGPU();
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  Graph graph;
  auto input = graph.make_node("input");

  auto conv_layer = Conv2DLayer(16, 128, 3, 3, 1, 1, 0, 0, true, "conv2d_test");
  auto conv_output = conv_layer(input);

  auto legacy_layer = LegacyConv2DLayer(16, 128, 3, 3, 1, 1, 0, 0, true, "legacy_conv2d_test");
  auto legacy_conv_output = legacy_layer(input);

  graph.compile(allocator);

  Tensor input_data = Tensor({128, 224, 224, 16}, DType_t::FP32, getGPU());
  input_data.fill_random_normal(0.5f, 0.2f, 676767);

  // cold pass
  Tensor output = conv_layer.forward({input_data})[0];

  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    output = conv_layer.forward({input_data})[0];
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();

    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Conv2D: Pass " << i + 1 << " took " << pass_duration.count() << " ms"
              << std::endl;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Conv2D Average time per forward pass: " << duration.count() / passes << " ms"
            << std::endl;

  Tensor nchw_input = Tensor({128, 16, 224, 224}, DType_t::FP32, getGPU());
  nchw_input.fill_random_normal(0.5f, 0.2f, 676767);
  // legacy conv2d benchmark
  // cold pass
  Tensor nchw_output = legacy_layer.forward({nchw_input})[0];
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    nchw_output = legacy_layer.forward({nchw_input})[0];
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();
    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Legacy Conv2D: Pass " << i + 1 << " took " << pass_duration.count() << " ms"
              << std::endl;
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Legacy Conv2D Average time per forward pass: " << duration.count() / passes << " ms"
            << std::endl;
  return 0;
}
