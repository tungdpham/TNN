#include "device/device_manager.hpp"
#include "nn/graph.hpp"
#include "nn/graph_builder.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "nn/layers_impl/legacy_conv2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

signed main() {
  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  GraphBuilder builder;

  auto conv_layer = make_unique<Conv2DLayer>(16, 128, 3, 3, 1, 1, 0, 0, true, "conv2d_test");
  auto &conv_node = builder.add_layer(std::move(conv_layer));

  auto legacy_layer =
      make_unique<LegacyConv2DLayer>(16, 128, 3, 3, 1, 1, 0, 0, true, "legacy_conv2d_test");
  auto &legacy_node = builder.add_layer(std::move(legacy_layer));

  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({128, 224, 224, 16}, getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor output = make_tensor<float>({128, 224, 224, 128}, getGPU());

  // cold pass
  conv_node.forward({input}, {output});

  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    conv_node.forward({input}, {output});
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

  Tensor nchw_input = make_tensor<float>({128, 16, 224, 224}, getGPU());
  nchw_input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor nchw_output = make_tensor<float>({128, 128, 224, 224}, getGPU());
  // legacy conv2d benchmark
  // cold pass
  legacy_node.forward({nchw_input}, {nchw_output});
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    legacy_node.forward({nchw_input}, {nchw_output});
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
