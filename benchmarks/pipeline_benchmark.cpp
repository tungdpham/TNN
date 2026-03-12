#include "device/device_manager.hpp"
#include "nn/graph.hpp"
#include "nn/graph_builder.hpp"
#include "nn/layers_impl/batchnorm_layer.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "nn/layers_impl/maxpool2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

signed main() {
  auto& allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  GraphBuilder builder;
  auto conv_layer = make_unique<Conv2DLayer>(3, 64, 3, 3, 1, 1, 1, 1, true, "conv2d_test");
  auto& conv_node = builder.add_layer(std::move(conv_layer));

  auto bn_layer = make_unique<BatchNormLayer>(64, 1e-5f, 0.1, true, true, "batchnorm_test");
  auto& bn_node = builder.add_layer(std::move(bn_layer));

  auto maxpool_layer = make_unique<MaxPool2DLayer>(2, 2, 2, 2, 0, 0, "maxpool_test");
  auto& maxpool_node = builder.add_layer(std::move(maxpool_layer));

  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({128, 224, 224, 3}, getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor conv2d_output = make_tensor<float>({128, 224, 224, 64}, getGPU());
  Tensor batchnorm_output = make_tensor<float>({128, 224, 224, 64}, getGPU());
  Tensor maxpool_output = make_tensor<float>({128, 112, 112, 64}, getGPU());
  // cold pass
  conv_node.forward({input}, {conv2d_output});
  bn_node.forward({conv2d_output}, {batchnorm_output});
  maxpool_node.forward({batchnorm_output}, {maxpool_output});
  Flow* flow = getGPU().getFlow(defaultFlowHandle);
  flow->synchronize();

  // warm pass
  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    conv_node.forward({input}, {conv2d_output});
    bn_node.forward({conv2d_output}, {batchnorm_output});
    maxpool_node.forward({batchnorm_output}, {maxpool_output});
    Flow* flow = getGPU().getFlow(defaultFlowHandle);
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
