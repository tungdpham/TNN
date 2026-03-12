#include <cstddef>
#include <memory>

#include "device/device_manager.hpp"
#include "nn/activations_impl/relu.hpp"
#include "nn/graph.hpp"
#include "nn/graph_builder.hpp"
#include "nn/layers_impl/activation_layer.hpp"
#include "nn/layers_impl/batchnorm_layer.hpp"
#include "nn/layers_impl/legacy_batchnorm_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

constexpr size_t BATCH_SIZE = 32;
constexpr size_t NUM_FEATURES = 512;
constexpr size_t HEIGHT = 128;
constexpr size_t WIDTH = 128;
constexpr float EPSILON = 2e-2f;

signed main() {
  auto &allocator = PoolAllocator::instance(getGPU(), defaultFlowHandle);
  GraphBuilder builder;

  // fuse relu
  auto bn_layer =
      make_unique<BatchNormLayer>(NUM_FEATURES, 1e-5f, 0.1f, true, true, "batchnorm_test");
  auto &bn_node = builder.add_layer(std::move(bn_layer));

  auto legacy_batchnorm_layer =
      make_unique<LegacyBatchNormLayer>(NUM_FEATURES, 1e-5f, 0.1f, true, "legacy_batchnorm_test");
  auto relu_layer = make_unique<ActivationLayer>(std::make_unique<ReLU>(), "relu_activation");
  auto &legacy_bn_node = builder.add_layer(std::move(legacy_batchnorm_layer));
  auto &relu_node = builder.add_layer(std::move(relu_layer));
  builder.output(bn_node, builder.input());
  Graph graph = builder.compile(allocator);

  Tensor input = make_tensor<float>({BATCH_SIZE, HEIGHT, WIDTH, NUM_FEATURES}, getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor output = make_tensor<float>({BATCH_SIZE, HEIGHT, WIDTH, NUM_FEATURES}, getGPU());

  // cold pass
  bn_node.forward({input}, {output});

  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    bn_node.forward({input}, {output});
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();

    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "BatchNorm: Pass " << i + 1 << " took " << pass_duration.count() << " ms"
              << std::endl;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "BatchNorm Average time per forward pass: " << duration.count() / passes << " ms"
            << std::endl;

  Tensor legacy_input = make_tensor<float>({BATCH_SIZE, NUM_FEATURES, HEIGHT, WIDTH}, getGPU());
  legacy_input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor legacy_output = make_tensor<float>({BATCH_SIZE, NUM_FEATURES, HEIGHT, WIDTH}, getGPU());
  Tensor legacy_relu_output =
      make_tensor<float>({BATCH_SIZE, NUM_FEATURES, HEIGHT, WIDTH}, getGPU());

  // legacy batchnorm benchmark

  // cold pass
  legacy_bn_node.forward({legacy_input}, {legacy_output});
  relu_node.forward({legacy_output}, {legacy_relu_output});

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    legacy_bn_node.forward({legacy_input}, {legacy_output});
    relu_node.forward({legacy_output}, {legacy_relu_output});
    Flow *flow = getGPU().getFlow(defaultFlowHandle);
    flow->synchronize();
    auto pass_end = std::chrono::high_resolution_clock::now();
    auto pass_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(pass_end - pass_start);
    std::cout << "Legacy BatchNorm: Pass " << i + 1 << " took " << pass_duration.count() << " ms"
              << std::endl;
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Legacy BatchNorm Average time per forward pass: " << duration.count() / passes
            << " ms" << std::endl;

  auto cpu_current_output = output->to_device(getHost());
  auto cpu_legacy_output = legacy_relu_output->to_device(getHost());

  float *current_data = cpu_current_output->data_as<float>();
  float *legacy_data = cpu_legacy_output->data_as<float>();
  float max_diff = 0.0f;
  for (size_t i = 0; i < cpu_current_output->size(); ++i) {
    float diff = std::abs(current_data[i] - legacy_data[i]);
    if (diff > EPSILON) {
      std::cout << "Mismatch at index " << i << ": current = " << current_data[i]
                << ", legacy = " << legacy_data[i] << ", diff = " << diff << std::endl;
    }
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  std::cout << "Max diff: " << max_diff << std::endl;
  return 0;
}
