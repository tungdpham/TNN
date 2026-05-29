#include <cstddef>
#include <memory>

#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/activations_impl/relu.hpp"
#include "nn/graph_api.hpp"
#include "nn/layers.hpp"
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
  auto &device = getGPU();
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  Graph graph;
  auto input = graph.make_node("input");

  // fuse relu
  auto bn_layer = BatchNormLayer(NUM_FEATURES, 1e-5f, 0.1f, true, true, "batchnorm_test");
  auto bn_output_node = bn_layer(input);

  auto legacy_batchnorm_layer =
      LegacyBatchNormLayer(NUM_FEATURES, 1e-5f, 0.1f, true, "legacy_batchnorm_test");
  auto legacy_bn_output = legacy_batchnorm_layer(input);
  auto relu_layer = ActivationLayer(std::make_unique<ReLU>(), "relu_activation");
  auto legacy_relu_output_node = relu_layer(legacy_bn_output);

  graph.compile(allocator);

  Tensor input_data = make_tensor<float>({BATCH_SIZE, HEIGHT, WIDTH, NUM_FEATURES}, getGPU());
  input_data->fill_random_normal(0.5f, 0.2f, 676767);

  // cold pass
  Tensor output = bn_layer->forward({input_data})[0];
  Tensor legacy_output = legacy_batchnorm_layer->forward({input_data})[0];
  Tensor legacy_relu_output = relu_layer->forward({legacy_output})[0];

  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    output = bn_layer->forward({input_data})[0];
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

  // legacy batchnorm benchmark

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    legacy_output = legacy_batchnorm_layer->forward({input_data})[0];
    legacy_relu_output = relu_layer->forward({legacy_output})[0];
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
