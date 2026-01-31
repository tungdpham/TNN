#include <cstddef>
#include <memory>

#include "device/device_manager.hpp"
#include "nn/activations_impl/relu.hpp"
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
  // fuse relu
  BatchNormLayer batchnorm_layer(NUM_FEATURES, 1e-5f, 0.1f, true, true, "batchnorm_test");
  batchnorm_layer.set_device(getGPU());
  batchnorm_layer.init();

  LegacyBatchNormLayer legacy_batchnorm_layer(NUM_FEATURES, 1e-5f, 0.1f, true,
                                              "legacy_batchnorm_test");
  ActivationLayer relu_layer(std::make_unique<ReLU>(), "relu_activation");
  legacy_batchnorm_layer.set_device(getGPU());
  legacy_batchnorm_layer.init();
  relu_layer.set_device(getGPU());
  relu_layer.init();

  Tensor input = Tensor::create<float>({BATCH_SIZE, HEIGHT, WIDTH, NUM_FEATURES}, getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor output = Tensor::create<float>({BATCH_SIZE, HEIGHT, WIDTH, NUM_FEATURES}, getGPU());

  // cold pass
  batchnorm_layer.forward(input, output);

  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    batchnorm_layer.forward(input, output);
    Flow *flow = getGPU().getFlow("default");
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

  Tensor legacy_input = Tensor::create<float>({BATCH_SIZE, NUM_FEATURES, HEIGHT, WIDTH}, getGPU());
  legacy_input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor legacy_output = Tensor::create<float>({BATCH_SIZE, NUM_FEATURES, HEIGHT, WIDTH}, getGPU());
  Tensor legacy_relu_output =
      Tensor::create<float>({BATCH_SIZE, NUM_FEATURES, HEIGHT, WIDTH}, getGPU());

  // legacy batchnorm benchmark

  // cold pass
  legacy_batchnorm_layer.forward(legacy_input, legacy_output);
  relu_layer.forward(legacy_output, legacy_relu_output);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    legacy_batchnorm_layer.forward(legacy_input, legacy_output);
    relu_layer.forward(legacy_output, legacy_relu_output);
    Flow *flow = getGPU().getFlow("default");
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

  auto cpu_current_output = output->to_device(getCPU());
  auto cpu_legacy_output = legacy_relu_output->to_device(getCPU());

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
