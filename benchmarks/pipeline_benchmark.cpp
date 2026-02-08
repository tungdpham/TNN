#include "device/device_manager.hpp"
#include "nn/activations_impl/base_activation.hpp"
#include "nn/activations_impl/relu.hpp"
#include "nn/layers_impl/activation_layer.hpp"
#include "nn/layers_impl/batchnorm_layer.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "nn/layers_impl/maxpool2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

signed main() {
  Conv2DLayer conv_layer(3, 64, 3, 3, 1, 1, 1, 1, true, "conv2d_test");
  BatchNormLayer batchnorm_layer(64, 1e-5f, 0.1, true, true, "batchnorm_test");
  MaxPool2DLayer maxpool_layer(2, 2, 2, 2, 0, 0, "maxpool_test");

  conv_layer.set_device(getGPU());
  conv_layer.set_training(true);
  conv_layer.init();

  batchnorm_layer.set_device(getGPU());
  batchnorm_layer.set_training(true);
  batchnorm_layer.init();

  maxpool_layer.set_device(getGPU());
  maxpool_layer.set_training(true);
  maxpool_layer.init();

  Tensor input = make_tensor<float>({128, 224, 224, 3}, getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor conv2d_output = make_tensor<float>({128, 224, 224, 64}, getGPU());
  Tensor batchnorm_output = make_tensor<float>({128, 224, 224, 64}, getGPU());
  Tensor maxpool_output = make_tensor<float>({128, 112, 112, 64}, getGPU());
  // cold pass
  conv_layer.forward(input, conv2d_output);
  batchnorm_layer.forward(conv2d_output, batchnorm_output);
  maxpool_layer.forward(batchnorm_output, maxpool_output);
  Flow *flow = getGPU().getFlow(defaultFlowHandle);
  flow->synchronize();

  // warm pass
  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    conv_layer.forward(input, conv2d_output);
    batchnorm_layer.forward(conv2d_output, batchnorm_output);
    maxpool_layer.forward(batchnorm_output, maxpool_output);
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
