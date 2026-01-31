#include "device/device_manager.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "nn/layers_impl/legacy_conv2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

signed main() {
  Conv2DLayer conv_layer(3, 128, 3, 3, 1, 1, 1, 1, true, "conv2d_test");
  conv_layer.set_device(getGPU());
  conv_layer.init();

  LegacyConv2DLayer legacy_conv_layer(3, 128, 3, 3, 1, 1, 1, 1, true, "legacy_conv2d_test");
  legacy_conv_layer.set_device(getGPU());
  legacy_conv_layer.init();

  Tensor input = Tensor::create<float>({128, 224, 224, 3}, getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor output = Tensor::create<float>({128, 224, 224, 128}, getGPU());

  // cold pass
  conv_layer.forward(input, output);

  int passes = 10;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    conv_layer.forward(input, output);
    Flow *flow = getGPU().getFlow("default");
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

  Tensor nchw_input = Tensor::create<float>({128, 3, 224, 224}, getGPU());
  nchw_input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor nchw_output = Tensor::create<float>({128, 128, 224, 224}, getGPU());
  // legacy conv2d benchmark
  // cold pass
  legacy_conv_layer.forward(nchw_input, nchw_output);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < passes; ++i) {
    auto pass_start = std::chrono::high_resolution_clock::now();
    legacy_conv_layer.forward(nchw_input, nchw_output);
    Flow *flow = getGPU().getFlow("default");
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
