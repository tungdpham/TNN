#include "device/device_manager.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

signed main() {
  Conv2DLayer conv_layer(3, 128, 3, 3, 1, 1, 1, 1, true, "conv2d_test");
  conv_layer.set_device(getGPU());
  conv_layer.init();

  Tensor input = make_tensor<float>({128, 224, 224, 3}, &getGPU());
  input->fill_random_normal(0.5f, 0.2f, 676767);
  Tensor output = make_tensor<float>({128, 224, 224, 128}, &getGPU());

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
    std::cout << "Pass " << i + 1 << " took " << pass_duration.count() << " ms" << std::endl;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Average time per forward pass: " << duration.count() / passes << " ms" << std::endl;

  return 0;
}
