#include "device/device_manager.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

signed main() {
  Conv2DLayer<float> conv_layer(3, 64, 3, 3, 1, 1, 1, 1, true, "conv2d_test");
  conv_layer.set_device(&getGPU());
  conv_layer.init();

  Tensor<float> output;
  for (int i = 0; i < 10; ++i) {
    Tensor<float> input({128, 3, 224, 224}, &getGPU());
    input.fill_random_normal(0.5f, 0.2f, 676767);

    conv_layer.forward(input, output);
  }

  return 0;
}
