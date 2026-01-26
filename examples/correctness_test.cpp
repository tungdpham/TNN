#include "device/device_manager.hpp"
#include "nn/layers_impl/conv2d_layer.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;
using namespace std;

signed main() {
  Conv2DLayer conv_layer(3, 6, 3, 3, 1, 1, 1, 1, true, "conv2d_test");
  conv_layer.set_device(getGPU());
  conv_layer.init();

  auto params = conv_layer.parameters();
  for (auto param : params) {
    param->fill(1.0);
  }

  Tensor input = Tensor::create<float>({1, 4, 4, 3});
  float *input_data = input->data_as<float>();
  for (int i = 0; i < 48; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  Tensor device_input = input->to_device(&getGPU());
  Tensor output = Tensor::create<float>({1, 4, 4, 6}, &getGPU());
  conv_layer.forward(device_input, output);

  output->print_data();

  return 0;
}
