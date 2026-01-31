#include "nn/sequential.hpp"

#include "device/device_manager.hpp"
#include "nn/example_models.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"

using namespace std;
using namespace tnn;

signed main() {
  ExampleModels::register_defaults();
  Sequential model = ExampleModels::create("cifar10_resnet9", DType_t::FP32);
  auto config = model.get_config();
  cout << config.to_json().dump(2) << endl;

  std::unique_ptr<Sequential> deserialized_model = Sequential::create_from_config(config);

  cout << "Deserialized model config:" << endl;
  auto deserialized_config = deserialized_model->get_config();
  cout << deserialized_config.to_json().dump(2) << endl;

  Tensor input = Tensor::create<float>({1, 32, 32, 3});
  Tensor output = Tensor::create<float>({1});
  model.set_device(getGPU());
  model.init();
  model.forward(input, output, 0);

  std::cout << "Output shape: " << output->shape_str() << std::endl;
  return 0;
}