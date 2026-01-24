#include "nn/example_models.hpp"
#include "nn/layers_impl/base_layer.hpp"
#include "nn/sequential.hpp"

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
  return 0;
}