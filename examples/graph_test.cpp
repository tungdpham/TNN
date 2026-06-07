#include "nn/graph.hpp"

#include <getopt.h>

#include "data_loading/data_loader_factory.hpp"
#include "device/device_type.hpp"
#include "device/pool_allocator.hpp"
#include "nn/example_models.hpp"
#include "nn/metrics.hpp"
#include "nn/train.hpp"

using namespace std;
using namespace tnn;

signed main(int argc, char* argv[]) {
  ExampleModels::register_defaults();

  std::string config_path;
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'}, {"help", no_argument, 0, 'h'}, {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "c:h", long_options, nullptr)) != -1) {
    switch (opt) {
      case 'c':
        config_path = optarg;
        break;
      case 'h':
        cout << "Usage: " << argv[0] << " [options]" << endl;
        cout << "Options:" << endl;
        cout << "  --config <path>    Path to the JSON configuration file" << endl;
        cout << "  -h, --help         Show this help message" << endl;
        return 0;
      default:
        return 1;
    }
  }

  TrainingConfig train_config;
  if (config_path.empty()) {
    cerr << "Error: Configuration file path is required. Use --config <path> to specify it."
         << endl;
    return 1;
  }
  train_config.load_from_json(config_path);
  train_config.print_config();

  const Device& device = train_config.device_type == DeviceType::GPU ? getGPU() : getHost();
  auto& allocator = PoolAllocator::instance(device, defaultFlowHandle);

  Graph graph;
  auto input_node = graph.make_node("input");
  Sequential model = ExampleModels::create(train_config.model_name);
  auto output_node = model(input_node);
  output_node->set_uid("output");
  graph.compile(allocator);

  auto [train_loader, val_loader] =
      DataLoaderFactory::create(train_config.dataset_name, train_config.dataset_path);
  train_loader->set_seed(123456);

  Tensor input, label;
  auto criterion = LossFactory::create_crossentropy();
  auto optimizer =
      OptimizerFactory::create_adam(train_config.lr_initial, 0.9f, 0.999f, 10e-4f, 3e-4f, false);

  optimizer->attach(graph);

  while (train_loader->get_batch(256, input, label)) {
    Tensor device_input = input->to_device(graph.context()->device());
    TensorBundle inputs{{"input", device_input}};
    TensorBundle outputs = graph.forward(inputs);
    Tensor device_output = outputs.get("output");

    float loss;
    Tensor device_labels = label->to_device(graph.context()->device());
    criterion->compute_loss(device_output, device_labels, loss);
    int class_corrects = compute_class_corrects(device_output, device_labels);
    std::cout << "Loss: " << loss << ", Accuracy: "
              << (static_cast<float>(class_corrects) / device_output->dimension(0)) * 100.0f << "%"
              << std::endl;
    Tensor grad_output = create_like(device_output);
    criterion->compute_gradient(device_output, device_labels, grad_output);

    TensorBundle grad_outputs{{"output", grad_output}};
    graph.backward(grad_outputs);

    optimizer->update();
    optimizer->zero_grads();
  }

  return 0;
}