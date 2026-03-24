#include "nn/graph.hpp"

#include <getopt.h>

#include "data_loading/data_loader_factory.hpp"
#include "device/del_allocator_v2.hpp"
#include "device/device_manager.hpp"
#include "device/device_type.hpp"
#include "device/pool_allocator.hpp"
#include "nn/accuracy.hpp"
#include "nn/example_models.hpp"
#include "nn/graph_builder.hpp"
#include "nn/graph_executor.hpp"
#include "nn/io_node.hpp"
#include "nn/layers.hpp"
#include "nn/reducer.hpp"
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
  GraphBuilder builder;

  auto model_uptr = make_unique<Sequential>(ExampleModels::create(train_config.model_name));

  LayerFactory::register_defaults();
  vector<Layer*> layers = model_uptr->get_layers();
  IONode& input_node = builder.io("input");
  IONode* current_input = &input_node;
  for (size_t i = 0; i < layers.size(); ++i) {
    auto layer_config = layers[i]->get_config();
    std::unique_ptr<Layer> layer = LayerFactory::create(layer_config.type, layer_config);
    auto& op_node = builder.add_layer(std::move(layer));
    IONode& layer_output = builder.io("layer_output_" + std::to_string(i));
    builder.add_edge({current_input}, {&layer_output}, op_node);
    current_input = &layer_output;
  }
  auto& output_node = *current_input;

  Reducer reducer = ReducerBuilder().seq_reduction().build();
  reducer.reduce(builder);

  auto& allocator = PoolAllocator::instance(device, defaultFlowHandle);
  Graph graph = builder.compile(allocator);

  auto ws_allocator = DELAllocatorV2::instance(device, defaultFlowHandle);
  GraphExecutor executor(graph, ws_allocator);

  auto [train_loader, val_loader] =
      DataLoaderFactory::create(train_config.dataset_name, train_config.dataset_path);
  train_loader->set_seed(123456);

  Tensor input, label;
  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer =
      OptimizerFactory::create_adam(train_config.lr_initial, 0.9f, 0.999f, 10e-4f, 3e-4f, false);

  optimizer->attach(graph.context());

  while (train_loader->get_batch(256, input, label)) {
    Tensor device_input = input->to_device(graph.context().device());
    Tensor device_output;

    InputPack inputs = {{
        input_node.uid(),
        &device_input,
    }};
    OutputPack outputs = {{
        output_node.uid(),
        &device_output,
    }};

    executor.forward(inputs, outputs);

    float loss;
    Tensor device_labels = label->to_device(graph.context().device());
    criterion->compute_loss(device_output, device_labels, loss);
    int class_corrects = compute_class_corrects(device_output, device_labels);
    std::cout << "Loss: " << loss << ", Accuracy: "
              << (static_cast<float>(class_corrects) / device_output->dimension(0)) * 100.0f << "%"
              << std::endl;
    Tensor grad_output = create_like(device_output), grad_input = create_like(device_input);
    criterion->compute_gradient(device_output, device_labels, grad_output);

    InputPack grad_outputs = {{
        output_node.uid(),
        &grad_output,
    }};
    OutputPack grad_inputs = {{
        input_node.uid(),
        &grad_input,
    }};

    executor.backward(grad_outputs, grad_inputs);

    optimizer->update();
    optimizer->zero_grads();
  }

  return 0;
}