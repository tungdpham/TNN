#include <getopt.h>

#include <memory>

#include "data_loading/data_loader_factory.hpp"
#include "device/device_manager.hpp"
#include "nn/example_graphs.hpp"
#include "nn/graph.hpp"
#include "nn/schedulers.hpp"
#include "nn/train.hpp"

using namespace std;
using namespace synet;

signed main(int argc, char *argv[]) {
  ExampleGraphs::register_defaults();

  std::string config_path;
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0},
  };

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

  if (config_path.empty()) {
    cerr << "Error: Configuration file path is required. Use --config <path> to specify it."
         << endl;
    return 1;
  }

  TrainingConfig train_config;

  train_config.load_from_json(config_path);

  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  const auto &device = DeviceManager::getInstance().getDevice(train_config.device_type);
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  if (train_config.dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  auto [train_loader, val_loader] = DataLoaderFactory::create(
      train_config.dataset_name, train_config.dataset_path, train_config.io_dtype);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for dataset: " << train_config.dataset_name << endl;
    return 1;
  }
  train_loader->set_seed(123456);

  Graph graph = load_or_create_graph(train_config.model_name, train_config.model_path, allocator);

  auto criterion = LossFactory::create_from_config(train_config.loss_config);

  auto optimizer = OptimizerFactory::create_from_config(train_config.optimizer_config);

  auto scheduler =
      SchedulerFactory::create_from_config(train_config.scheduler_config, optimizer.get());

  try {
    train_model(graph, train_loader, val_loader, optimizer, criterion, scheduler, train_config);
  } catch (const std::exception &e) {
    cerr << "Training failed: " << e.what() << endl;
    return 1;
  }

  return 0;
}