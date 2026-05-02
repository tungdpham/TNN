#include <getopt.h>

#include <memory>

#include "data_loading/data_loader_factory.hpp"
#include "device/device_manager.hpp"
#include "nn/example_models.hpp"
#include "nn/graph.hpp"
#include "nn/graph_builder.hpp"
#include "nn/schedulers.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace std;
using namespace tnn;

signed main(int argc, char *argv[]) {
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

  if (!config_path.empty()) {
    train_config.load_from_json(config_path);
  } else {
    train_config.load_from_env();
  }
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  const auto &device = DeviceManager::getInstance().getDevice(train_config.device_type);
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);
  GraphBuilder builder;

  if (train_config.dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  auto [train_loader, val_loader] =
      DataLoaderFactory::create(train_config.dataset_name, train_config.dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for dataset: " << train_config.dataset_name << endl;
    return 1;
  }
  train_loader->set_seed(123456);

  Graph graph = load_or_create_model(train_config.model_name, train_config.model_path, allocator);

  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer =
      OptimizerFactory::create_adam(train_config.lr_initial, 0.9f, 0.999f, 10e-4f, 3e-4f, false);
  int step_lr_epochs = 5;
  float step_lr_gamma = 0.1f;
  int step_lr_steps = 0;
  Env::get("STEP_LR_EPOCHS", step_lr_epochs);
  Env::get("STEP_LR_GAMMA", step_lr_gamma);
  Env::get("STEP_LR_STEPS", step_lr_steps);

  size_t step_size = step_lr_steps > 0 ? static_cast<size_t>(step_lr_steps)
                                       : static_cast<size_t>(step_lr_epochs) *
                                             train_loader->size() / train_config.batch_size;

  auto scheduler = SchedulerFactory::create_step_lr(optimizer.get(), step_size, step_lr_gamma);

  try {
    train_model(graph, train_loader, val_loader, optimizer, criterion, scheduler, train_config);
  } catch (const std::exception &e) {
    cerr << "Training failed: " << e.what() << endl;
    return 1;
  }

  return 0;
}