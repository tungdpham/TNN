#include <memory>

#include "data_loading/legacy/data_loader_factory.hpp"
#include "device/device_manager.hpp"
#include "nn/graph_builder.hpp"
#include "nn/legacy/example_models.hpp"
#include "nn/schedulers.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace std;
using namespace tnn;
using namespace tnn::legacy;

signed main() {
  legacy::ExampleModels::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  std::string model_name = Env::get<std::string>("MODEL_NAME", "cifar10_resnet9");
  std::string model_path = Env::get<std::string>("MODEL_PATH", "");

  std::string device_str = Env::get<std::string>("DEVICE_TYPE", "CPU");
  DeviceType device_type = (device_str == "GPU") ? DeviceType::GPU : DeviceType::CPU;
  const auto &device = DeviceManager::getInstance().getDevice(device_type);
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);
  GraphBuilder builder;

  string dataset_name = Env::get<std::string>("DATASET_NAME", "");
  if (dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  string dataset_path = Env::get<std::string>("DATASET_PATH", "data");
  auto [train_loader, val_loader] = legacy::DataLoaderFactory::create(dataset_name, dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for model: " << model_name << endl;
    return 1;
  }

  Graph graph = legacy::load_or_create_model(model_name, model_path, allocator);

  cout << "Training model on device: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;

  float lr_initial = Env::get("LR_INITIAL", 0.001f);
  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer = OptimizerFactory::create_adam(lr_initial, 0.9f, 0.999f, 1e-5f, 1e-4f, false);
  auto scheduler = SchedulerFactory::create_step_lr(
      optimizer.get(), 5 * train_loader->size() / train_config.batch_size, 0.1f);

  try {
    train_model(graph, train_loader, val_loader, optimizer, criterion, scheduler, train_config);
  } catch (const std::exception &e) {
    cerr << "Training failed: " << e.what() << endl;
    return 1;
  }

  return 0;
}