#include "data_loading/data_loader_factory.hpp"
#include "device/device_manager.hpp"
#include "nn/example_models.hpp"
#include "nn/graph_builder.hpp"
#include "nn/loss.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace std;
using namespace tnn;

signed main() {
  ExampleModels::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  const auto &device = train_config.device_type == DeviceType::GPU ? getGPU(0) : getHost();
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);
  GraphBuilder builder;

  string dataset_name = "";
  Env::get("DATASET_NAME", dataset_name);
  if (dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  string dataset_path = "data";
  Env::get("DATASET_PATH", dataset_path);
  auto [train_loader, val_loader] = DataLoaderFactory::create(dataset_name, dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for model: " << train_config.model_name << endl;
    return 1;
  }

  Graph graph = load_or_create_model(train_config.model_name, train_config.model_path, allocator);

  auto criterion = LossFactory::create_logsoftmax_crossentropy();

  try {
    auto res = validate_model(graph, val_loader, criterion, train_config);
    std::cout << "Validation Loss: " << res.avg_loss << ", Accuracy: " << res.avg_accuracy * 100.0
              << "%" << std::endl;
  } catch (const std::exception &e) {
    cerr << "Inference failed: " << e.what() << endl;
    return 1;
  }

  return 0;
}