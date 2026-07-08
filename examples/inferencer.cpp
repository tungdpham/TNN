#include <getopt.h>

#include "data_loading/data_loader_factory.hpp"
#include "device/device_manager.hpp"
#include "nn/example_graphs.hpp"
#include "nn/loss.hpp"
#include "nn/train.hpp"

using namespace std;
using namespace tunx;

signed main(int argc, char *argv[]) {
  ExampleGraphs::register_defaults();

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

  if (config_path.empty()) {
    cerr << "Error: Configuration file path is required. Use --config <path> to specify it."
         << endl;
    return 1;
  }

  TrainingConfig train_config;
  train_config.load_from_json(config_path);
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  const auto &device = train_config.device_type == DeviceType::CUDA ? getGPU(0) : getHost();
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  auto [train_loader, val_loader] =
      DataLoaderFactory::create(train_config.dataset_name, train_config.dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for model: " << train_config.model_name << endl;
    return 1;
  }

  Graph graph = load_or_create_graph(train_config.model_name, train_config.model_path, allocator);

  auto criterion = LossFactory::create_crossentropy();

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