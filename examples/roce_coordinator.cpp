#include "distributed/roce_coordinator.hpp"

#include <getopt.h>

#include <iostream>

#include "data_loading/data_loader_factory.hpp"
#include "distributed/roce_worker.hpp"
#include "distributed/train.hpp"
#include "nn/example_graphs.hpp"
#include "nn/optimizers.hpp"
#include "partitioner/graph_partitioner.hpp"

using namespace tunx;
using namespace std;

void print_usage(const char *program_name) {
  cout << "Usage: " << program_name << " [options] <worker_host:port>..." << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  --config <path>        Path to JSON training configuration file (required)" << endl;
  cout << "  -h, --help             Show this help message" << endl;
}

int main(int argc, char *argv[]) {
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

  ExampleGraphs::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_json(config_path);
  train_config.print_config();

  RoCEConfig roce_config;
  roce_config.load_from_json(config_path);

  // Prioritize loading existing model, else create from available ones
  DeviceType device_type = train_config.device_type;
  const auto &device = DeviceManager::getInstance().getDevice(device_type);
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  Graph graph = load_or_create_graph(train_config.model_name, train_config.model_path, allocator);

  if (train_config.dataset_name.empty()) {
    throw std::runtime_error("dataset_name variable is not set!");
  }
  auto [train_loader, val_loader] =
      DataLoaderFactory::create(train_config.dataset_name, train_config.dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for model: " << train_config.model_name << endl;
    return 1;
  }

  auto optimizer = OptimizerFactory::create_from_config(train_config.optimizer_config);

  auto scheduler =
      SchedulerFactory::create_from_config(train_config.scheduler_config, optimizer.get());

  auto criterion = LossFactory::create_from_config(train_config.loss_config);

  Endpoint coordinator_endpoint = Endpoint::roce(roce_config.host, roce_config.port,
                                                 roce_config.device_name, roce_config.gid_index);

  Endpoint local_worker_endpoint = Endpoint::roce(roce_config.host, roce_config.port,
                                                  roce_config.device_name, roce_config.gid_index);
  auto local_worker =
      std::make_unique<RoCEWorker>(local_worker_endpoint, device_type == DeviceType::CUDA);
  roce_config.worker_endpoints.push_back(local_worker_endpoint);

  auto partitioner = std::make_unique<GraphPartitioner>(roce_config.partition_ratios);

  CoordinatorConfig coordinator_config{
      .model = std::move(graph),
      .optimizer = std::move(optimizer),
      .scheduler = std::move(scheduler),
      .partitioner = std::move(partitioner),
      .local_worker = std::move(local_worker),
      .worker_endpoints = std::move(roce_config.worker_endpoints),
  };

  RoCECoordinator coordinator(std::move(roce_config), std::move(coordinator_config));

  coordinator.initialize();

  std::cout << "Deploying stages to remote endpoints." << std::endl;
  for (const auto &ep : roce_config.worker_endpoints) {
    std::cout << "  Worker expected at " << ep.to_json().dump(4) << std::endl;
  }

  if (!coordinator.deploy_stages()) {
    std::cerr << "Failed to deploy stages. Make sure workers are running." << std::endl;
    return 1;
  }

  try {
    train_model(coordinator, train_loader, val_loader, criterion, train_config);
    std::cout << "Coordinator initialized successfully." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
