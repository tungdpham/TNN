#include "distributed/roce_coordinator.hpp"

#include <getopt.h>

#include <iostream>
#include <vector>

#include "data_loading/data_loader_factory.hpp"
#include "distributed/coordinator.hpp"
#include "distributed/roce_worker.hpp"
#include "distributed/train.hpp"
#include "nn/example_models.hpp"
#include "nn/optimizers.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

struct Config {
  std::string device_name = "";
  int gid_index = -1;
};

void print_usage(const char *program_name) {
  cout << "Usage: " << program_name << " [options] <worker_host:port>..." << endl;
  cout << endl;
  cout << "Options:" << endl;
  cout << "  --host <address>       Hostname or IP to bind to (required)" << endl;
  cout << "  --port <number>        TCP port for initial connection (required)" << endl;
  cout << "  --device <name>        IB device name (e.g., mlx5_0) (required)" << endl;
  cout << "  --gid-index <index>    GID index for RoCE (required)" << endl;
  cout << "  -h, --help             Show this help message" << endl;
}

bool parse_arguments(int argc, char *argv[], Config &cfg) {
  int c;

  static struct option long_options[] = {{"device", required_argument, 0, 'd'},
                                         {"gid-index", required_argument, 0, 'g'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  while ((c = getopt_long(argc, argv, "H:p:d:g:h", long_options, nullptr)) != -1) {
    switch (c) {
      case 'd':
        cfg.device_name = optarg;
        break;
      case 'g':
        try {
          cfg.gid_index = stoi(optarg);
        } catch (...) {
          cerr << "Invalid gid-index value: " << optarg << endl;
          return false;
        }
        break;
      case 'h':
        print_usage(argv[0]);
        return false;
      case '?':
        return false;
      default:
        return false;
    }
  }

  if (cfg.device_name.empty()) {
    cerr << "Missing required argument: --device" << endl;
    print_usage(argv[0]);
    return false;
  }
  if (cfg.gid_index < 0) {
    cout << "Since gid-index is not specified, auto-selecting GID index." << endl;
  }

  return true;
}

int main(int argc, char *argv[]) {
  Config cfg;

  if (!parse_arguments(argc, argv, cfg)) {
    return 1;
  }

  ExampleModels::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  DeviceType device_type = train_config.device_type;
  const auto &device = DeviceManager::getInstance().getDevice(device_type);
  auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);

  Graph graph = load_or_create_model(train_config.model_name, train_config.model_path, allocator);

  if (train_config.dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  auto [train_loader, val_loader] =
      DataLoaderFactory::create(train_config.dataset_name, train_config.dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for model: " << train_config.model_name << endl;
    return 1;
  }

  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer =
      OptimizerFactory::create_adam(train_config.lr_initial, 0.9f, 0.999f, 1e-5f, 1e-4f, false);
  auto scheduler = SchedulerFactory::create_step_lr(
      optimizer.get(), 5 * train_loader->size() / train_config.batch_size, 0.6f);
  std::string host = Env::get<std::string>("COORDINATOR_HOST", "localhost");
  int port = Env::get<int>("COORDINATOR_PORT", 9000);

  Endpoint coordinator_endpoint = Endpoint::roce(host, port, cfg.device_name, cfg.gid_index);
  Endpoint local_worker_endpoint =
      Endpoint::roce(Env::get<std::string>("LOCAL_WORKER_HOST", "localhost"),
                     Env::get<int>("LOCAL_WORKER_PORT", 8000), cfg.device_name, cfg.gid_index);
  int worker_position = Env::get<std::string>("LOCAL_WORKER_POSITION", "last") == "first" ? 0 : 1;

  Vec<Endpoint> endpoints = {
      Endpoint::roce(Env::get<std::string>("WORKER1_HOST", "10.10.0.2"),
                     Env::get<int>("WORKER1_PORT", 8001), "rocep131s0f0", -1),
  };

  if (worker_position) {
    endpoints.push_back(local_worker_endpoint);
  } else {
    endpoints.insert(endpoints.begin(), local_worker_endpoint);
  }

  auto local_worker =
      std::make_unique<RoCEWorker>(local_worker_endpoint, device_type == DeviceType::GPU);

  // initialize a partitioner with weights 2:1
  auto partitioner = std::make_unique<NaivePipelinePartitioner>(NaivePartitionerConfig({2, 1}));

  CoordinatorConfig config{
      ParallelMode_t::PIPELINE, std::move(graph),        std::move(optimizer), std::move(scheduler),
      std::move(partitioner),   std::move(local_worker), coordinator_endpoint, endpoints};

  RoCECoordinator coordinator(std::move(config));

  coordinator.initialize();

  std::cout << "Deploying stages to remote endpoints." << std::endl;
  for (const auto &ep : endpoints) {
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
