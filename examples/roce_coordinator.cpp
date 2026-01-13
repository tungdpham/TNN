#include "distributed/roce_coordinator.hpp"
#include "data_loading/cifar100_data_loader.hpp"
#include "distributed/train.hpp"
#include "nn/example_models.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "utils/env.hpp"
#include <getopt.h>
#include <iostream>
#include <vector>

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

  ExampleModels<float>::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  std::string model_name = Env::get<std::string>("MODEL_NAME", "cifar10_resnet9");
  std::string model_path = Env::get<std::string>("MODEL_PATH", "");

  std::string device_str = Env::get<std::string>("DEVICE_TYPE", "CPU");
  DeviceType device_type = (device_str == "GPU") ? DeviceType::GPU : DeviceType::CPU;
  const auto &device = DeviceManager::getInstance().getDevice(device_type);

  Sequential<float> model;
  if (!model_path.empty()) {
    cout << "Loading model from: " << model_path << endl;
    model = Sequential<float>::from_file(model_path, &device); // automatically init
  } else {
    cout << "Creating model: " << model_name << endl;
    try {
      model = ExampleModels<float>::create(model_name);
    } catch (const std::exception &e) {
      cerr << "Error creating model: " << e.what() << endl;
      cout << "Available models are: ";
      for (const auto &name : ExampleModels<float>::available_models()) {
        cout << name << "\n";
      }
      cout << endl;
      return 1;
    }
    model.set_device(&device);
    model.init();
  }

  string dataset_name = Env::get<std::string>("DATASET_NAME", "");
  if (dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  string dataset_path = Env::get<std::string>("DATASET_PATH", "data");
  auto [train_loader, val_loader] = DataLoaderFactory<float>::create(dataset_name, dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for model: " << model_name << endl;
    return 1;
  }

  cout << "Training model on device: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;

  auto criterion = LossFactory<float>::create_logsoftmax_crossentropy();
  float lr_initial = Env::get<float>("LR_INITIAL", 0.001f);
  auto optimizer = OptimizerFactory<float>::create_adam(lr_initial, 0.9f, 0.999f, 1e-8f);

  std::vector<Endpoint> endpoints = {
      Endpoint::roce(Env::get<std::string>("WORKER1_HOST", "10.10.0.2"),
                     Env::get<int>("WORKER1_PORT", 8001), "rocep131s0f0", -1),
      Endpoint::roce(Env::get<std::string>("WORKER2_HOST", "10.10.0.1"),
                     Env::get<int>("WORKER2_PORT", 8002), "rocep5s0f0", -1),
  };

  std::string host = Env::get<std::string>("COORDINATOR_HOST", "localhost");
  int port = Env::get<int>("COORDINATOR_PORT", 9000);

  Endpoint coordinator_endpoint = Endpoint::roce(host, port, cfg.device_name, cfg.gid_index);
  RoceCoordinator coordinator("coordinator", std::move(model), std::move(optimizer),
                              coordinator_endpoint, endpoints);

  // initialize a partitioner with weights 2:1
  auto partitioner = std::make_unique<NaivePartitioner<float>>(NaivePartitionerConfig({2, 1}));

  coordinator.set_partitioner(std::move(partitioner));
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
    train_model(coordinator, *train_loader, *val_loader, criterion, train_config);
    std::cout << "Coordinator initialized successfully." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
