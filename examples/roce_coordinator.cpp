#include "distributed/roce_coordinator.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "distributed/train.hpp"
#include "nn/example_models.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "partitioner/naive_partitioner.hpp"
#include <getopt.h>
#include <iostream>
#include <vector>

using namespace tnn;
using namespace std;

struct Config {
  std::string host = "localhost";
  int port = 0;
  std::string device_name = "";
  int gid_index = -1;
  std::vector<std::string> workers;
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

  static struct option long_options[] = {
      {"host", required_argument, 0, 'H'},   {"port", required_argument, 0, 'p'},
      {"device", required_argument, 0, 'd'}, {"gid-index", required_argument, 0, 'g'},
      {"help", no_argument, 0, 'h'},         {0, 0, 0, 0}};

  while ((c = getopt_long(argc, argv, "H:p:d:g:h", long_options, nullptr)) != -1) {
    switch (c) {
    case 'H':
      cfg.host = optarg;
      break;
    case 'p':
      try {
        cfg.port = stoi(optarg);
      } catch (...) {
        cerr << "Invalid port value: " << optarg << endl;
        return false;
      }
      break;
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

  if (cfg.host.empty()) {
    cerr << "Missing required argument: --host" << endl;
    print_usage(argv[0]);
    return false;
  }
  if (cfg.port <= 0 || cfg.port > 65535) {
    cerr << "Invalid or missing port: " << cfg.port << endl;
    print_usage(argv[0]);
    return false;
  }
  if (cfg.device_name.empty()) {
    cerr << "Missing required argument: --device" << endl;
    print_usage(argv[0]);
    return false;
  }
  if (cfg.gid_index < 0) {
    cerr << "Invalid or missing gid-index: " << cfg.gid_index << endl;
    print_usage(argv[0]);
    return false;
  }

  // Parse worker endpoints from remaining arguments
  for (int i = optind; i < argc; ++i) {
    cfg.workers.push_back(argv[i]);
  }

  if (cfg.workers.empty()) {
    cerr << "No workers specified." << endl;
    print_usage(argv[0]);
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  Config cfg;

  if (!parse_arguments(argc, argv, cfg)) {
    return 1;
  }

  std::vector<Endpoint> endpoints;
  for (const auto &worker_info : cfg.workers) {
    size_t colon_pos = worker_info.find(':');
    if (colon_pos == std::string::npos) {
      std::cerr << "Invalid worker info: " << worker_info << ". Expected host:port" << std::endl;
      return 1;
    }
    std::string worker_host = worker_info.substr(0, colon_pos);
    int worker_port = std::stoi(worker_info.substr(colon_pos + 1));

    endpoints.push_back(Endpoint::roce(worker_host, worker_port, "", 0));
  }

  CIFAR10DataLoader<float> train_loader, test_loader;

  create_cifar10_dataloader("./data", train_loader, test_loader);

  Sequential<float> model = create_resnet9_cifar10();

  auto optimizer = OptimizerFactory<float>::create_adam(0.001f, 0.9f, 0.999f, 1e-8f);

  RoceCoordinator coordinator("coordinator", std::move(model), std::move(optimizer), cfg.host,
                              cfg.port, cfg.device_name, cfg.gid_index, endpoints);

  coordinator.set_partitioner(std::make_unique<NaivePartitioner<float>>());
  coordinator.initialize();

  auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();
  coordinator.set_loss_function(std::move(loss_function));
  std::cout << "Deploying stages to remote endpoints." << std::endl;
  for (const auto &ep : endpoints) {
    std::cout << "  Worker expected at " << ep.to_json().dump(4) << std::endl;
  }

  if (!coordinator.deploy_stages()) {
    std::cerr << "Failed to deploy stages. Make sure workers are running." << std::endl;
    return 1;
  }

  try {
    coordinator.initialize();
    train_model(coordinator, train_loader, test_loader);
    std::cout << "Coordinator initialized successfully." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
