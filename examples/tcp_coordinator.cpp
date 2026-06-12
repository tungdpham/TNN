/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "distributed/tcp_coordinator.hpp"

#include <getopt.h>

#include <cstdlib>
#include <iostream>
#include <memory>

#include "data_loading/data_loader_factory.hpp"
#include "device/pool_allocator.hpp"
#include "distributed/coordinator.hpp"
#include "distributed/endpoint.hpp"
#include "distributed/tcp_worker.hpp"
#include "distributed/train.hpp"
#include "nn/example_graphs.hpp"
#include "partitioner/graph_partitioner.hpp"

using namespace synet;
using namespace std;

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

  TCPConfig tcp_config;
  tcp_config.load_from_json(config_path);

  const auto &device = DeviceManager::getInstance().getDevice(train_config.device_type);
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
  train_loader->set_seed(123456);

  auto criterion = LossFactory::create_from_config(train_config.loss_config);

  auto optimizer = OptimizerFactory::create_from_config(train_config.optimizer_config);

  auto scheduler =
      SchedulerFactory::create_from_config(train_config.scheduler_config, optimizer.get());

  cout << "Configured " << tcp_config.worker_endpoints.size() << " remote endpoints:" << endl;
  for (const auto &ep : tcp_config.worker_endpoints) {
    cout << ep.to_json().dump(4) << endl;
  }

  Endpoint coordinator_endpoint = Endpoint::tcp(tcp_config.host, tcp_config.port);
  Endpoint local_worker_endpoint =
      Endpoint::tcp(tcp_config.local_worker_host, tcp_config.local_worker_port);

  tcp_config.worker_endpoints.push_back(local_worker_endpoint);

  cout << "Local worker endpoint: " << local_worker_endpoint.to_json().dump(4) << endl;

  // hard-coded for now
  auto worker = std::make_unique<TCPWorker>(local_worker_endpoint,
                                            train_config.device_type == DeviceType::GPU);

  auto partitioner = make_unique<GraphPartitioner>(tcp_config.partition_ratios);

  CoordinatorConfig coordinator_config{
      std::move(graph),
      std::move(optimizer),
      std::move(scheduler),
      std::move(partitioner),
      coordinator_endpoint,
      std::move(worker),
      tcp_config.worker_endpoints,
  };

  NetworkCoordinator coordinator(std::move(tcp_config), std::move(coordinator_config));

  coordinator.initialize();

  if (!coordinator.deploy_stages()) {
    cerr << "Failed to deploy stages. Make sure workers are running." << endl;
    return 1;
  }

  coordinator.start();
  train_model(coordinator, train_loader, val_loader, criterion, train_config);
  coordinator.stop();
  return 0;
}