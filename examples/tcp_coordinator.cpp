/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "distributed/tcp_coordinator.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "data_loading/data_loader_factory.hpp"
#include "device/pool_allocator.hpp"
#include "distributed/coordinator.hpp"
#include "distributed/endpoint.hpp"
#include "distributed/tcp_worker.hpp"
#include "distributed/train.hpp"
#include "nn/example_models.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

int main() {
  ExampleModels::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  const auto &device = DeviceManager::getInstance().getDevice(train_config.device_type);
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
  train_loader->set_seed(123456);

  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer =
      OptimizerFactory::create_adam(train_config.lr_initial, 0.9f, 0.999f, 10e-4f, 3e-4f, false);
  auto scheduler = SchedulerFactory::create_step_lr(
      optimizer.get(), 5 * train_loader->size() / train_config.batch_size, 0.6f);

  std::string coordinator_host = "localhost";
  int coordinator_port = 9000;
  Env::get("COORDINATOR_HOST", coordinator_host);
  Env::get("COORDINATOR_PORT", coordinator_port);
  Endpoint coordinator_endpoint = Endpoint::tcp(coordinator_host, coordinator_port);

  std::string local_worker_host = "localhost";
  int local_worker_port = 8000;
  Env::get("LOCAL_WORKER_HOST", local_worker_host);
  Env::get("LOCAL_WORKER_PORT", local_worker_port);
  Endpoint local_worker_endpoint = Endpoint::tcp(local_worker_host, local_worker_port);

  int local_worker_position = 0;  // default to first
  std::string position_str = "first";
  Env::get("LOCAL_WORKER_POSITION", position_str);
  if (position_str == "last") {
    local_worker_position = 1;
  }

  std::string worker1_host = "localhost";
  int worker1_port = 8001;
  Env::get("WORKER1_HOST", worker1_host);
  Env::get("WORKER1_PORT", worker1_port);
  vector<Endpoint> endpoints = {
      Endpoint::tcp(worker1_host, worker1_port),
  };

  if (local_worker_position) {
    endpoints.push_back(local_worker_endpoint);
  } else {
    endpoints.insert(endpoints.begin(), local_worker_endpoint);
  }

  cout << "Configured " << endpoints.size() << " remote endpoints:" << endl;
  for (const auto &ep : endpoints) {
    cout << ep.to_json().dump(4) << endl;
  }

  cout << "Local worker endpoint: " << local_worker_endpoint.to_json().dump(4) << endl;

  // hard-coded for now
  auto worker = std::make_unique<TCPWorker>(local_worker_endpoint,
                                            train_config.device_type == DeviceType::GPU);

  unique_ptr<Partitioner> partitioner =
      make_unique<NaivePipelinePartitioner>(NaivePartitionerConfig({2, 1}));

  CoordinatorConfig config{
      ParallelMode_t::PIPELINE, std::move(graph),  std::move(optimizer), std::move(scheduler),
      std::move(partitioner),   std::move(worker), coordinator_endpoint, endpoints,
  };

  NetworkCoordinator coordinator(std::move(config));

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