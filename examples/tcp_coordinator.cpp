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
#include <sstream>
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

  auto criterion = LossFactory::create_crossentropy();
  int adamw = 1;
  float adam_beta1 = 0.9f;
  float adam_beta2 = 0.95f;
  float adam_eps = 1e-8f;
  float weight_decay = 0.1f;
  Env::get("ADAMW", adamw);
  Env::get("ADAM_BETA1", adam_beta1);
  Env::get("ADAM_BETA2", adam_beta2);
  Env::get("ADAM_EPS", adam_eps);
  Env::get("ADAM_EPSILON", adam_eps);
  Env::get("WEIGHT_DECAY", weight_decay);

  auto optimizer = OptimizerFactory::create_adam(train_config.lr_initial, adam_beta1, adam_beta2,
                                                 adam_eps, weight_decay, adamw != 0);

  std::string lr_scheduler = "warmup_cosine";
  Env::get("SCHEDULER_TYPE", lr_scheduler);

  int step_lr_epochs = 5;
  float step_lr_gamma = 0.1f;
  int step_lr_steps = 0;
  Env::get("STEP_LR_EPOCHS", step_lr_epochs);
  Env::get("STEP_LR_GAMMA", step_lr_gamma);
  Env::get("STEP_LR_STEPS", step_lr_steps);

  size_t steps_per_epoch = train_loader->size() / train_config.batch_size;
  if (steps_per_epoch == 0) steps_per_epoch = 1;

  size_t total_steps = 0;
  if (train_config.max_steps > 0) {
    total_steps = static_cast<size_t>(train_config.max_steps);
  } else {
    total_steps = steps_per_epoch * static_cast<size_t>(train_config.epochs);
  }
  if (total_steps == 0) total_steps = 1;

  int warmup_steps = total_steps / 10;
  float cosine_start_lr = 0.0f;
  float cosine_eta_min = 0.0f;
  Env::get("WARMUP_STEPS", warmup_steps);
  Env::get("COSINE_START_LR", cosine_start_lr);
  Env::get("COSINE_ETA_MIN", cosine_eta_min);
  if (warmup_steps < 0) warmup_steps = 0;

  size_t step_size = step_lr_steps > 0 ? static_cast<size_t>(step_lr_steps)
                                       : static_cast<size_t>(step_lr_epochs) * steps_per_epoch;
  if (step_size == 0) step_size = 1;

  auto scheduler =
      (lr_scheduler == "warmup_cosine" || lr_scheduler == "cosine")
          ? SchedulerFactory::create_warmup_cosine(optimizer.get(),
                                                   static_cast<size_t>(warmup_steps), total_steps,
                                                   cosine_start_lr, cosine_eta_min)
          : SchedulerFactory::create_step_lr(optimizer.get(), step_size, step_lr_gamma);

  std::cout << fmt::format(
                   "Optimizer: {}, lr:{}, beta1:{}, beta2:{}, eps:{}, "
                   "weight_decay:{}, "
                   "scheduler:{}, warmup_steps:{}, total_steps:{}",
                   optimizer->name(), train_config.lr_initial, adam_beta1, adam_beta2, adam_eps,
                   weight_decay, scheduler->name(), warmup_steps, total_steps)
            << std::endl;

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

  // Parse partition split ratio from environment variable
  std::string split_ratio_str = "2,1";
  Env::get("PARTITION_SPLIT_RATIO", split_ratio_str);
  std::vector<size_t> split_ratios;
  std::stringstream ss(split_ratio_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    split_ratios.push_back(static_cast<size_t>(std::stoi(token)));
  }

  unique_ptr<Partitioner> partitioner =
      make_unique<NaivePipelinePartitioner>(NaivePartitionerConfig(split_ratios));

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