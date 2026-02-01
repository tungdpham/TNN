/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "data_loading/legacy/data_loader_factory.hpp"
#include "distributed/coordinator.hpp"
#include "distributed/endpoint.hpp"
#include "distributed/tcp_coordinator.hpp"
#include "distributed/tcp_worker.hpp"
#include "distributed/train.hpp"
#include "nn/layers.hpp"
#include "nn/legacy/example_models.hpp"
#include "nn/sequential.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "threading/thread_wrapper.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace tnn::legacy;
using namespace std;

int main() {
  ExampleModels::register_defaults();

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  // Prioritize loading existing model, else create from available ones
  std::string model_name = Env::get<std::string>("MODEL_NAME", "cifar10_resnet9");
  std::string model_path = Env::get<std::string>("MODEL_PATH", "");

  std::string device_str = Env::get<std::string>("DEVICE_TYPE", "CPU");
  DeviceType device_type = (device_str == "GPU") ? DeviceType::GPU : DeviceType::CPU;
  const auto &device = DeviceManager::getInstance().getDevice(device_type);

  string dataset_name = Env::get<std::string>("DATASET_NAME", "");
  if (dataset_name.empty()) {
    throw std::runtime_error("DATASET_NAME environment variable is not set!");
  }
  string dataset_path = Env::get<std::string>("DATASET_PATH", "data");
  auto [train_loader, val_loader] = DataLoaderFactory::create(dataset_name, dataset_path);
  if (!train_loader || !val_loader) {
    cerr << "Failed to create data loaders for model: " << model_name << endl;
    return 1;
  }

  std::unique_ptr<Sequential> model;
  if (!model_path.empty()) {
    cout << "Loading model from: " << model_path << endl;
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open model file");
    }
    model = load_state<Sequential>(file, device);
    file.close();
  } else {
    cout << "Creating model: " << model_name << endl;
    try {
      Sequential temp_model = ExampleModels::create(model_name);
      model = std::make_unique<Sequential>(std::move(temp_model));
    } catch (const std::exception &e) {
      cerr << "Error creating model: " << e.what() << endl;
      cout << "Available models are: ";
      for (const auto &name : ExampleModels::available_models()) {
        cout << name << "\n";
      }
      cout << endl;
      return 1;
    }
    model->set_device(device);
    model->init();
  }

  cout << "Training model on device: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;

  auto criterion = LossFactory::create_logsoftmax_crossentropy();
  auto optimizer = OptimizerFactory::create_adam(train_config.lr_initial, 0.9f, 0.999f, 1e-8f);

  Endpoint coordinator_endpoint = Endpoint::tcp(Env::get<string>("COORDINATOR_HOST", "localhost"),
                                                Env::get<int>("COORDINATOR_PORT", 9000));

  Endpoint local_worker_endpoint =
      Endpoint::tcp(Env::get<std::string>("LOCAL_WORKER_HOST", "localhost"),
                    Env::get<int>("LOCAL_WORKER_PORT", 8000));
  vector<Endpoint> endpoints = {
      Endpoint::tcp(Env::get<string>("WORKER1_HOST", "localhost"),
                    Env::get<int>("WORKER1_PORT", 8001)),

  };

  unique_ptr<Partitioner> partitioner =
      make_unique<NaivePipelinePartitioner>(NaivePartitionerConfig({2, 1}));

  cout << "Configured " << endpoints.size() << " remote endpoints:" << endl;
  for (const auto &ep : endpoints) {
    cout << ep.to_json().dump(4) << endl;
  }

  cout << "Local worker endpoint: " << local_worker_endpoint.to_json().dump(4) << endl;

  // hard-coded for now
  auto worker =
      std::make_unique<TCPWorker>(local_worker_endpoint, device_type == DeviceType::GPU, 4);

  CoordinatorConfig config{ParallelMode_t::PIPELINE,
                           std::move(model),
                           std::move(optimizer),
                           std::move(partitioner),
                           std::move(worker),
                           coordinator_endpoint,
                           endpoints};

  NetworkCoordinator coordinator(std::move(config));

  coordinator.initialize();

  if (!coordinator.deploy_stages()) {
    cerr << "Failed to deploy stages. Make sure workers are running." << endl;
    return 1;
  }

  coordinator.start();

  ThreadWrapper thread_wrapper({Env::get<unsigned int>("COORDINATOR_NUM_THREADS", 4)});

  thread_wrapper.execute(
      [&]() { train_model(coordinator, *train_loader, *val_loader, criterion, train_config); });

  coordinator.stop();

  return 0;
}
