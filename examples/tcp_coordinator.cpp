/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "distributed/tcp_coordinator.hpp"
#include "data_augmentation/augmentation.hpp"
#include "data_loading/data_loader.hpp"
#include "distributed/train.hpp"
#include "nn/example_models.hpp"
#include "nn/sequential.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_wrapper.hpp"
#include "utils/env.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

using namespace tnn;
using namespace std;

int main() {
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
  auto optimizer = OptimizerFactory<float>::create_adam(0.001f, 0.9f, 0.999f, 1e-8f);
  auto scheduler = SchedulerFactory<float>::create_step_lr(optimizer.get(), 10, 0.1f);

  Endpoint coordinator_endpoint = Endpoint::tcp(Env::get<string>("COORDINATOR_HOST", "localhost"),
                                                Env::get<int>("COORDINATOR_PORT", 8000));

  vector<Endpoint> endpoints = {
      Endpoint::tcp(Env::get<string>("WORKER1_HOST", "localhost"),
                    Env::get<int>("WORKER1_PORT", 8001)),
      Endpoint::tcp(Env::get<string>("WORKER2_HOST", "localhost"),
                    Env::get<int>("WORKER2_PORT", 8002)),

  };

  cout << "Configured " << endpoints.size() << " remote endpoints:" << endl;
  for (const auto &ep : endpoints) {
    cout << ep.to_json().dump(4) << endl;
  }

  NetworkCoordinator coordinator("coordinator", std::move(model), std::move(optimizer),
                                 coordinator_endpoint, endpoints);

  unique_ptr<Partitioner<float>> partitioner =
      make_unique<NaivePartitioner<float>>(NaivePartitionerConfig({2, 1}));

  coordinator.set_partitioner(std::move(partitioner));
  coordinator.initialize();

  coordinator.set_loss_function(std::move(criterion));
  cout << "Deploying stages to remote endpoints." << endl;
  for (const auto &ep : endpoints) {
    cout << "  Worker expected at " << ep.to_json().dump(4) << endl;
  }

  if (!coordinator.deploy_stages()) {
    cerr << "Failed to deploy stages. Make sure workers are running." << endl;
    return 1;
  }

  coordinator.start();

  auto train_transform =
      AugmentationBuilder<float>()
          .random_crop(0.5f, 4)
          .horizontal_flip(0.5f)
          .cutout(0.5f, 8)
          .normalize({0.49139968, 0.48215827, 0.44653124}, {0.24703233f, 0.24348505f, 0.26158768f})
          .build();
  train_loader->set_augmentation(std::move(train_transform));

  auto val_transform =
      AugmentationBuilder<float>()
          .normalize({0.49139968, 0.48215827, 0.44653124}, {0.24703233f, 0.24348505f, 0.26158768f})
          .build();
  val_loader->set_augmentation(std::move(val_transform));

  ThreadWrapper thread_wrapper({Env::get<unsigned int>("COORDINATOR_NUM_THREADS", 4)});

  thread_wrapper.execute([&coordinator, &train_loader, &val_loader, &train_config]() {
    train_model(coordinator, *train_loader, *val_loader, train_config);
  });

  coordinator.stop();

  return 0;
}