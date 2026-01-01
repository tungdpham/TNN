/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "distributed/tcp_coordinator.hpp"
#include "data_augmentation/augmentation.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "data_loading/mnist_data_loader.hpp"
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

constexpr float LR_INITIAL = 0.001f; // Careful, too big can cause exploding gradients
constexpr float EPSILON = 1e-7f;

int main() {
  auto model = create_resnet9_cifar10();

  string device_type_str = Env::get<string>("DEVICE_TYPE", "CPU");

  float lr_initial = Env::get<float>("LR_INITIAL", LR_INITIAL);

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  auto optimizer = OptimizerFactory<float>::create_adam(lr_initial, 0.9f, 0.999f, 1e-8f);

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

  cout << "Creating distributed coordinator." << endl;
  NetworkCoordinator coordinator("coordinator", move(model), move(optimizer), coordinator_endpoint,
                                 endpoints);

  unique_ptr<Partitioner<float>> partitioner =
      make_unique<NaivePartitioner<float>>(NaivePartitionerConfig({2, 1}));

  coordinator.set_partitioner(move(partitioner));
  coordinator.initialize();

  auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();
  coordinator.set_loss_function(move(loss_function));
  cout << "Deploying stages to remote endpoints." << endl;
  for (const auto &ep : endpoints) {
    cout << "  Worker expected at " << ep.to_json().dump(4) << endl;
  }

  if (!coordinator.deploy_stages()) {
    cerr << "Failed to deploy stages. Make sure workers are running." << endl;
    return 1;
  }

  coordinator.start();

  CIFAR10DataLoader<float> train_loader, test_loader;

  create_cifar10_dataloader("./data", train_loader, test_loader);

  auto train_transform =
      AugmentationBuilder<float>()
          .random_crop(0.5f, 4)
          .horizontal_flip(0.5f)
          .cutout(0.5f, 8)
          .normalize({0.49139968, 0.48215827, 0.44653124}, {0.24703233f, 0.24348505f, 0.26158768f})
          .build();
  cout << "Configuring data augmentation for training." << endl;
  train_loader.set_augmentation(move(train_transform));

  auto val_transform =
      AugmentationBuilder<float>()
          .normalize({0.49139968, 0.48215827, 0.44653124}, {0.24703233f, 0.24348505f, 0.26158768f})
          .build();
  cout << "Configuring data normalization for test." << endl;
  test_loader.set_augmentation(move(val_transform));
  Tensor<float> batch_data, batch_labels;

  ThreadWrapper thread_wrapper({Env::get<unsigned int>("COORDINATOR_NUM_THREADS", 4)});

  thread_wrapper.execute([&coordinator, &train_loader, &test_loader, &train_config]() {
    train_model(coordinator, train_loader, test_loader, train_config);
  });

  coordinator.stop();

  return 0;
}