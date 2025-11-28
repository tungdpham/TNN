/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "data_augmentation/augmentation.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "data_loading/mnist_data_loader.hpp"
#include "nn/example_models.hpp"
#include "nn/sequential.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "pipeline/distributed_coordinator.hpp"
#include "pipeline/train.hpp"
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
  if (!load_env_file("./.env")) {
    std::cout << "No .env file found, using system environment variables only." << std::endl;
  }

  auto model = create_cifar10_trainer_v2();

  string device_type_str = get_env<string>("DEVICE_TYPE", "CPU");

  float lr_initial = get_env<float>("LR_INITIAL", LR_INITIAL);

  TrainingConfig train_config;
  train_config.load_from_env();
  train_config.print_config();

  auto optimizer = std::make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, EPSILON);
  model.set_optimizer(std::move(optimizer));

  Endpoint coordinator_endpoint =
      Endpoint::network(get_env<std::string>("COORDINATOR_HOST", "localhost"),
                        get_env<int>("COORDINATOR_PORT", 8000));

  std::vector<Endpoint> endpoints = {
      Endpoint::network(get_env<std::string>("WORKER1_HOST", "localhost"),
                        get_env<int>("WORKER1_PORT", 8001)),
      Endpoint::network(get_env<std::string>("WORKER2_HOST", "localhost"),
                        get_env<int>("WORKER2_PORT", 8002)),

  };

  std::cout << "Configured " << endpoints.size() << " remote endpoints:" << std::endl;
  for (const auto &ep : endpoints) {
    std::cout << ep.to_json().dump(4) << std::endl;
  }

  std::cout << "Creating distributed coordinator." << std::endl;
  DistributedCoordinator coordinator(std::move(model), coordinator_endpoint, endpoints);

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

  coordinator.start();

  CIFAR10DataLoader<float> train_loader, test_loader;

  create_cifar10_dataloader("./data", train_loader, test_loader);

  auto aug_strategy = AugmentationBuilder<float>()
                          .horizontal_flip(0.25f)
                          .rotation(0.3f, 10.0f)
                          .brightness(0.3f, 0.15f)
                          .contrast(0.3f, 0.15f)
                          .gaussian_noise(0.3f, 0.05f)
                          .build();
  std::cout << "Configuring data augmentation for training." << std::endl;
  train_loader.set_augmentation(std::move(aug_strategy));

  Tensor<float> batch_data, batch_labels;

  ThreadWrapper thread_wrapper({get_env<unsigned int>("COORDINATOR_NUM_THREADS", 4)});

  thread_wrapper.execute([&coordinator, &train_loader, &test_loader, &train_config]() {
    train_model(coordinator, train_loader, test_loader, train_config);
  });

  coordinator.stop();

  return 0;
}