

#include "data_augmentation/augmentation.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "data_loading/mnist_data_loader.hpp"
#include "nn/example_models.hpp"
#include "nn/layers.hpp"
#include "nn/sequential.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "pipeline/distributed_coordinator.hpp"
#include "pipeline/train.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_wrapper.hpp"
#include "utils/env.hpp"
#include "utils/mkl_utils.hpp"
#include "utils/ops.hpp"
#include "utils/utils_extended.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

using namespace tnn;
using namespace tpipeline;
using namespace data_augmentation;
using namespace data_loading;
using namespace ::utils;
using namespace tthreads;

namespace semi_async_constants {
constexpr float LR_INITIAL = 0.001f; // Careful, too big can cause exploding gradients
constexpr float EPSILON = 1e-15f;
constexpr int BATCH_SIZE = 64;
constexpr int NUM_MICROBATCHES = 2;
constexpr int NUM_EPOCHS = 1;
constexpr size_t PROGRESS_PRINT_INTERVAL = 100;
} // namespace semi_async_constants

TrainingConfig get_training_config() {
  TrainingConfig config;
  config.epochs = semi_async_constants::NUM_EPOCHS;
  config.batch_size = semi_async_constants::BATCH_SIZE;
  config.lr_decay_factor = 0.9f;
  config.lr_decay_interval = 5; // in epochs
  config.num_threads = 8;       // Typical number of P-Cores on laptop CPUs
  config.profiler_type = ProfilerType::NONE;
  config.progress_print_interval = semi_async_constants::PROGRESS_PRINT_INTERVAL;
  return config;
}

int main() {
  if (!load_env_file("./.env")) {
    std::cout << "No .env file found, using system environment variables only." << std::endl;
  }

  // auto model = create_mnist_trainer();

  auto model = create_cifar10_trainer_v1();

  auto optimizer = std::make_unique<Adam<float>>(semi_async_constants::LR_INITIAL, 0.9f, 0.999f,
                                                 semi_async_constants::EPSILON);
  model.set_optimizer(std::move(optimizer));

  // auto model = create_cifar10_trainer_v2();

  model.print_config();

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

  coordinator.set_partitioner(std::make_unique<partitioner::NaivePartitioner<float>>());
  coordinator.initialize();

  auto loss_function = LossFactory<float>::create_softmax_crossentropy();
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

#ifdef USE_MKL
  mkl::initialize_mkl();
#endif

  ThreadWrapper thread_wrapper({get_env<unsigned int>("COORDINATOR_NUM_THREADS", 4)});

  thread_wrapper.execute([&coordinator, &train_loader, &test_loader]() {
    train_model(coordinator, train_loader, test_loader, get_training_config());
  });

  coordinator.stop();

  return 0;
}