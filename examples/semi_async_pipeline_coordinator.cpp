

#include "data_augmentation/augmentation.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "data_loading/mnist_data_loader.hpp"
#include "nn/layers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "pipeline/distributed_coordinator.hpp"
#include "tensor/tensor.hpp"
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

namespace semi_async_constants {
constexpr float LR_INITIAL = 0.01f;
constexpr float EPSILON = 1e-15f;
constexpr int BATCH_SIZE = 64;
constexpr int NUM_MICROBATCHES = 2;
constexpr int NUM_EPOCHS = 5;
constexpr size_t PROGRESS_PRINT_INTERVAL = 100;
} // namespace semi_async_constants

Sequential<float> create_mnist_trainer() {
  auto model = SequentialBuilder<float>("mnist_cnn_model")
                   .input({1, 32, 32})
                   .conv2d(8, 5, 5, 1, 1, 0, 0, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")
                   .conv2d(16, 1, 1, 1, 1, 0, 0, true, "conv2_1x1")
                   .batchnorm(1e-5f, 0.1f, true, "bn2")
                   .activation("relu", "relu2")
                   .conv2d(48, 5, 5, 1, 1, 0, 0, true, "conv3")
                   .batchnorm(1e-5f, 0.1f, true, "bn3")
                   .activation("relu", "relu3")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                   .flatten("flatten")
                   .dense(::mnist_constants::NUM_CLASSES, "linear", true, "output")
                   .build();

  auto optimizer =
      std::make_unique<Adam<float>>(semi_async_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
  model.set_optimizer(std::move(optimizer));

  auto loss_function = LossFactory<float>::create_crossentropy(semi_async_constants::EPSILON);
  model.set_loss_function(std::move(loss_function));
  return model;
}

Sequential<float> create_cifar10_trainer_v1() {
  auto model = SequentialBuilder<float>("cifar10_cnn_classifier_v1")
                   .input({3, 32, 32})
                   .conv2d(16, 3, 3, 1, 1, 0, 0, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(3, 3, 3, 3, 0, 0, "maxpool1")
                   .conv2d(64, 3, 3, 1, 1, 0, 0, true, "conv2")
                   .batchnorm(1e-5f, 0.1f, true, "bn2")
                   .activation("relu", "relu2")
                   .maxpool2d(4, 4, 4, 4, 0, 0, "maxpool2")
                   .flatten("flatten")
                   .dense(10, "linear", true, "fc1")
                   .build();

  auto optimizer = std::make_unique<SGD<float>>(semi_async_constants::LR_INITIAL, 0.9f);
  model.set_optimizer(std::move(optimizer));

  auto loss_function = LossFactory<float>::create_crossentropy(semi_async_constants::EPSILON);
  model.set_loss_function(std::move(loss_function));
  return model;
}

Sequential<float> create_cifar10_trainer_v2() {
  auto model = SequentialBuilder<float>("cifar10_cnn_classifier")
                   .input({3, 32, 32})
                   .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv0")
                   .batchnorm(1e-5f, 0.1f, true, "bn0")
                   .activation("relu", "relu0")
                   .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool0")
                   .conv2d(128, 3, 3, 1, 1, 1, 1, true, "conv2")
                   .batchnorm(1e-5f, 0.1f, true, "bn2")
                   .activation("relu", "relu2")
                   .conv2d(128, 3, 3, 1, 1, 1, 1, true, "conv3")
                   .batchnorm(1e-5f, 0.1f, true, "bn3")
                   .activation("relu", "relu3")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool1")
                   .conv2d(256, 3, 3, 1, 1, 1, 1, true, "conv4")
                   .batchnorm(1e-5f, 0.1f, true, "bn5")
                   .activation("relu", "relu5")
                   .conv2d(256, 3, 3, 1, 1, 1, 1, true, "conv5")
                   .activation("relu", "relu6")
                   .conv2d(256, 3, 3, 1, 1, 1, 1, true, "conv6")
                   .batchnorm(1e-5f, 0.1f, true, "bn6")
                   .activation("relu", "relu6")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                   .conv2d(512, 3, 3, 1, 1, 1, 1, true, "conv7")
                   .batchnorm(1e-5f, 0.1f, true, "bn8")
                   .activation("relu", "relu7")
                   .conv2d(512, 3, 3, 1, 1, 1, 1, true, "conv8")
                   .batchnorm(1e-5f, 0.1f, true, "bn9")
                   .activation("relu", "relu8")
                   .conv2d(512, 3, 3, 1, 1, 1, 1, true, "conv9")
                   .batchnorm(1e-5f, 0.1f, true, "bn10")
                   .activation("relu", "relu9")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool3")
                   .flatten("flatten")
                   .dense(512, "linear", true, "fc0")
                   .activation("relu", "relu10")
                   .dense(10, "linear", true, "fc1")
                   .build();

  auto optimizer =
      std::make_unique<Adam<float>>(semi_async_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
  model.set_optimizer(std::move(optimizer));

  auto loss_function = LossFactory<float>::create_crossentropy(semi_async_constants::EPSILON);
  model.set_loss_function(std::move(loss_function));
  return model;
}

std::string get_host(const std::string &env_var, const std::string &default_host) {
#ifdef _WIN32
#ifdef _MSC_VER

  char *env_value = nullptr;
  size_t len = 0;
  if (_dupenv_s(&env_value, &len, env_var.c_str()) == 0 && env_value != nullptr) {
    std::string result(env_value);
    free(env_value);
    return result;
  }
  return default_host;
#else

  const char *env_value = std::getenv(env_var.c_str());
  return env_value ? std::string(env_value) : default_host;
#endif
#else
  const char *env_value = std::getenv(env_var.c_str());
  return env_value ? std::string(env_value) : default_host;
#endif
}

void get_cifar10_data_loaders(data_loading::CIFAR10DataLoader<float> &train_loader,
                              data_loading::CIFAR10DataLoader<float> &test_loader) {
  if (!train_loader.load_multiple_files({"./data/cifar-10-batches-bin/data_batch_1.bin",
                                         "./data/cifar-10-batches-bin/data_batch_2.bin",
                                         "./data/cifar-10-batches-bin/data_batch_3.bin",
                                         "./data/cifar-10-batches-bin/data_batch_4.bin",
                                         "./data/cifar-10-batches-bin/data_batch_5.bin"})) {
    throw std::runtime_error("Failed to load training data!");
  }

  if (!test_loader.load_data("./data/cifar-10-batches-bin/test_batch.bin")) {
    throw std::runtime_error("Failed to load test data!");
  }

  train_loader.shuffle();
  test_loader.shuffle();

  train_loader.prepare_batches(semi_async_constants::BATCH_SIZE);
  test_loader.prepare_batches(semi_async_constants::BATCH_SIZE);

  train_loader.reset();
  test_loader.reset();
}

void get_mnist_data_loaders(data_loading::MNISTDataLoader<float> &train_loader,
                            data_loading::MNISTDataLoader<float> &test_loader) {
  if (!train_loader.load_data("./data/mnist/train.csv")) {
    throw std::runtime_error("Failed to load training data!");
  }

  if (!test_loader.load_data("./data/mnist/test.csv")) {
    throw std::runtime_error("Failed to load test data!");
  }

  train_loader.shuffle();
  test_loader.shuffle();

  train_loader.prepare_batches(semi_async_constants::BATCH_SIZE);
  test_loader.prepare_batches(semi_async_constants::BATCH_SIZE);

  train_loader.reset();
  test_loader.reset();
}

ClassResult train_semi_async_epoch(DistributedPipelineCoordinator<float> &coordinator,
                                   CIFAR10DataLoader<float> &train_loader);
ClassResult validate_semi_async_epoch(DistributedPipelineCoordinator<float> &coordinator,
                                      CIFAR10DataLoader<float> &test_loader);

int main() {
  auto model = create_cifar10_trainer_v2();

  model.print_config();

  std::string coordinator_host = get_host("COORDINATOR_HOST", "localhost");

  std::vector<DistributedPipelineCoordinator<float>::RemoteEndpoint> endpoints = {
      {get_host("WORKER_HOST_8001", "localhost"), 8001, "stage_0"},
      {get_host("WORKER_HOST_8002", "localhost"), 8002, "stage_1"},

  };

  std::cout << "Using coordinator host: " << coordinator_host << std::endl;

  std::cout << "Configured " << endpoints.size() << " remote endpoints:" << std::endl;
  for (const auto &ep : endpoints) {
    std::cout << "  " << ep.stage_id << " -> " << ep.host << ":" << ep.port << std::endl;
  }

  std::cout << "Creating distributed coordinator." << std::endl;
  DistributedPipelineCoordinator<float> coordinator(
      std::move(model), endpoints, semi_async_constants::NUM_MICROBATCHES, coordinator_host, 8000);

  std::cout << "Deploying stages to remote endpoints." << std::endl;
  for (const auto &ep : endpoints) {
    std::cout << "  Worker expected at " << ep.host << ":" << ep.port << std::endl;
  }

  if (!coordinator.deploy_stages()) {
    std::cerr << "Failed to deploy stages. Make sure workers are running." << std::endl;
    return 1;
  }

  coordinator.start();

  // data_loading::MNISTDataLoader<float> train_loader, test_loader;

  data_loading::CIFAR10DataLoader<float> train_loader, test_loader;

  get_cifar10_data_loaders(train_loader, test_loader);

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
  std::cout << "Setting MKL number of threads to: " << 8 << std::endl;
  mkl_set_threading_layer(MKL_THREADING_TBB);
#endif

#ifdef USE_TBB
  tbb::task_arena arena(tbb::task_arena::constraints{}.set_max_concurrency(2));
  std::cout << "TBB max threads limited to: " << arena.max_concurrency() << std::endl;
  arena.execute([&]() {
#endif
    for (size_t epoch = 0; epoch < semi_async_constants::NUM_EPOCHS; ++epoch) {
      std::cout << "\n=== Epoch " << (epoch + 1) << "/" << semi_async_constants::NUM_EPOCHS
                << " ===" << std::endl;
      train_loader.reset();
      test_loader.reset();

      train_loader.shuffle();

      train_semi_async_epoch(coordinator, train_loader);

      validate_semi_async_epoch(coordinator, test_loader);

      train_loader.prepare_batches(semi_async_constants::BATCH_SIZE);
    }
#ifdef USE_TBB
  });
#endif
  return 0;
}

ClassResult train_semi_async_epoch(DistributedPipelineCoordinator<float> &coordinator,
                                   CIFAR10DataLoader<float> &train_loader) {
  Tensor<float> batch_data, batch_labels;

  size_t batch_index = 0;

  auto epoch_start = std::chrono::high_resolution_clock::now();

  float total_loss = 0.0f;

  while (train_loader.get_next_batch(batch_data, batch_labels)) {
    auto split_start = std::chrono::high_resolution_clock::now();

    std::vector<Tensor<float>> micro_batches =
        batch_data.split(semi_async_constants::NUM_MICROBATCHES);

    std::vector<Tensor<float>> micro_batch_labels =
        batch_labels.split(semi_async_constants::NUM_MICROBATCHES);

    auto split_end = std::chrono::high_resolution_clock::now();
    auto split_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(split_end - split_start);

    auto process_start = std::chrono::high_resolution_clock::now();
    total_loss += coordinator.async_process_batch(micro_batches, micro_batch_labels);
    auto process_end = std::chrono::high_resolution_clock::now();
    auto process_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(process_end - process_start);

    auto update_start = std::chrono::high_resolution_clock::now();
    coordinator.update_parameters();

    auto update_end = std::chrono::high_resolution_clock::now();
    auto update_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(update_end - update_start);

    if ((batch_index + 1) % semi_async_constants::PROGRESS_PRINT_INTERVAL == 0) {

      std::cout << "Split completed in " << split_duration.count() << " microseconds" << std::endl;
      std::cout << "Async process completed in " << process_duration.count() << " microseconds"
                << std::endl;
      std::cout << "Parameter update completed in " << update_duration.count() << " microseconds"
                << std::endl;
      std::cout << "Batch " << batch_index + 1 << "/"
                << train_loader.size() / train_loader.get_batch_size() << std::endl;
      coordinator.balance_load();
      coordinator.print_profiling_on_all_stages();
      coordinator.clear_profiling_data();
    }
    ++batch_index;
  }

  auto epoch_end = std::chrono::high_resolution_clock::now();
  auto epoch_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
  std::cout << "\nEpoch " << (batch_index / train_loader.size()) + 1 << " completed in "
            << epoch_duration.count() << " milliseconds" << std::endl;
  return {total_loss / batch_index, -1.0f};
}

ClassResult validate_semi_async_epoch(DistributedPipelineCoordinator<float> &coordinator,
                                      CIFAR10DataLoader<float> &test_loader) {
  Tensor<float> batch_data, batch_labels;

  double val_loss = 0.0;
  double val_accuracy = 0.0;
  int val_batches = 0;

  while (test_loader.get_batch(semi_async_constants::BATCH_SIZE, batch_data, batch_labels)) {

    std::vector<Tensor<float>> micro_batches =
        batch_data.split(semi_async_constants::NUM_MICROBATCHES);

    std::vector<Tensor<float>> micro_batch_labels =
        batch_labels.split(semi_async_constants::NUM_MICROBATCHES);

    for (size_t i = 0; i < micro_batches.size(); ++i) {
      coordinator.forward(micro_batches[i], i);
    }

    coordinator.join(CommandType::FORWARD_TASK, semi_async_constants::NUM_MICROBATCHES, 60);

    std::vector<tpipeline::Message<float>> all_messages =
        coordinator.dequeue_all_messages(tpipeline::CommandType::FORWARD_TASK);

    if (all_messages.size() != semi_async_constants::NUM_MICROBATCHES) {
      throw std::runtime_error(
          "Unexpected number of messages: " + std::to_string(all_messages.size()) +
          ", expected: " + std::to_string(semi_async_constants::NUM_MICROBATCHES));
    }

    std::vector<tpipeline::Task<float>> forward_tasks;
    for (const auto &message : all_messages) {
      if (message.command_type == CommandType::FORWARD_TASK) {
        forward_tasks.push_back(message.get_task());
      }
    }

    for (auto &task : forward_tasks) {
      task.data.apply_softmax();
      val_loss += coordinator.compute_loss(task.data, micro_batch_labels[task.micro_batch_id]);
      val_accuracy += ::utils::compute_class_accuracy<float>(
          task.data, micro_batch_labels[task.micro_batch_id]);
    }
    ++val_batches;
  }

  std::cout << "Validation completed." << std::endl;
  std::cout << "Average Validation Loss: " << (val_loss / val_batches)
            << ", Average Validation Accuracy: "
            << (val_accuracy * 100.0f / val_batches / semi_async_constants::NUM_MICROBATCHES) << "%"
            << std::endl;
  return {static_cast<float>(val_loss / val_batches),
          static_cast<float>(
              (val_accuracy * 100.0f / val_batches / semi_async_constants::NUM_MICROBATCHES))};
}