#include "data_loading/mnist_data_loader.hpp"
#include "nn/sequential.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "pipeline/distributed_coordinator.hpp"
#include "tensor/tensor.hpp"
#include "utils/env.hpp"

#include "utils/utils_extended.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace tnn;

namespace mnist_constants {
constexpr float LR_INITIAL = 0.01f;
constexpr float EPSILON = 1e-15f;
constexpr int BATCH_SIZE = 64;
constexpr int NUM_MICROBATCHES = 1;
constexpr int NUM_EPOCHS = 1;
constexpr size_t PROGRESS_PRINT_INTERVAL = 100;
} // namespace mnist_constants

Sequential<float> create_demo_model() {
  auto model = SequentialBuilder<float>("optimized_mnist_cnn_classifier")
                   .input({1, 28, 28})
                   .conv2d(16, 3, 3, 1, 1, 0, 0, true, "conv1")
                   .activation("relu", "relu1")
                   .maxpool2d(3, 3, 3, 3, 0, 0, "maxpool1")
                   .conv2d(64, 3, 3, 1, 1, 0, 0, true, "conv2")
                   .activation("relu", "relu2")
                   .maxpool2d(4, 4, 4, 4, 0, 0, "maxpool2")
                   .flatten("flatten")
                   .dense(10, "linear", true, "fc1")
                   .activation("softmax", "softmax_output")
                   .build();

  auto optimizer = std::make_unique<Adam<float>>(mnist_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
  model.set_optimizer(std::move(optimizer));

  auto loss_function = LossFactory<float>::create_crossentropy(mnist_constants::EPSILON);
  model.set_loss_function(std::move(loss_function));
  return model;
}

int main() {
  // Load environment variables from .env file
  std::cout << "Loading environment variables..." << std::endl;
  if (!load_env_file("./.env")) {
    std::cout << "No .env file found, using system environment variables only." << std::endl;
  }

  // Get training parameters from environment or use defaults
  const int epochs = get_env<int>("EPOCHS", mnist_constants::NUM_EPOCHS);
  const int batch_size = get_env<int>("BATCH_SIZE", mnist_constants::BATCH_SIZE);
  const float lr_initial = get_env<float>("LR_INITIAL", mnist_constants::LR_INITIAL);
  const int num_microbatches = get_env<int>("NUM_MICROBATCHES", mnist_constants::NUM_MICROBATCHES);
  const int progress_print_interval =
      get_env<int>("PROGRESS_PRINT_INTERVAL", mnist_constants::PROGRESS_PRINT_INTERVAL);

  auto model = create_demo_model();

  model.print_config();

  std::cout << "Training Parameters:" << std::endl;
  std::cout << "  Epochs: " << epochs << std::endl;
  std::cout << "  Batch Size: " << batch_size << std::endl;
  std::cout << "  Initial Learning Rate: " << lr_initial << std::endl;
  std::cout << "  Number of Microbatches: " << num_microbatches << std::endl;

  Endpoint coordinator_endpoint =
      Endpoint::network(get_env<std::string>("COORDINATOR_HOST", "localhost"),
                        get_env<int>("COORDINATOR_PORT", 8000));

  std::vector<Endpoint> endpoints = {
      Endpoint::network(get_env<std::string>("WORKER_HOST_8001", "localhost"), 8001),
      Endpoint::network(get_env<std::string>("WORKER_HOST_8002", "localhost"), 8002),
  };

  std::cout << "\nCreating distributed coordinator..." << std::endl;
  DistributedCoordinator coordinator(std::move(model), coordinator_endpoint, endpoints);

  coordinator.set_partitioner(std::make_unique<NaivePartitioner<float>>());

  std::cout << "\nDeploying stages to remote endpoints..." << std::endl;
  for (const auto &ep : endpoints) {
    std::cout << "  Worker expected at " << ep.to_json().dump(4) << std::endl;
  }

  if (!coordinator.deploy_stages()) {
    std::cerr << "Failed to deploy stages. Make sure workers are running." << std::endl;
    return 1;
  }

  std::cout << "\nStarting distributed pipeline..." << std::endl;
  coordinator.start();

  MNISTDataLoader<float> train_loader, test_loader;

  if (!train_loader.load_data("./data/mnist/train.csv")) {
    std::cerr << "Failed to load training data!" << std::endl;
    return -1;
  }

  if (!test_loader.load_data("./data/mnist/test.csv")) {
    std::cerr << "Failed to load test data!" << std::endl;
    return -1;
  }

  Tensor<float> batch_data, batch_labels;

  auto loss_function = LossFactory<float>::create("crossentropy");

  size_t batch_index = 0;

  train_loader.shuffle();

  train_loader.prepare_batches(batch_size);
  test_loader.prepare_batches(batch_size);

  train_loader.reset();
  test_loader.reset();

  auto epoch_start = std::chrono::high_resolution_clock::now();

  while (true) {
    auto get_next_batch_start = std::chrono::high_resolution_clock::now();
    bool is_valid_batch = train_loader.get_next_batch(batch_data, batch_labels);
    if (!is_valid_batch) {
      break;
    }
    auto get_next_batch_end = std::chrono::high_resolution_clock::now();
    auto get_next_batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        get_next_batch_end - get_next_batch_start);

    float loss = 0.0f, avg_accuracy = 0.0f;
    auto split_start = std::chrono::high_resolution_clock::now();

    std::vector<Tensor<float>> micro_batches = split(batch_data, num_microbatches);

    std::vector<Tensor<float>> micro_batch_labels = split(batch_labels, num_microbatches);
    auto split_end = std::chrono::high_resolution_clock::now();
    auto split_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(split_end - split_start);

    auto forward_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < micro_batches.size(); ++i) {
      coordinator.forward(micro_batches[i], i);
    }

    // Wait for all forward jobs to complete with a timeout
    coordinator.join(CommandType::FORWARD_JOB, num_microbatches, 60);

    auto forward_end = std::chrono::high_resolution_clock::now();
    auto forward_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(forward_end - forward_start);
    auto compute_loss_start = std::chrono::high_resolution_clock::now();

    std::vector<Message> all_messages = coordinator.dequeue_all_messages(CommandType::FORWARD_JOB);

    std::vector<Job<float>> forward_jobs;
    for (const auto &message : all_messages) {
      if (message.header.command_type == CommandType::FORWARD_JOB) {
        forward_jobs.push_back(message.get<Job<float>>());
      }
    }

    std::vector<Job<float>> backward_jobs;
    for (auto &job : forward_jobs) {
      loss += loss_function->compute_loss(job.data, micro_batch_labels[job.micro_batch_id]);
      avg_accuracy +=
          compute_class_accuracy<float>(job.data, micro_batch_labels[job.micro_batch_id]);

      Tensor<float> gradient =
          loss_function->compute_gradient(job.data, micro_batch_labels[job.micro_batch_id]);

      Job<float> backward_job{gradient, job.micro_batch_id};

      backward_jobs.push_back(backward_job);
    }

    loss /= num_microbatches;
    avg_accuracy /= num_microbatches;

    auto compute_loss_end = std::chrono::high_resolution_clock::now();
    auto compute_loss_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        compute_loss_end - compute_loss_start);

    auto backward_start = std::chrono::high_resolution_clock::now();

    for (const auto &job : backward_jobs) {
      coordinator.backward(job.data, job.micro_batch_id);
    }

    coordinator.join(CommandType::BACKWARD_JOB, num_microbatches, 60);

    coordinator.dequeue_all_messages(CommandType::BACKWARD_JOB);

    auto backward_end = std::chrono::high_resolution_clock::now();
    auto backward_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(backward_end - backward_start);

    auto update_start = std::chrono::high_resolution_clock::now();
    coordinator.update_parameters();

    auto update_end = std::chrono::high_resolution_clock::now();
    auto update_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(update_end - update_start);

    if (batch_index % progress_print_interval == 0) {
      std::cout << "Get batch completed in " << get_next_batch_duration.count() << " microseconds"
                << std::endl;
      std::cout << "Split completed in " << split_duration.count() << " microseconds" << std::endl;
      std::cout << "Forward pass completed in " << forward_duration.count() << " microseconds"
                << std::endl;
      std::cout << "Loss computation completed in " << compute_loss_duration.count()
                << " microseconds" << std::endl;
      std::cout << "Backward pass completed in " << backward_duration.count() << " microseconds"
                << std::endl;
      std::cout << "Parameter update completed in " << update_duration.count() << " microseconds"
                << std::endl;
      std::cout << "Batch " << batch_index << "/"
                << train_loader.size() / train_loader.get_batch_size() << " - Loss: " << loss
                << ", Accuracy: " << avg_accuracy * 100.0f << "%" << std::endl;
      coordinator.print_profiling_on_all_stages();
    }
    coordinator.clear_profiling_data();
    ++batch_index;
  }

  auto epoch_end = std::chrono::high_resolution_clock::now();
  auto epoch_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
  std::cout << "\nEpoch " << (batch_index / train_loader.size()) + 1 << " completed in "
            << epoch_duration.count() << " milliseconds" << std::endl;

  double val_loss = 0.0;
  double val_accuracy = 0.0;
  int val_batches = 0;
  while (test_loader.get_batch(batch_size, batch_data, batch_labels)) {

    std::vector<Tensor<float>> micro_batches = split(batch_data, num_microbatches);

    std::vector<Tensor<float>> micro_batch_labels = split(batch_labels, num_microbatches);

    for (size_t i = 0; i < micro_batches.size(); ++i) {
      coordinator.forward(micro_batches[i], i);
    }

    coordinator.join(CommandType::FORWARD_JOB, num_microbatches, 60);

    std::vector<Message> all_messages = coordinator.dequeue_all_messages(CommandType::FORWARD_JOB);

    if (all_messages.size() != static_cast<size_t>(num_microbatches)) {
      throw std::runtime_error(
          "Unexpected number of messages: " + std::to_string(all_messages.size()) +
          ", expected: " + std::to_string(num_microbatches));
    }

    std::vector<Job<float>> forward_jobs;
    for (const auto &message : all_messages) {
      if (message.header.command_type == CommandType::FORWARD_JOB) {
        forward_jobs.push_back(message.get<Job<float>>());
      }
    }

    for (auto &job : forward_jobs) {
      val_loss += loss_function->compute_loss(job.data, micro_batch_labels[job.micro_batch_id]);
      val_accuracy +=
          compute_class_accuracy<float>(job.data, micro_batch_labels[job.micro_batch_id]);
    }
    ++val_batches;
  }

  std::cout << "\nValidation completed!" << std::endl;
  std::cout << "Average Validation Loss: " << (val_loss / val_batches)
            << ", Average Validation Accuracy: "
            << (val_accuracy / val_batches / mnist_constants::NUM_MICROBATCHES) * 100.0f << "%"
            << std::endl;
  return 0;
}