#include "data_loading/mnist_data_loader.hpp"
#include "nn/example_models.hpp"
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
using namespace std;

constexpr float LR_INITIAL = 0.01f;
constexpr float EPSILON = 1e-15f;
constexpr int BATCH_SIZE = 64;
constexpr int NUM_MICROBATCHES = 1;
constexpr int NUM_EPOCHS = 1;
constexpr size_t PROGRESS_PRINT_INTERVAL = 100;

int main() {
  // Load environment variables from .env file
  cout << "Loading environment variables..." << endl;
  if (!load_env_file("./.env")) {
    cout << "No .env file found, using system environment variables only." << endl;
  }

  // Get training parameters from environment or use defaults
  int epochs = get_env<int>("EPOCHS", NUM_EPOCHS);
  int batch_size = get_env<int>("BATCH_SIZE", BATCH_SIZE);
  float lr_initial = get_env<float>("LR_INITIAL", LR_INITIAL);
  int num_microbatches = get_env<int>("NUM_MICROBATCHES", NUM_MICROBATCHES);
  int progress_print_interval = get_env<int>("PROGRESS_PRINT_INTERVAL", PROGRESS_PRINT_INTERVAL);

  auto model = create_mnist_trainer();

  ;

  cout << "Training Parameters:" << endl;
  cout << "  Epochs: " << epochs << endl;
  cout << "  Batch Size: " << batch_size << endl;
  cout << "  Initial Learning Rate: " << lr_initial << endl;
  cout << "  Number of Microbatches: " << num_microbatches << endl;

  Endpoint coordinator_endpoint = Endpoint::network(
      get_env<string>("COORDINATOR_HOST", "localhost"), get_env<int>("COORDINATOR_PORT", 8000));

  vector<Endpoint> endpoints = {
      Endpoint::network(get_env<string>("WORKER_HOST_8001", "localhost"), 8001),
      Endpoint::network(get_env<string>("WORKER_HOST_8002", "localhost"), 8002),
  };

  cout << "\nCreating distributed coordinator..." << endl;
  DistributedCoordinator coordinator(std::move(model), coordinator_endpoint, endpoints);

  coordinator.set_partitioner(make_unique<NaivePartitioner<float>>());

  cout << "\nDeploying stages to remote endpoints..." << endl;
  for (auto &ep : endpoints) {
    cout << "  Worker expected at " << ep.to_json().dump(4) << endl;
  }

  if (!coordinator.deploy_stages()) {
    cerr << "Failed to deploy stages. Make sure workers are running." << endl;
    return 1;
  }

  coordinator.start();

  MNISTDataLoader<float> train_loader, test_loader;

  if (!train_loader.load_data("./data/mnist/train.csv")) {
    cerr << "Failed to load training data!" << endl;
    return -1;
  }

  if (!test_loader.load_data("./data/mnist/test.csv")) {
    cerr << "Failed to load test data!" << endl;
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

  auto epoch_start = chrono::high_resolution_clock::now();

  while (true) {
    auto get_next_batch_start = chrono::high_resolution_clock::now();
    bool is_valid_batch = train_loader.get_next_batch(batch_data, batch_labels);
    if (!is_valid_batch) {
      break;
    }
    auto get_next_batch_end = chrono::high_resolution_clock::now();
    auto get_next_batch_duration =
        chrono::duration_cast<chrono::microseconds>(get_next_batch_end - get_next_batch_start);

    float loss = 0.0f, avg_accuracy = 0.0f;
    auto split_start = chrono::high_resolution_clock::now();

    vector<Tensor<float>> micro_batches(
        num_microbatches,
        Tensor<float>({batch_data.batch_size() / num_microbatches, batch_data.channels(),
                       batch_data.height(), batch_data.width()}));
    split(batch_data, micro_batches, num_microbatches);

    vector<Tensor<float>> micro_batch_labels(
        num_microbatches,
        Tensor<float>({batch_labels.batch_size() / num_microbatches, batch_labels.channels(),
                       batch_labels.height(), batch_labels.width()}));
    split(batch_labels, micro_batch_labels, num_microbatches);
    auto split_end = chrono::high_resolution_clock::now();
    auto split_duration = chrono::duration_cast<chrono::microseconds>(split_end - split_start);

    auto forward_start = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < micro_batches.size(); ++i) {
      coordinator.forward(micro_batches[i], i);
    }

    // Wait for all forward jobs to complete with a timeout
    coordinator.join(CommandType::FORWARD_JOB, num_microbatches, 60);

    auto forward_end = chrono::high_resolution_clock::now();
    auto forward_duration =
        chrono::duration_cast<chrono::microseconds>(forward_end - forward_start);
    auto compute_loss_start = chrono::high_resolution_clock::now();

    vector<Message> all_messages = coordinator.dequeue_all_messages(CommandType::FORWARD_JOB);

    vector<Job<float>> forward_jobs;
    for (auto &message : all_messages) {
      if (message.header.command_type == CommandType::FORWARD_JOB) {
        forward_jobs.push_back(message.get<Job<float>>());
      }
    }

    vector<Job<float>> backward_jobs;
    for (auto &job : forward_jobs) {
      loss += loss_function->compute_loss(job.data, micro_batch_labels[job.micro_batch_id]);
      avg_accuracy += compute_class_accuracy(job.data, micro_batch_labels[job.micro_batch_id]);

      Tensor<float> gradient =
          loss_function->compute_gradient(job.data, micro_batch_labels[job.micro_batch_id]);

      Job<float> backward_job{gradient, job.micro_batch_id};

      backward_jobs.push_back(backward_job);
    }

    loss /= num_microbatches;
    avg_accuracy /= num_microbatches;

    auto compute_loss_end = chrono::high_resolution_clock::now();
    auto compute_loss_duration =
        chrono::duration_cast<chrono::microseconds>(compute_loss_end - compute_loss_start);

    auto backward_start = chrono::high_resolution_clock::now();

    for (auto &job : backward_jobs) {
      coordinator.backward(job.data, job.micro_batch_id);
    }

    coordinator.join(CommandType::BACKWARD_JOB, num_microbatches, 60);

    coordinator.dequeue_all_messages(CommandType::BACKWARD_JOB);

    auto backward_end = chrono::high_resolution_clock::now();
    auto backward_duration =
        chrono::duration_cast<chrono::microseconds>(backward_end - backward_start);

    auto update_start = chrono::high_resolution_clock::now();
    coordinator.update_parameters();

    auto update_end = chrono::high_resolution_clock::now();
    auto update_duration = chrono::duration_cast<chrono::microseconds>(update_end - update_start);

    if (batch_index % progress_print_interval == 0) {
      cout << "Get batch completed in " << get_next_batch_duration.count() << " microseconds"
           << endl;
      cout << "Split completed in " << split_duration.count() << " microseconds" << endl;
      cout << "Forward pass completed in " << forward_duration.count() << " microseconds" << endl;
      cout << "Loss computation completed in " << compute_loss_duration.count() << " microseconds"
           << endl;
      cout << "Backward pass completed in " << backward_duration.count() << " microseconds" << endl;
      cout << "Parameter update completed in " << update_duration.count() << " microseconds"
           << endl;
      cout << "Batch " << batch_index << "/" << train_loader.size() / train_loader.get_batch_size()
           << " - Loss: " << loss << ", Accuracy: " << avg_accuracy * 100.0f << "%" << endl;
      coordinator.print_profiling_on_all_stages();
    }
    coordinator.clear_profiling_data();
    ++batch_index;
  }

  auto epoch_end = chrono::high_resolution_clock::now();
  auto epoch_duration = chrono::duration_cast<chrono::milliseconds>(epoch_end - epoch_start);
  cout << "\nEpoch " << (batch_index / train_loader.size()) + 1 << " completed in "
       << epoch_duration.count() << " milliseconds" << endl;

  double val_loss = 0.0;
  double val_accuracy = 0.0;
  int val_batches = 0;
  while (test_loader.get_batch(batch_size, batch_data, batch_labels)) {
    vector<Tensor<float>> micro_batches;
    split(batch_data, micro_batches, num_microbatches);

    vector<Tensor<float>> micro_batch_labels;

    split(batch_labels, micro_batch_labels, num_microbatches);

    for (size_t i = 0; i < micro_batches.size(); ++i) {
      coordinator.forward(micro_batches[i], i);
    }

    coordinator.join(CommandType::FORWARD_JOB, num_microbatches, 60);

    vector<Message> all_messages = coordinator.dequeue_all_messages(CommandType::FORWARD_JOB);

    if (all_messages.size() != static_cast<size_t>(num_microbatches)) {
      throw runtime_error("Unexpected number of messages: " + to_string(all_messages.size()) +
                          ", expected: " + to_string(num_microbatches));
    }

    vector<Job<float>> forward_jobs;
    for (auto &message : all_messages) {
      if (message.header.command_type == CommandType::FORWARD_JOB) {
        forward_jobs.push_back(message.get<Job<float>>());
      }
    }

    for (auto &job : forward_jobs) {
      val_loss += loss_function->compute_loss(job.data, micro_batch_labels[job.micro_batch_id]);
      val_accuracy += compute_class_accuracy(job.data, micro_batch_labels[job.micro_batch_id]);
    }
    ++val_batches;
  }

  cout << "\nValidation completed!" << endl;
  cout << "Average Validation Loss: " << (val_loss / val_batches)
       << ", Average Validation Accuracy: "
       << (val_accuracy / val_batches / NUM_MICROBATCHES) * 100.0f << "%" << endl;
  return 0;
}