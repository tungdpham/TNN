#include "nn/layers.hpp"
#include "nn/sequential.hpp"
#include "pipeline/distributed_coordinator.hpp"
#include "tensor/tensor.hpp"
#include "utils/cifar10_data_loader.hpp"
#include "utils/mnist_data_loader.hpp"
#include "utils/ops.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace tnn;
using namespace tpipeline;

namespace mnist_constants {
constexpr float LR_INITIAL = 0.001f;
constexpr float EPSILON = 1e-15f;
constexpr int NUM_MICROBATCHES =
    4;                        // Number of microbatches for distributed training
constexpr int NUM_EPOCHS = 1; // Number of epochs for training
constexpr size_t PROGRESS_PRINT_INTERVAL =
    100; // Print progress every 100 batches
} // namespace mnist_constants

// Create a simple CNN model for demonstration
Sequential<float> create_demo_model() {
  auto model = tnn::SequentialBuilder<float>("optimized_mnist_cnn_classifier")
                   // C1: First convolution layer - 5x5 kernel, stride 1, ReLU
                   // activation Input: 1x28x28 → Output: 8x24x24 (28-5+1=24)
                   .conv2d(1, 8, 5, 5, 1, 1, 0, 0, "relu", true, "conv1")
                   // P1: Max pooling layer - 3x3 blocks, stride 3
                   // Input: 8x24x24 → Output: 8x8x8 (24/3=8)
                   .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")

                   // C2: Inception-style 1x1 convolution for dimensionality
                   // reduction Input: 8x8x8 → Output: 16x8x8
                   .conv2d(8, 16, 1, 1, 1, 1, 0, 0, "relu", true, "conv2_1x1")

                   // C3: Second convolution layer - 5x5 kernel, stride 1, ReLU
                   // activation Input: 16x8x8 → Output: 48x4x4 (8-5+1=4)
                   .conv2d(16, 48, 5, 5, 1, 1, 0, 0, "relu", true, "conv3")

                   // P2: Second max pooling layer - 2x2 blocks, stride 2
                   // Input: 48x4x4 → Output: 48x2x2 (4/2=2)
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")

                   // FC1: Fully connected output layer
                   // Input: 48x2x2 = 192 features → Output: 10 classes
                   .dense(48 * 2 * 2, mnist_constants::NUM_CLASSES, "linear",
                          true, "output")
                   .build();

  auto optimizer = std::make_unique<tnn::Adam<float>>(
      mnist_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
  model.set_optimizer(std::move(optimizer));

  // Set loss function for the model
  auto loss_function =
      tnn::LossFactory<float>::create_crossentropy(mnist_constants::EPSILON);
  model.set_loss(std::move(loss_function));
  return model;
}

int main() {
  try {
    // Create the model
    auto model = create_demo_model();

    model.print_config();

    // Define remote endpoints where stages will be deployed
    std::vector<DistributedPipelineCoordinator<float>::RemoteEndpoint>
        endpoints = {
            {"localhost", 8001, "stage_0"}, // First stage
            // {"localhost", 8002, "stage_1"}, // Second stage
            // {"localhost", 8003, "stage_2"}, // Third stage
            // {"localhost", 8004, "stage_3"}  // Fourth stage
        };

    std::cout << "\nConfigured " << endpoints.size()
              << " remote endpoints:" << std::endl;
    for (const auto &ep : endpoints) {
      std::cout << "  " << ep.stage_id << " -> " << ep.host << ":" << ep.port
                << std::endl;
    }

    // Create distributed coordinator
    std::cout << "\nCreating distributed coordinator..." << std::endl;
    DistributedPipelineCoordinator<float> coordinator(
        std::move(model), endpoints, mnist_constants::NUM_MICROBATCHES, "localhost", 8000);

    // Deploy stages to remote machines
    std::cout << "\nDeploying stages to remote endpoints..." << std::endl;
    for (const auto &ep : endpoints) {
      std::cout << "  ./network_worker " << ep.port << " &" << std::endl;
    }
    std::cout << std::endl;

    // Boilerplate to wait for workers to start
    std::cout << "Waiting 2 seconds for workers to start..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    if (!coordinator.deploy_stages()) {
      std::cerr << "Failed to deploy stages. Make sure workers are running."
                << std::endl;
      return 1;
    }

    // Start the pipeline
    std::cout << "\nStarting distributed pipeline..." << std::endl;
    coordinator.start();

    data_loading::MNISTDataLoader<float> train_loader, test_loader;

    // Validate data loading
    if (!train_loader.load_data("./data/mnist/train.csv")) {
      std::cerr << "Failed to load training data!" << std::endl;
      return -1;
    }

    if (!test_loader.load_data("./data/mnist/test.csv")) {
      std::cerr << "Failed to load test data!" << std::endl;
      return -1;
    }

    Tensor<float> batch_data, batch_labels;

    float loss = 0.0f, avg_accuracy = 0.0f;

    auto loss_function = tnn::LossFactory<float>::create("crossentropy");

    size_t batch_index = 0;

    while (train_loader.get_batch(32, batch_data, batch_labels)) {
      std::vector<Tensor<float>> micro_batches =
          batch_data.split(mnist_constants::NUM_MICROBATCHES);

      std::vector<Tensor<float>> micro_batch_labels =
          batch_labels.split(mnist_constants::NUM_MICROBATCHES);

      auto forward_start = std::chrono::high_resolution_clock::now();

      // Process a batch of data
      for (int i = 0; i < micro_batches.size(); ++i) {
        coordinator.forward(micro_batches[i], i);
      }

      coordinator.join(1);

      auto forward_end = std::chrono::high_resolution_clock::now();
      auto forward_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(forward_end -
                                                                forward_start);
      auto compute_loss_start = std::chrono::high_resolution_clock::now();

      std::vector<tpipeline::Message<float>> all_messages =
          coordinator.get_task_messages();

      if (all_messages.size() != mnist_constants::NUM_MICROBATCHES) {
        throw std::runtime_error(
            "Unexpected number of messages: " +
            std::to_string(all_messages.size()) +
            ", expected: " + std::to_string(mnist_constants::NUM_MICROBATCHES));
      }

      // Extract tasks from messages
      std::vector<tpipeline::Task<float>> forward_tasks;
      for (const auto &message : all_messages) {
        if (message.is_task_message()) {
          forward_tasks.push_back(message.task.value());
        }
      }

      std::vector<tpipeline::Task<float>> backward_tasks;
      for (auto &task : forward_tasks) {
        // Compute loss for each microbatch
        loss = loss_function->compute_loss(
            task.data, micro_batch_labels[task.micro_batch_id]);
        avg_accuracy = utils::compute_class_accuracy<float>(
            task.data, micro_batch_labels[task.micro_batch_id]);

        Tensor<float> gradient = loss_function->compute_gradient(
            task.data, micro_batch_labels[task.micro_batch_id]);

        // Create backward task
        tpipeline::Task<float> backward_task{tpipeline::TaskType::BACKWARD,
                                             gradient, task.micro_batch_id};

        backward_tasks.push_back(backward_task);
      }

      auto compute_loss_end = std::chrono::high_resolution_clock::now();
      auto compute_loss_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              compute_loss_end - compute_loss_start);

      auto backward_start = std::chrono::high_resolution_clock::now();

      // Backward pass
      for (const auto &task : backward_tasks) {
        coordinator.backward(task.data, task.micro_batch_id);
      }

      coordinator.join(0); // join backward tasks

      coordinator.get_task_messages(); // clear task messages

      coordinator.update_parameters();

      auto backward_end = std::chrono::high_resolution_clock::now();
      auto backward_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(backward_end -
                                                                backward_start);

      if (batch_index % mnist_constants::PROGRESS_PRINT_INTERVAL == 0) {
        std::cout << "Forward pass completed in " << forward_duration.count()
                  << " ms" << std::endl;
        std::cout << "Loss computation completed in "
                  << compute_loss_duration.count() << " ms" << std::endl;
        std::cout << "Backward pass completed in " << backward_duration.count() 
                  << " ms" << std::endl;
        std::cout << "Batch " << batch_index << "/"
                  << train_loader.size() / train_loader.get_batch_size()
                  << " - Loss: " << loss << ", Accuracy: "
                  << avg_accuracy * 100.0f << "%" << std::endl;
        // Print profiling information for all stages
        coordinator.print_profiling_on_all_stages();
      }
      ++batch_index;
    }
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
