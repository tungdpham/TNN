#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "pipeline/message.hpp"
#include "pipeline/pipeline_coordinator.hpp"
#include "tensor/tensor.hpp"
#include "utils/mnist_data_loader.hpp"
#include "utils/ops.hpp"

namespace mnist_constants {
constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 100;
constexpr int EPOCHS = 20;
constexpr size_t BATCH_SIZE =
    128; // Good balance between memory and convergence
constexpr int LR_DECAY_INTERVAL = 2;
constexpr float LR_DECAY_FACTOR = 0.8f;
constexpr float LR_INITIAL = 0.01f; // Initial learning rate for training
constexpr int NUM_MICROBATCHES =
    1; // Number of microbatches for pipeline processing
} // namespace mnist_constants

signed main() {
  // Load MNIST dataset
  data_loading::MNISTDataLoader<float> train_loader, test_loader;
  if (!train_loader.load_data("./data/mnist/train.csv")) {
    std::cerr << "Failed to load training data!" << std::endl;
    return -1;
  }

  if (!test_loader.load_data("./data/mnist/test.csv")) {
    std::cerr << "Failed to load test data!" << std::endl;
    return -1;
  }

  omp_set_num_threads(8);
  // Create a sequential model using the builder pattern
  auto model = tnn::SequentialBuilder<float>("mnist_cnn_classifier")
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
  auto pipeline_coordinator = tpipeline::InProcessPipelineCoordinator<float>(
      model, 1, mnist_constants::NUM_MICROBATCHES);
  // Get the stages from the coordinator
  auto stages = pipeline_coordinator.get_stages();

  for (auto &stage : stages) {
    stage->start();
  }

  // Prepare the training data loader
  train_loader.prepare_batches(mnist_constants::BATCH_SIZE);
  test_loader.prepare_batches(mnist_constants::BATCH_SIZE);

  Tensor<float> batch_data;
  Tensor<float> batch_labels;
  int batch_index = 0;
  printf("Starting training loop...\n");
  // pipeline_coordinator.start();

  auto epoch_start = std::chrono::high_resolution_clock::now();
  while (train_loader.get_next_batch(batch_data, batch_labels)) {
    auto forward_start = std::chrono::high_resolution_clock::now();
    // Process a batch of data
    pipeline_coordinator.forward(batch_data);

    pipeline_coordinator.join(1);

    auto forward_end = std::chrono::high_resolution_clock::now();
    auto forward_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(forward_end -
                                                              forward_start);
    auto compute_loss_start = std::chrono::high_resolution_clock::now();

    std::vector<tpipeline::Message<float>> all_messages =
        pipeline_coordinator.get_task_messages();
    // printf("Total messages processed: %zu\n", all_messages.size());

    if (all_messages.size() != mnist_constants::NUM_MICROBATCHES) {
      throw std::runtime_error(
          "Unexpected number of messages: " +
          std::to_string(all_messages.size()) +
          ", expected: " + std::to_string(mnist_constants::NUM_MICROBATCHES));
    }

    // Extract tasks from messages
    std::vector<tpipeline::Task<float>> all_tasks;
    for (const auto &message : all_messages) {
      if (message.is_task_message()) {
        all_tasks.push_back(message.task.value());
      }
    }

    std::vector<Tensor<float>> outputs;
    sort(all_tasks.begin(), all_tasks.end(),
         [](const tpipeline::Task<float> &a, const tpipeline::Task<float> &b) {
           return a.micro_batch_id < b.micro_batch_id;
         });
    for (const auto &task : all_tasks) {
      if (task.type == tpipeline::TaskType::FORWARD) {
        outputs.push_back(task.data);
      } else {
        throw new std::runtime_error(
            "Unexpected task type in forward processing: " +
            std::to_string(static_cast<int>(task.type)));
      }
    }

    // Compute loss and accuracy using the output tensors
    // Apply softmax to outputs
    for (auto &output : outputs) {
      utils::apply_softmax<float>(output);
    }

    // Compute loss and accuracy using the output tensors
    auto loss_function = tnn::LossFactory<float>::create("crossentropy");
    float total_loss = 0.0f;

    std::vector<Tensor<float>> micro_batch_labels =
        batch_labels.split(mnist_constants::NUM_MICROBATCHES);

    for (int i = 0; i < outputs.size(); ++i) {
      total_loss +=
          loss_function->compute_loss(outputs[i], micro_batch_labels[i]);
    }

    float accuracy = 0;

    for (int i = 0; i < outputs.size(); ++i) {
      accuracy += utils::compute_class_accuracy<float>(outputs[i], micro_batch_labels[i]);
    }

    accuracy /= outputs.size();

    // Compute gradient with respect to the loss
    std::vector<Tensor<float>> gradients;
    for (int i = 0; i < micro_batch_labels.size(); ++i) {
      gradients.push_back(
          loss_function->compute_gradient(outputs[i], micro_batch_labels[i]));
    }
    auto compute_loss_end = std::chrono::high_resolution_clock::now();
    auto compute_loss_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            compute_loss_end - compute_loss_start);

    auto backward_start = std::chrono::high_resolution_clock::now();
    // Backward pass
    pipeline_coordinator.backward(gradients);

    pipeline_coordinator.join(0); // join backward tasks

    pipeline_coordinator.get_task_messages(); // clear task messages

    pipeline_coordinator.update_parameters();

    auto backward_end = std::chrono::high_resolution_clock::now();
    auto backward_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(backward_end -
                                                              backward_start);

    if (batch_index % mnist_constants::PROGRESS_PRINT_INTERVAL == 0) {
      printf("Forward pass completed in %ld ms\n", forward_duration.count());
      printf("Loss computation completed in %ld ms\n",
             compute_loss_duration.count());
      printf("Backward pass completed in %ld ms\n", backward_duration.count());
      printf("Batch %d/%zu - Loss: %.4f, Accuracy: %.2f%%\n", batch_index,
             train_loader.size() / train_loader.get_batch_size(), total_loss,
             accuracy * 100.0f);
      pipeline_coordinator.print_profiling_on_all_stages();
    }
    ++batch_index;
  }
  auto epoch_end = std::chrono::high_resolution_clock::now();
  auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      epoch_end - epoch_start);
  printf("Epoch completed in %ld ms\n", epoch_duration.count());

  printf("Program stopped successfully.\n");
  return 0;
}