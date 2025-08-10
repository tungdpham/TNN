#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "pipeline_experimental/pipeline_coordinator.hpp"
#include "tensor/tensor.hpp"
#include "utils/mnist_data_loader.hpp"
using namespace tnn;
using namespace data_loading;

namespace mnist_constants {
constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 100;
constexpr int EPOCHS = 20;
constexpr size_t BATCH_SIZE = 128; // Good balance between memory and convergence
constexpr int LR_DECAY_INTERVAL = 2;
constexpr float LR_DECAY_FACTOR = 0.8f;
constexpr float LR_INITIAL = 0.01f; // Initial learning rate for training
} // namespace mnist_constants

void apply_tensor_softmax(Tensor<float> &tensor) {
  const size_t batch_size = tensor.shape()[0];
  const size_t num_classes = tensor.shape()[1];

#pragma omp parallel for if (batch_size > 16)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    float max_val = tensor(batch, 0, 0, 0);
    for (size_t j = 1; j < num_classes; ++j) {
      max_val = std::max(max_val, tensor(batch, j, 0, 0));
    }

    float sum = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      const float exp_val = std::exp(tensor(batch, j, 0, 0) - max_val);
      tensor(batch, j, 0, 0) = exp_val;
      sum += exp_val;
    }

    const float inv_sum = 1.0f / std::max(sum, mnist_constants::EPSILON);
    for (size_t j = 0; j < num_classes; ++j) {
      tensor(batch, j, 0, 0) *= inv_sum;
    }
  }
}

// Optimized accuracy calculation for tensors
float calculate_tensor_accuracy(const Tensor<float> &predictions,
                                const Tensor<float> &targets) {
  const size_t batch_size = predictions.shape()[0];
  const size_t num_classes = predictions.shape()[1];

  int total_correct = 0;

// Parallelize accuracy computation with reduction
#pragma omp parallel for reduction(+ : total_correct) if (batch_size > 16)
  for (size_t i = 0; i < batch_size; ++i) {
    // Find predicted class (argmax) - more efficient implementation
    int pred_class = 0;
    float max_pred = predictions(i, 0, 0, 0);
    for (size_t j = 1; j < num_classes; ++j) {
      const float pred_val = predictions(i, j, 0, 0);
      if (pred_val > max_pred) {
        max_pred = pred_val;
        pred_class = static_cast<int>(j);
      }
    }

    // Find true class - early termination when found
    int true_class = -1;
    for (size_t j = 0; j < num_classes; ++j) {
      if (targets(i, j, 0, 0) > 0.5f) {
        true_class = static_cast<int>(j);
        break;
      }
    }

    // printf("Predicted class: %d, True class: %d\n", pred_class, true_class);
    if (pred_class == true_class && true_class != -1) {
      total_correct++;
    }
  }

  return static_cast<float>(total_correct) / static_cast<float>(batch_size);
}

signed main() {
  // Load MNIST dataset
  MNISTDataLoader<float> train_loader, test_loader;
  if (!train_loader.load_data("./data/mnist/train.csv")) {
    std::cerr << "Failed to load training data!" << std::endl;
    return -1;
  }

  if (!test_loader.load_data("./data/mnist/test.csv")) {
    std::cerr << "Failed to load test data!" << std::endl;
    return -1;
  }

  omp_set_num_threads(2);
  // Create a sequential model using the builder pattern
  auto model = SequentialBuilder<float>("mnist_cnn_classifier")
                   // C1: First convolution layer - 5x5 kernel, stride 1, ReLU
                   // activation Input: 1x28x28 → Output: 8x24x24 (28-5+1=24)
                   .conv2d(1, 8, 5, 5, 1, 1, 0, 0, "relu", true, "conv1")
                   // .batchnorm(8, 1e-5, 0.1, true, "batchnorm1")
                   // .activation("relu", "relu1")
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

  auto pipeline_coordinator =
      tpipeline::InProcessPipelineCoordinator<float>(model, 2, 4);

  // Prepare the training data loader
  train_loader.prepare_batches(mnist_constants::BATCH_SIZE);
  test_loader.prepare_batches(mnist_constants::BATCH_SIZE);

  Tensor<float> batch_data;
  Tensor<float> batch_labels;
  int batch_index = 0;
  printf("Starting training loop...\n");
  while (train_loader.get_next_batch(batch_data, batch_labels)) {
    pipeline_coordinator.start();
    // Process a batch of data
    pipeline_coordinator.forward(batch_data, 4);

    pipeline_coordinator.join();

    std::vector<tpipeline::Task<float>> all_tasks =
        pipeline_coordinator.get_all_tasks();
    // printf("Total tasks processed: %zu\n", all_tasks.size());

    std::vector<Tensor<float>> outputs;
    sort(all_tasks.begin(), all_tasks.end(),
         [](const tpipeline::Task<float> &a, const tpipeline::Task<float> &b) {
           return a.micro_batch_id < b.micro_batch_id;
         });
    for (const auto &task : all_tasks) {
      if (task.type == tpipeline::TaskType::Forward) {
        outputs.push_back(task.data);
      } else {
      }
    }

    // Apply softmax to outputs
    for (auto &output : outputs) {
      apply_tensor_softmax(output);
    }

    // Compute loss and accuracy using the output tensors
    auto loss_function = tnn::LossFactory<float>::create("crossentropy");
    float total_loss = 0.0f;

    std::vector<Tensor<float>> micro_batch_labels = batch_labels.split(4);

    for (int i = 0; i < outputs.size(); ++i) {
      total_loss +=
          loss_function->compute_loss(outputs[i], micro_batch_labels[i]);
    }

    float accuracy = 0;

    for (int i = 0; i < outputs.size(); ++i) {
      accuracy += calculate_tensor_accuracy(outputs[i], micro_batch_labels[i]);
    }

    accuracy /= outputs.size();

    if (batch_index % mnist_constants::PROGRESS_PRINT_INTERVAL == 0) {
      printf("Batch %d/%zu - Loss: %.4f, Accuracy: %.2f%%\n", batch_index,
             train_loader.size() / train_loader.get_batch_size(), total_loss,
             accuracy * 100.0f);
    }

    // Compute gradient with respect to the loss
    std::vector<Tensor<float>> gradients;
    for (int i = 0; i < micro_batch_labels.size(); ++i) {
      gradients.push_back(
          loss_function->compute_gradient(outputs[i], micro_batch_labels[i]));
    }

    // Backward pass
    pipeline_coordinator.start();

    pipeline_coordinator.backward(gradients);

    pipeline_coordinator.join();

    pipeline_coordinator.update_params();

    ++batch_index;
  }

  printf("Program stopped successfully.\n");
  return 0;
}