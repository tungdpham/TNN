#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string_view>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <vector>

#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include "utils/mnist_data_loader.hpp"
#include "utils/ops.hpp"

// Constants for MNIST dataset
namespace mnist_constants {

constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 100;
constexpr int EPOCHS = 3;
constexpr size_t BATCH_SIZE = 64; // Good balance between memory and convergence
constexpr int LR_DECAY_INTERVAL = 2;
constexpr float LR_DECAY_FACTOR = 0.8f;
constexpr float LR_INITIAL = 0.01f; // Initial learning rate for training

} // namespace mnist_constants

void train_cnn_model(tnn::Sequential<float> &model,
                     data_loading::MNISTDataLoader<float> &train_loader,
                     data_loading::MNISTDataLoader<float> &test_loader,
                     int epochs = 10, int batch_size = 32,
                     float learning_rate = 0.001f) {
  tnn::Adam<float> optimizer(learning_rate, 0.9f, 0.999f, 1e-8f);

  // Create loss function using the new base class approach
  auto loss_function =
      tnn::LossFactory<float>::create_crossentropy(mnist_constants::EPSILON);

  Tensor<float> batch_data, batch_labels, predictions;

  // Pre-compute batches for training and validation data
  std::cout << "\nPreparing training batches..." << std::endl;
  train_loader.prepare_batches(batch_size);

  std::cout << "Preparing validation batches..." << std::endl;
  test_loader.prepare_batches(batch_size);

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  std::cout << "Validation batches: " << test_loader.num_batches() << std::endl;
  std::cout << std::string(70, '=') << std::endl;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    const auto epoch_start = std::chrono::high_resolution_clock::now();

    // Training phase - shuffle data and re-prepare batches for each epoch
    model.train();
    train_loader.shuffle();
    train_loader.prepare_batches(batch_size); // Re-prepare after shuffle
    train_loader.reset();

    double total_loss = 0.0; // Use double for better numerical precision
    double total_accuracy = 0.0;
    int num_batches = 0;
    std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
    // Use fast batch iteration with pre-computed batches
    while (train_loader.get_next_batch(batch_data, batch_labels)) {
      ++num_batches;

      // Forward pass
      predictions = model.forward(batch_data);
      utils::apply_softmax<float>(predictions);

      // Compute loss and accuracy
      const float loss = loss_function->compute_loss(predictions, batch_labels);
      const float accuracy =
          utils::compute_class_accuracy<float>(predictions, batch_labels);

      total_loss += loss;
      total_accuracy += accuracy;

      // Backward pass
      const Tensor<float> loss_gradient =
          loss_function->compute_gradient(predictions, batch_labels);
      model.backward(loss_gradient);

      // Update parameters
      model.update_parameters();

      // Print progress at intervals
      if (num_batches % mnist_constants::PROGRESS_PRINT_INTERVAL == 0) {
        model.print_profiling_summary();
        std::cout << "Batch ID: " << num_batches
                  << ", Batch's Loss: " << std::fixed << std::setprecision(4)
                  << loss << ", Batch's Accuracy: " << std::setprecision(2)
                  << accuracy * 100.0f << "%" << std::endl;
      }
      model.clear_profiling_data();
    }

    const float avg_train_loss = static_cast<float>(total_loss / num_batches);
    const float avg_train_accuracy =
        static_cast<float>(total_accuracy / num_batches);

    // Optimized validation phase - use pre-computed batches
    model.eval();
    test_loader.reset();

    double val_loss = 0.0;
    double val_accuracy = 0.0;
    int val_batches = 0;

    // Use fast batch iteration with pre-computed validation batches
    while (test_loader.get_next_batch(batch_data, batch_labels)) {
      predictions = model.forward(batch_data);
      utils::apply_softmax<float>(predictions);

      val_loss += loss_function->compute_loss(predictions, batch_labels);
      val_accuracy +=
          utils::compute_class_accuracy<float>(predictions, batch_labels);
      ++val_batches;
    }

    const float avg_val_loss = static_cast<float>(val_loss / val_batches);
    const float avg_val_accuracy =
        static_cast<float>(val_accuracy / val_batches);

    const auto epoch_end = std::chrono::high_resolution_clock::now();
    const auto epoch_duration =
        std::chrono::duration_cast<std::chrono::seconds>(epoch_end -
                                                         epoch_start);

    // Print epoch summary with improved formatting
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed in "
              << epoch_duration.count() << "s" << std::endl;
    std::cout << "Training   - Loss: " << std::fixed << std::setprecision(4)
              << avg_train_loss << ", Accuracy: " << std::setprecision(2)
              << avg_train_accuracy * 100.0f << "%" << std::endl;
    std::cout << "Validation - Loss: " << std::fixed << std::setprecision(4)
              << avg_val_loss << ", Accuracy: " << std::setprecision(2)
              << avg_val_accuracy * 100.0f << "%" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Learning rate decay
    if ((epoch + 1) % mnist_constants::LR_DECAY_INTERVAL == 0) {
      const float current_lr = optimizer.get_learning_rate();
      const float new_lr = current_lr * mnist_constants::LR_DECAY_FACTOR;
      optimizer.set_learning_rate(new_lr);
      std::cout << "Learning rate decayed: " << std::fixed
                << std::setprecision(6) << current_lr << " → " << new_lr
                << std::endl;
    }
  }
}

int main() {
  try {
    std::cout << "Optimized MNIST CNN Tensor<float> Neural Network Training"
              << std::endl;
    std::cout << std::string(60, '=') << std::endl;

#ifdef _OPENMP
    // Set OpenMP configuration for optimal performance
    const int num_threads = omp_get_max_threads();
    omp_set_num_threads(
        std::min(num_threads, 8)); // Limit threads to avoid overhead
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads"
              << std::endl;
#endif

#ifdef USE_TBB
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, 8);
    std::cout << "tbb::global_control::active_value(max_allowed_parallelism): "
              << tbb::global_control::active_value(
                     tbb::global_control::max_allowed_parallelism)
              << "\n";
#endif

    // Initialize data loaders with improved error handling
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

    std::cout << "Successfully loaded training data: " << train_loader.size()
              << " samples" << std::endl;
    std::cout << "Successfully loaded test data: " << test_loader.size()
              << " samples" << std::endl;

    // Create optimized CNN model architecture
    std::cout << "\nBuilding CNN model architecture..." << std::endl;

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

                     // C3: Second convolution layer - 5x5 kernel, stride 1,
                     // ReLU activation Input: 16x8x8 → Output: 48x4x4 (8-5+1=4)
                     .conv2d(16, 48, 5, 5, 1, 1, 0, 0, "sigmoid", true, "conv3")

                     // P2: Second max pooling layer - 2x2 blocks, stride 2
                     // Input: 48x4x4 → Output: 48x2x2 (4/2=2)
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")

                     // FC1: Fully connected output layer
                     // Input: 48x2x2 = 192 features → Output: 10 classes
                     .dense(48 * 2 * 2, mnist_constants::NUM_CLASSES, "linear",
                            true, "output")
                     .build();
    // Set optimizer for the model
    auto optimizer = std::make_unique<tnn::Adam<float>>(
        mnist_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
    model.set_optimizer(std::move(optimizer));

    // Set loss function for the model
    auto loss_function =
        tnn::LossFactory<float>::create_crossentropy(mnist_constants::EPSILON);
    model.set_loss(std::move(loss_function));

    // Enable profiling for performance analysis
    model.enable_profiling(true);

    // Print model summary with optimized batch size
    std::cout << "\nModel Architecture Summary:" << std::endl;
    model.print_summary(std::vector<size_t>{mnist_constants::BATCH_SIZE, 1,
                                            mnist_constants::IMAGE_HEIGHT,
                                            mnist_constants::IMAGE_WIDTH});

    train_cnn_model(model, train_loader, test_loader, mnist_constants::EPOCHS,
                    mnist_constants::BATCH_SIZE, mnist_constants::LR_INITIAL);

    std::cout << "\nOptimized MNIST CNN model training completed successfully!"
              << std::endl;

    // Save model with error handling
    try {
      model.save_to_file("model_snapshots/mnist_cnn_model");
      std::cout << "Model saved to: model_snapshots/mnist_cnn_model"
                << std::endl;
    } catch (const std::exception &save_error) {
      std::cerr << "Warning: Failed to save model: " << save_error.what()
                << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error during training: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "Unknown error occurred during training!" << std::endl;
    return -1;
  }

  return 0;
}
