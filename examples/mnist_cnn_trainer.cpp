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
#include "utils/train.hpp"

// Constants for MNIST training (additional to those in utils/mnist_data_loader.hpp)
namespace mnist_constants {

constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 100;
constexpr int EPOCHS = 3;
constexpr size_t BATCH_SIZE = 64; // Good balance between memory and convergence
constexpr int LR_DECAY_INTERVAL = 2;
constexpr float LR_DECAY_FACTOR = 0.8f;
constexpr float LR_INITIAL = 0.01f; // Initial learning rate for training

// Use constants from utils/mnist_data_loader.hpp: IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES

} // namespace mnist_constants

int main() {
  try {
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

    // Create optimized CNN model architecture with automatic shape inference
    std::cout << "\nBuilding CNN model architecture with automatic shape inference..." << std::endl;

    auto model = tnn::SequentialBuilder<float>("optimized_mnist_cnn_classifier")
                     // Set input shape first: 1 channel, 28x28 image (without batch dimension)
                     .input({1, ::mnist_constants::IMAGE_HEIGHT, ::mnist_constants::IMAGE_WIDTH})
                     
                     // C1: First convolution layer - 5x5 kernel, stride 1, ELU activation
                     // Input: 1x28x28 → Output: 8x24x24 (28-5+1=24) - in_channels=1 automatically inferred
                     .conv2d(8, 5, 5, 1, 1, 0, 0, "elu", true, "conv1")
                     
                     // P1: Max pooling layer - 3x3 blocks, stride 3
                     // Input: 8x24x24 → Output: 8x8x8 (24/3=8)
                     .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")

                     // C2: Inception-style 1x1 convolution for dimensionality reduction
                     // Input: 8x8x8 → Output: 16x8x8 - in_channels=8 automatically inferred
                     .conv2d(16, 1, 1, 1, 1, 0, 0, "elu", true, "conv2_1x1")

                     // C3: Second convolution layer - 5x5 kernel, stride 1, ELU activation
                     // Input: 16x8x8 → Output: 48x4x4 (8-5+1=4) - in_channels=16 automatically inferred
                     .conv2d(48, 5, 5, 1, 1, 0, 0, "elu", true, "conv3")

                     // P2: Second max pooling layer - 2x2 blocks, stride 2
                     // Input: 48x4x4 → Output: 48x2x2 (4/2=2)
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")

                     // Flatten layer to prepare for dense layer
                     // Input: 48x2x2 → Output: 192 features
                     .flatten("flatten")

                     // FC1: Fully connected output layer
                     // Input: 192 features → Output: 10 classes - input_features=192 automatically inferred
                     .dense(::mnist_constants::NUM_CLASSES, "linear", true, "output")
                     .build();

    // Set optimizer for the model
    auto optimizer = std::make_unique<tnn::Adam<float>>(
        mnist_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
    model.set_optimizer(std::move(optimizer));

    // Set loss function for the model
    auto loss_function =
        tnn::LossFactory<float>::create_crossentropy(mnist_constants::EPSILON);
    model.set_loss_function(std::move(loss_function));

    // Enable profiling for performance analysis
    model.enable_profiling(true);

    // Print model summary with optimized batch size
    std::cout << "\nModel Architecture Summary:" << std::endl;
    model.print_summary(std::vector<size_t>{mnist_constants::BATCH_SIZE, 1,
                                            ::mnist_constants::IMAGE_HEIGHT,
                                            ::mnist_constants::IMAGE_WIDTH});

    train_cnn_model(model, train_loader, test_loader, mnist_constants::EPOCHS,
                    mnist_constants::BATCH_SIZE, mnist_constants::LR_DECAY_FACTOR,
                    mnist_constants::PROGRESS_PRINT_INTERVAL);

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
