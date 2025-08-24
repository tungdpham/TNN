#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include "utils/cifar10_data_loader.hpp"
#include "utils/ops.hpp"
#include "utils/train.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <random>
#include <sstream>
#include <vector>

namespace cifar10_constants {
constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 100;
constexpr int EPOCHS = 20;               // Number of epochs for training
constexpr size_t BATCH_SIZE = 32;        // Batch size for training
constexpr int LR_DECAY_INTERVAL = 5;    // Learning rate decay interval
constexpr float LR_DECAY_FACTOR = 0.85f; // Learning rate decay factor
constexpr float LR_INITIAL = 0.001f;     // Initial learning rate for training
} // namespace cifar10_constants

int main() {
  try {
#ifdef _OPENMP
    // Set OpenMP configuration for optimal performance
    const int num_threads = omp_get_max_threads();
    omp_set_num_threads(
        std::min(num_threads, 32)); // Limit threads to avoid overhead
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
    std::cout << "CIFAR-10 CNN Tensor<float> Neural Network Training"
              << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Load data
    data_loading::CIFAR10DataLoader<float> train_loader, test_loader;

    std::vector<std::string> train_files;
    for (int i = 1; i <= 5; ++i) {
      train_files.push_back("./data/cifar-10-batches-bin/data_batch_" +
                            std::to_string(i) + ".bin");
    }

    if (!train_loader.load_multiple_files(train_files)) {
      return -1;
    }

    if (!test_loader.load_multiple_files(
            {"./data/cifar-10-batches-bin/test_batch.bin"})) {
      return -1;
    }

    std::cout << "Successfully loaded training data: " << train_loader.size()
              << " samples" << std::endl;
    std::cout << "Successfully loaded test data: " << test_loader.size()
              << " samples" << std::endl;

    // Create CNN model architecture for CIFAR-10
    std::cout << "\nBuilding CNN model architecture for CIFAR-10..."
              << std::endl;

    auto model =
        tnn::SequentialBuilder<float>("cifar10_cnn_classifier")
            // Input: 3x32x32 (channels, height, width)
            .input({3, 32, 32})
            .conv2d(32, 3, 3, 1, 1, 1, 1, "linear", true,
                    "conv1")                         // 3x32x32 -> 32x32x32
            // .batchnorm(0.9f, 1e-5f, true, "bn1") // BatchNorm layer
            .activation("relu", "relu1")
            .maxpool2d(2, 2, 2, 2, 0, 0, "pool1") // 32x32x32 -> 32x16x16
            .conv2d(64, 3, 3, 1, 1, 1, 1, "relu", true,
                    "conv2")                         // 32x16x16 -> 64x16x16
            // .batchnorm(0.9f, 1e-5f, true, "bn2") // BatchNorm layer
            .activation("relu", "relu2")
            .maxpool2d(2, 2, 2, 2, 0, 0, "pool2") // 64x16x16 -> 64x8x8
            .conv2d(128, 3, 3, 1, 1, 1, 1, "relu", true,
                    "conv3")                          // 64x8x8 -> 128x8x8
            .maxpool2d(2, 2, 2, 2, 0, 0, "pool3") // 128x8x8 -> 128x4x4
            .flatten("flatten")
            .dense(512, "relu", true, "fc0") // Auto-inferred input features -> 512
            .dense(10, "linear", true, "fc1") // Auto-inferred input features -> 10
            .build();

    // Set optimizer for the model
    auto optimizer = std::make_unique<tnn::SGD<float>>(
        cifar10_constants::LR_INITIAL, 0.9f);
    model.set_optimizer(std::move(optimizer));

    // Set loss function for the model
    auto loss_function =
        tnn::LossFactory<float>::create_crossentropy(cifar10_constants::EPSILON);
    model.set_loss_function(std::move(loss_function));

    model.enable_profiling(true); // Enable profiling for performance analysis

    // Print model summary
    model.print_summary(
        std::vector<size_t>{cifar10_constants::BATCH_SIZE, 3, 32,
                            32}); // Show summary with single image input

    // Train the CNN model with appropriate hyperparameters
    std::cout << "\nStarting CIFAR-10 CNN training..." << std::endl;
    train_cnn_model(model, train_loader, test_loader,
                    cifar10_constants::EPOCHS,         // epochs
                    cifar10_constants::BATCH_SIZE,     // batch_size
                    cifar10_constants::LR_DECAY_FACTOR, // lr_decay_factor
                    cifar10_constants::PROGRESS_PRINT_INTERVAL // progress_print_interval
    );

    std::cout
        << "\nCIFAR-10 CNN Tensor<float> model training completed successfully!"
        << std::endl;

    // Save model with error handling
    try {
      model.save_to_file("model_snapshots/cifar10_cnn_model");
      std::cout << "Model saved to: model_snapshots/cifar10_cnn_model"
                << std::endl;
    } catch (const std::exception &save_error) {
      std::cerr << "Warning: Failed to save model: " << save_error.what()
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
