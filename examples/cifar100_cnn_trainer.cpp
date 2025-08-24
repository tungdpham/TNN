#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "nn/layers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include "nn/optimizers.hpp"
#include "nn/loss.hpp"
#include "utils/cifar100_data_loader.hpp"
#include "utils/ops.hpp"
#include "utils/train.hpp"

namespace cifar100_constants {
  constexpr float EPSILON = 1e-15f;
  constexpr int PROGRESS_PRINT_INTERVAL = 50;
  constexpr int EPOCHS = 50; // Increased epochs for better convergence
  constexpr size_t BATCH_SIZE = 64; // Increased batch size for better stability
  constexpr int LR_DECAY_INTERVAL = 15; // Learning rate decay interval
  constexpr float LR_DECAY_FACTOR = 0.5f; // More aggressive decay
  constexpr float LR_INITIAL = 0.001f; // Lower initial learning rate for Adam
} // namespace cifar100_constants

int main() {
  try {
    std::cout << "CIFAR-100 CNN Tensor<float> Neural Network Training" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Load data
    data_loading::CIFAR100DataLoader<float> train_loader, test_loader;

    if (!train_loader.load_data("./data/cifar-100-binary/train.bin")) {
      return -1;
    }

    if (!test_loader.load_data("./data/cifar-100-binary/test.bin")) {
      return -1;
    }

    // Print dataset statistics
    std::cout << "\nDataset Information:" << std::endl;
    train_loader.print_data_stats();
    
    std::cout << "Successfully loaded training data: " << train_loader.size()
              << " samples" << std::endl;
    std::cout << "Successfully loaded test data: " << test_loader.size()
              << " samples" << std::endl;
    
    // Create CNN model architecture for CIFAR-100
    std::cout << "\nBuilding CNN model architecture for CIFAR-100..."
              << std::endl;

    auto model =
        tnn::SequentialBuilder<float>("cifar100_cnn_classifier")
            // Input: 3x32x32
            .input({3, 32, 32})
            .conv2d(32, 3, 3, 1, 1, 0, 0, "relu", true, "conv1") // Conv Layer 1 - Output: 32x30x30
            .conv2d(64, 3, 3, 1, 1, 0, 0, "relu", true, "conv2") // Conv Layer 2 - Output: 64x28x28
            .conv2d(128, 5, 5, 1, 1, 0, 0, "relu", true, "conv2_1") // Conv Layer 2_1 - Output: 128x24x24
            .maxpool2d(2, 2, 2, 2, 0, 0, "pool1")              // Max Pooling - Output: 128x12x12
            .conv2d(256, 3, 3, 1, 1, 0, 0, "relu", true, "conv3") // Conv Layer 3 - Output: 256x10x10
            .conv2d(256, 3, 3, 1, 1, 0, 0, "relu", true, "conv4") // Conv Layer 4 - Output: 256x8x8
            .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")              // Max Pooling - Output: 256x4x4
            .flatten("flatten")
            .dense(512, "relu", true, "fc1")      // Fully Connected Layer 1 - Auto-inferred input features
            .batchnorm(1e-5f, 0.1f, true, "bn1")       // Batch Norm
            .dense(100, "linear", true, "output")         // Output Layer for 100 classes
            .build();

    // Set optimizer for the model
    auto optimizer = std::make_unique<tnn::Adam<float>>(
        cifar100_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
    model.set_optimizer(std::move(optimizer));

    // Set loss function for the model
    auto loss_function =
        tnn::LossFactory<float>::create_crossentropy(cifar100_constants::EPSILON);
    model.set_loss_function(std::move(loss_function));

    model.enable_profiling(true); // Enable profiling for performance analysis
    // Print model summary
    model.print_summary(std::vector<size_t>{
        cifar100_constants::BATCH_SIZE, 3, 32, 32}); // Show summary with batch input

    // Train the CNN model with appropriate hyperparameters
    std::cout << "\nStarting CIFAR-100 CNN training..." << std::endl;
    train_cnn_model(model, train_loader, test_loader,
                    cifar100_constants::EPOCHS,  // epochs
                    cifar100_constants::BATCH_SIZE, // batch_size
                    cifar100_constants::LR_DECAY_FACTOR, // lr_decay_factor
                    cifar100_constants::PROGRESS_PRINT_INTERVAL // progress_print_interval
    );

    std::cout
        << "\nCIFAR-100 CNN Tensor<float> model training completed successfully!"
        << std::endl;

    // Save model with error handling
    try {
      model.save_to_file("model_snapshots/cifar100_cnn_model");
      std::cout << "Model saved to: model_snapshots/cifar100_cnn_model"
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
