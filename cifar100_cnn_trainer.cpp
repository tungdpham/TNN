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

namespace cifar100_constants {
  constexpr float EPSILON = 1e-15f;
  constexpr int PROGRESS_PRINT_INTERVAL = 50;
  constexpr int EPOCHS = 20; // Number of epochs for training
  constexpr size_t BATCH_SIZE = 32; // Batch size for training
  constexpr int LR_DECAY_INTERVAL = 10; // Learning rate decay interval
  constexpr float LR_DECAY_FACTOR = 0.85f; // Learning rate decay factor
  constexpr float LR_INITIAL = 0.01f; // Initial learning rate for training
} // namespace cifar100_constants



// Training function for CNN tensor model
void train_cnn_model(tnn::Sequential<float> &model,
                     data_loading::CIFAR100DataLoader<float> &train_loader,
                     data_loading::CIFAR100DataLoader<float> &test_loader, int epochs = 10,
                     int batch_size = 32, float learning_rate = 0.001) {
  tnn::SGD<float> optimizer(learning_rate, 0.9);

  auto loss_function = tnn::LossFactory<float>::create_crossentropy(cifar100_constants::EPSILON);

  std::cout << "Starting CNN tensor model training..." << std::endl;
  std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size
            << ", Learning rate: " << learning_rate << std::endl;
  std::cout << std::string(70, '=') << std::endl;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    auto epoch_start = std::chrono::high_resolution_clock::now();

    // Training phase
    model.train();
    train_loader.shuffle();
    train_loader.reset();

    float total_loss = 0.0;
    float total_accuracy = 0.0;
    int num_batches = 0;

    Tensor<float> batch_data, batch_labels;
    while (train_loader.get_batch(batch_size, batch_data, batch_labels)) {
      // Forward pass
      Tensor<float> predictions = model.forward(batch_data);
      utils::apply_softmax<float>(predictions);

      // Compute loss and accuracy
      float loss =
          loss_function->compute_loss(predictions, batch_labels);
      float accuracy = utils::compute_class_accuracy<float>(predictions, batch_labels);

      total_loss += loss;
      total_accuracy += accuracy;
      num_batches++;

      // Backward pass
      Tensor<float> loss_gradient =
          loss_function->compute_gradient(predictions, batch_labels);
      model.backward(loss_gradient);

      // Update parameters
      auto params = model.parameters();
      auto grads = model.gradients();
      optimizer.update(params, grads);

      // Print progress every 10 batches (more frequent for CNN)
      if (num_batches % 100 == 0) {
        model.print_profiling_summary();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Batch "
                  << num_batches << ", Loss: " << std::fixed
                  << std::setprecision(4) << loss
                  << ", Acc: " << std::setprecision(3) << accuracy * 100 << "%"
                  << std::endl;
      }
    }

    float avg_train_loss = total_loss / num_batches;
    float avg_train_accuracy = total_accuracy / num_batches;

    // Validation phase
    model.eval();
    test_loader.reset();

    float val_loss = 0.0;
    float val_accuracy = 0.0;
    int val_batches = 0;

    while (test_loader.get_batch(batch_size, batch_data, batch_labels)) {
      Tensor<float> predictions = model.forward(batch_data);

      val_loss +=
          loss_function->compute_loss(predictions, batch_labels);
      val_accuracy += utils::compute_class_accuracy<float>(predictions, batch_labels);
      val_batches++;
    }

    float avg_val_loss = val_loss / val_batches;
    float avg_val_accuracy = val_accuracy / val_batches;

    auto epoch_end = std::chrono::high_resolution_clock::now();
    auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(
        epoch_end - epoch_start);

    // Print epoch summary
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed in "
              << epoch_duration.count() << "s" << std::endl;
    std::cout << "Training   - Loss: " << std::fixed << std::setprecision(4)
              << avg_train_loss << ", Accuracy: " << std::setprecision(2)
              << avg_train_accuracy * 100 << "%" << std::endl;
    std::cout << "Validation - Loss: " << std::fixed << std::setprecision(4)
              << avg_val_loss << ", Accuracy: " << std::setprecision(2)
              << avg_val_accuracy * 100 << "%" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Learning rate decay (more aggressive for CNN)
    if ((epoch + 1) % 2 == 0) {
      std::cout << "Current learning rate: " << optimizer.get_learning_rate()
                << std::endl;
      float new_lr = 1.0f * optimizer.get_learning_rate() * 0.8f;
      optimizer.set_learning_rate(new_lr);
      std::cout << "Decayed learning rate to: " << new_lr << std::endl;
    }
  }
}

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
    
    // Prepare batches for efficient training
    std::cout << "\nPreparing training batches..." << std::endl;
    train_loader.prepare_batches(cifar100_constants::BATCH_SIZE);
    test_loader.prepare_batches(cifar100_constants::BATCH_SIZE);

    // Create CNN model architecture for CIFAR-100
    std::cout << "\nBuilding CNN model architecture for CIFAR-100..."
              << std::endl;

    auto model =
        tnn::SequentialBuilder<float>("cifar100_cnn_classifier")
            // Input: 32x32x3
            .conv2d(3, 32, 3, 3, 1, 1, 0, 0, "relu", true, "conv1") // 32x32x32
            .maxpool2d(3, 3, 3, 3, 0, 0, "pool1") // 10x10x16
            .conv2d(32, 64, 3, 3, 1, 1, 0, 0, "relu", true, "conv2") // 10x10x64
            .maxpool2d(4, 4, 4, 4, 0, 0, "pool2") // 2x2x64
            .dense(64 * 2 * 2, 512, "relu", true, "fc1") // Flatten to 512
            .dense(512, 100, "linear", true, "output") // Output layer with 100 classes
            // .activation("softmax", "softmax_output") // Softmax activation
            .build();

    model.enable_profiling(true); // Enable profiling for performance analysis
    // Print model summary
    model.print_summary(std::vector<size_t>{
        64, 3, 32, 32}); // Show summary with single image input

    // Train the CNN model with appropriate hyperparameters
    std::cout << "\nStarting CIFAR-100 CNN training..." << std::endl;
    train_cnn_model(model, train_loader, test_loader,
                    cifar100_constants::EPOCHS,  // epochs
                    cifar100_constants::BATCH_SIZE, // batch_size
                    cifar100_constants::LR_INITIAL // learning_rate
    );

    std::cout
        << "\nCIFAR-100 CNN Tensor<float> model training completed successfully!"
        << std::endl;

    model.save_to_file("model_snapshots/cifar100_cnn_model");
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
