#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <omp.h>  
#include "nn/layers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include "nn/optimizers.hpp"
#include "nn/loss.hpp"

namespace cifar10_constants {
  constexpr float EPSILON = 1e-15f;
  constexpr int PROGRESS_PRINT_INTERVAL = 50;
  constexpr int EPOCHS = 20; // Number of epochs for training
  constexpr size_t BATCH_SIZE = 32; // Batch size for training
  constexpr int LR_DECAY_INTERVAL = 10; // Learning rate decay interval
  constexpr float LR_DECAY_FACTOR = 0.85f; // Learning rate decay factor
  constexpr float LR_INITIAL = 0.01f; // Initial learning rate for training
} // namespace cifar_constants

// CIFAR-10 data loader
class CIFAR10DataLoader {
private:
  std::vector<std::vector<float>> data_;
  std::vector<int> labels_;
  size_t current_index_;
  const int num_classes = 10;

public:
  bool load_data(const std::vector<std::string> &filenames) {
    data_.clear();
    labels_.clear();

    for (const auto &filename : filenames) {
      std::ifstream file(filename, std::ios::binary);
      if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
      }

      // CIFAR-10 binary format: [1-byte label][3072 bytes of pixel data]
      const int record_size = 1 + 32 * 32 * 3;

      char buffer[record_size];
      while (file.read(buffer, record_size)) {
        labels_.push_back(static_cast<int>(buffer[0]));

        std::vector<float> image_data(32 * 32 * 3);
        for (int i = 0; i < 3072; ++i) {
          image_data[i] = static_cast<unsigned char>(buffer[i + 1]) / 255.0f;
        }
        data_.push_back(image_data);
      }
      std::cout << "Loaded " << data_.size() << " samples from " << filename
                << std::endl;
    }

    current_index_ = 0;
    return true;
  }

  void shuffle() {
    std::vector<size_t> indices(data_.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<std::vector<float>> shuffled_data(data_.size());
    std::vector<int> shuffled_labels(labels_.size());

    for (size_t i = 0; i < indices.size(); ++i) {
      shuffled_data[i] = data_[indices[i]];
      shuffled_labels[i] = labels_[indices[i]];
    }

    data_ = std::move(shuffled_data);
    labels_ = std::move(shuffled_labels);
    current_index_ = 0;
  }

  bool get_batch(int batch_size, Tensor<float> &batch_data,
                 Tensor<float> &batch_labels) {
    if (current_index_ >= data_.size()) {
      return false; // No more data
    }

    int actual_batch_size =
        std::min(batch_size, static_cast<int>(data_.size() - current_index_));

    // Create batch data tensor for CNN: (batch_size, channels=3, height=32, width=32)
    batch_data = Tensor<float>(
        std::vector<size_t>{static_cast<size_t>(actual_batch_size), 3, 32, 32});

    // Create batch labels tensor (batch_size, 10, 1, 1) - one-hot encoded
    batch_labels = Tensor<float>(
        std::vector<size_t>{static_cast<size_t>(actual_batch_size), (size_t)num_classes, 1, 1});
    batch_labels.fill(0.0);

    for (int i = 0; i < actual_batch_size; ++i) {
      // Copy pixel data and reshape to 3x32x32
      for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 32; ++h) {
          for (int w = 0; w < 32; ++w) {
            // Data is stored in channel-major order: RRR...GGG...BBB...
            batch_data(i, c, h, w) = data_[current_index_ + i][c * 1024 + h * 32 + w];
          }
        }
      }

      // Set one-hot label
      int label = labels_[current_index_ + i];
      batch_labels(i, label, 0, 0) = 1.0;
    }

    current_index_ += actual_batch_size;
    return true;
  }

  void reset() { current_index_ = 0; }

  size_t size() const { return data_.size(); }
};

// Softmax activation for tensors
void apply_tensor_softmax(Tensor<float> &tensor) {
  size_t batch_size = tensor.shape()[0];
  size_t num_classes = tensor.shape()[1];

  for (size_t batch = 0; batch < batch_size; ++batch) {
    // Find max for numerical stability
    float max_val = tensor(batch, 0, 0, 0);
    for (size_t j = 1; j < num_classes; ++j) {
      max_val = std::max(max_val, tensor(batch, j, 0, 0));
    }

    // Compute exponentials and sum
    float sum = 0.0;
    for (size_t j = 0; j < num_classes; ++j) {
      tensor(batch, j, 0, 0) = std::exp(tensor(batch, j, 0, 0) - max_val);
      sum += tensor(batch, j, 0, 0);
    }

    // Normalize
    for (size_t j = 0; j < num_classes; ++j) {
      tensor(batch, j, 0, 0) /= sum;
    }
  }
}

// Accuracy calculation for tensors
float calculate_tensor_accuracy(const Tensor<float> &predictions,
                                const Tensor<float> &targets) {
  int correct = 0;
  size_t batch_size = predictions.shape()[0];
  size_t num_classes = predictions.shape()[1];

  for (size_t i = 0; i < batch_size; ++i) {
    // Find predicted class (argmax)
    int pred_class = 0;
    float max_pred = predictions(i, 0, 0, 0);
    for (size_t j = 1; j < num_classes; ++j) {
      if (predictions(i, j, 0, 0) > max_pred) {
        max_pred = predictions(i, j, 0, 0);
        pred_class = j;
      }
    }

    // Find true class
    int true_class = 0;
    for (size_t j = 0; j < num_classes; ++j) {
      if (targets(i, j, 0, 0) > 0.5) {
        true_class = j;
        break;
      }
    }

    if (pred_class == true_class) {
      correct++;
    }
  }

  return static_cast<float>(correct) / batch_size;
}

// Training function for CNN tensor model
void train_cnn_model(tnn::Sequential<float> &model,
                     CIFAR10DataLoader &train_loader,
                     CIFAR10DataLoader &test_loader, int epochs = 10,
                     int batch_size = 32, float learning_rate = 0.001) {
  tnn::SGD<float> optimizer(learning_rate, 0.9);
  
  auto loss_function = tnn::LossFactory<float>::create_crossentropy(cifar10_constants::EPSILON);

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
      apply_tensor_softmax(predictions);

      // Compute loss and accuracy
      float loss =
          loss_function->compute_loss(predictions, batch_labels);
      float accuracy = calculate_tensor_accuracy(predictions, batch_labels);

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
      apply_tensor_softmax(predictions);

      val_loss +=
          loss_function->compute_loss(predictions, batch_labels);
      val_accuracy += calculate_tensor_accuracy(predictions, batch_labels);
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
    if ((epoch + 1) % 4 == 0) {
      std::cout << "Current learning rate: " << optimizer.get_learning_rate()
                << std::endl;
      float new_lr = 1.0f * optimizer.get_learning_rate() * 0.8f;
      optimizer.set_learning_rate(new_lr);
      printf("Learning rate decayed to: %.6f\n", new_lr);
    }
  }
}

int main() {
  try {
    std::cout << "CIFAR-10 CNN Tensor<float> Neural Network Training" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Load data
    CIFAR10DataLoader train_loader, test_loader;

    std::vector<std::string> train_files;
    for (int i = 1; i <= 5; ++i) {
      train_files.push_back("./data/cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin");
    }
    if (!train_loader.load_data(train_files)) {
      return -1;
    }

    if (!test_loader.load_data({"./data/cifar-10-batches-bin/test_batch.bin"})) {
      return -1;
    }

    // Create CNN model architecture for CIFAR-10
    std::cout << "\nBuilding CNN model architecture for CIFAR-10..."
              << std::endl;

    auto model =
        tnn::SequentialBuilder<float>("cifar10_cnn_classifier")
            // Input: 3x32x32 (channels, height, width)
            .conv2d(3, 16, 3, 3, 1, 1, 0, 0, "relu", true, "conv1") // 3x32x32 -> 32x30x30
            .maxpool2d(3, 3, 3, 3, 0, 0, "maxpool1") // 32x30x30 -> 16x10x10
            .conv2d(16, 64, 3, 3, 1, 1, 0, 0, "relu", true, "conv2") // 16x10x10 -> 64x8x8
            .maxpool2d(4, 4, 4, 4, 0, 0, "maxpool2") // 64x8x8 -> 64x2x2
            .dense(64 * 2 * 2, 10, "linear", true, "fc1") // Flatten to 256 -> 10
            .build();

    model.enable_profiling(true); // Enable profiling for performance analysis

    // Print model summary
    model.print_summary(std::vector<size_t>{
        cifar10_constants::BATCH_SIZE, 3, 32, 32}); // Show summary with single image input

    // Train the CNN model with appropriate hyperparameters
    std::cout << "\nStarting CIFAR-10 CNN training..." << std::endl;
    train_cnn_model(model, train_loader, test_loader,
                    cifar10_constants::EPOCHS,  // epochs
                    cifar10_constants::BATCH_SIZE, // batch_size
                    cifar10_constants::LR_INITIAL // learning_rate
    );

    std::cout
        << "\nCIFAR-10 CNN Tensor<float> model training completed successfully!"
        << std::endl;

    model.save_to_file("model_snapshots/cifar10_cnn_model");
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
