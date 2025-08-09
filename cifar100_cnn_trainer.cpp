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

namespace cifar100_constants {
  constexpr float EPSILON = 1e-15f;
  constexpr int PROGRESS_PRINT_INTERVAL = 50;
  constexpr int EPOCHS = 20; // Number of epochs for training
  constexpr size_t BATCH_SIZE = 32; // Batch size for training
  constexpr int LR_DECAY_INTERVAL = 10; // Learning rate decay interval
  constexpr float LR_DECAY_FACTOR = 0.85f; // Learning rate decay factor
  constexpr float LR_INITIAL = 0.01f; // Initial learning rate for training
} // namespace cifar100_constants

// CIFAR-100 data loader
class CIFAR100DataLoader {
private:
  std::vector<std::vector<float>> data_;
  std::vector<int> labels_;
  size_t current_index_;
  const int num_classes = 100;

public:
  bool load_data(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return false;
    }

    // CIFAR-100 binary format: [1-byte coarse label][1-byte fine label][3072 bytes of pixel data]
    // Total record size = 1 + 1 + 3072 = 3074 bytes, but the website says 3073. Let's check.
    // The first byte is the coarse label, the second is the fine label.
    // Let's assume the record size is 1 (fine label) + 3072 bytes.
    // Ah, the website says "The first byte is the coarse label (the superclass), and the second byte is the fine label (the class within the superclass)."
    // So we need to skip the first byte (coarse label).
    const int record_size = 1 + 1 + 32 * 32 * 3; // coarse + fine + image data

    data_.clear();
    labels_.clear();

    char buffer[record_size];
    while (file.read(buffer, record_size)) {
      // Skip coarse label (buffer[0])
      unsigned char fine_label_char = buffer[1];
      labels_.push_back(static_cast<int>(fine_label_char));

      std::vector<float> image_data(32 * 32 * 3);
      for (int i = 0; i < 3072; ++i) {
        image_data[i] = static_cast<unsigned char>(buffer[i + 2]) / 255.0f;
      }
      data_.push_back(image_data);
    }

    current_index_ = 0;
    std::cout << "Loaded " << data_.size() << " samples from " << filename
              << std::endl;
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

    // Create batch labels tensor (batch_size, 100, 1, 1) - one-hot encoded
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
                     CIFAR100DataLoader &train_loader,
                     CIFAR100DataLoader &test_loader, int epochs = 10,
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
    if ((epoch + 1) % 2 == 0) {
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
    std::cout << "CIFAR-100 CNN Tensor<float> Neural Network Training" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Load data
    CIFAR100DataLoader train_loader, test_loader;

    if (!train_loader.load_data("./data/cifar-100-binary/train.bin")) {
      return -1;
    }

    if (!test_loader.load_data("./data/cifar-100-binary/test.bin")) {
      return -1;
    }

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
            .activation("softmax", "softmax_output") // Softmax activation
            .build();

    model.enable_profiling(true); // Enable profiling for performance analysis
    // Print model summary
    model.print_summary(std::vector<size_t>{
        64, 3, 32, 32}); // Show summary with single image input

    // Train the CNN model with appropriate hyperparameters
    std::cout << "\nStarting CIFAR-100 CNN training..." << std::endl;
    train_cnn_model(model, train_loader, test_loader,
                    20,  // epochs
                    32, // batch_size
                    0.01 // learning_rate
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
