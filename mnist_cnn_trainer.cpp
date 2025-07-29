#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "layers/layers.hpp"
#include "layers/sequential.hpp"
#include "tensor/tensor.hpp"

// Simple tensor optimizer implementation
class TensorSGDOptimizer {
private:
  float learning_rate_;
  float momentum_;
  std::vector<Tensor<float>> velocities_;
  bool initialized_;

public:
  TensorSGDOptimizer(float learning_rate = 0.01, float momentum = 0.9)
      : learning_rate_(learning_rate), momentum_(momentum),
        initialized_(false) {}

  void update(std::vector<Tensor<float> *> &parameters,
              std::vector<Tensor<float> *> &gradients) {
    if (!initialized_) {
      velocities_.resize(parameters.size());

      for (size_t i = 0; i < parameters.size(); ++i) {
        velocities_[i] = Tensor<float>(parameters[i]->shape());
        velocities_[i].fill(0.0);
      }
      initialized_ = true;
    }

    for (size_t i = 0; i < parameters.size(); ++i) {
      // Apply momentum: v = momentum * v - learning_rate * grad
      velocities_[i] *= momentum_;
      velocities_[i] -= (*gradients[i]) * learning_rate_;

      // Update parameters: param += v
      (*parameters[i]) += velocities_[i];
    }
  }

  void set_learning_rate(float lr) { learning_rate_ = lr; }

  float get_learning_rate() const { return learning_rate_; }
};

// Enhanced data loader for MNIST CSV format adapted for CNN (2D images)
class MNISTCNNDataLoader {
private:
  std::vector<std::vector<float>> data_;
  std::vector<int> labels_;
  size_t current_index_;

public:
  bool load_data(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return false;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);

    data_.clear();
    labels_.clear();

    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string cell;

      // First column is label
      std::getline(ss, cell, ',');
      labels_.push_back(std::stoi(cell));

      // Remaining columns are pixel values
      std::vector<float> row;
      while (std::getline(ss, cell, ',')) {
        row.push_back(std::stod(cell) / 255.0); // Normalize to [0, 1]
      }
      data_.push_back(row);
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

    // Create batch data tensor for CNN: (batch_size, channels=1, height=28,
    // width=28)
    batch_data = Tensor<float>(
        std::vector<size_t>{static_cast<size_t>(actual_batch_size), 1, 28, 28});

    // Create batch labels tensor (batch_size, 10, 1, 1) - one-hot encoded
    batch_labels = Tensor<float>(
        std::vector<size_t>{static_cast<size_t>(actual_batch_size), 10, 1, 1});
    batch_labels.fill(0.0);

    for (int i = 0; i < actual_batch_size; ++i) {
      // Copy pixel data and reshape to 28x28
      for (int h = 0; h < 28; ++h) {
        for (int w = 0; w < 28; ++w) {
          batch_data(i, 0, h, w) = data_[current_index_ + i][h * 28 + w];
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

// Loss functions for tensors
class TensorCrossEntropyLoss {
public:
  static float compute_loss(const Tensor<float> &predictions,
                            const Tensor<float> &targets) {
    float total_loss = 0.0;
    const float epsilon = 1e-15; // Small value to prevent log(0)

    size_t batch_size = predictions.shape()[0];
    size_t num_classes = predictions.shape()[1];

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_classes; ++j) {
        if (targets(i, j, 0, 0) > 0.5) { // This is the correct class
          float pred = std::max(
              epsilon, std::min(1.0f - epsilon, predictions(i, j, 0, 0)));
          total_loss -= std::log(pred);
        }
      }
    }

    return total_loss / batch_size;
  }

  static Tensor<float> compute_gradient(const Tensor<float> &predictions,
                                        const Tensor<float> &targets) {
    Tensor<float> gradient = predictions;

    size_t batch_size = predictions.shape()[0];
    size_t num_classes = predictions.shape()[1];

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_classes; ++j) {
        gradient(i, j, 0, 0) = predictions(i, j, 0, 0) - targets(i, j, 0, 0);
      }
    }

    // Average over batch
    gradient /= static_cast<float>(batch_size);
    return gradient;
  }
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
void train_cnn_model(tensor_layers::TensorSequential<float> &model,
                     MNISTCNNDataLoader &train_loader,
                     MNISTCNNDataLoader &test_loader, int epochs = 10,
                     int batch_size = 32, float learning_rate = 0.001) {
  TensorSGDOptimizer optimizer(learning_rate, 0.9);

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
          TensorCrossEntropyLoss::compute_loss(predictions, batch_labels);
      float accuracy = calculate_tensor_accuracy(predictions, batch_labels);

      total_loss += loss;
      total_accuracy += accuracy;
      num_batches++;

      // Backward pass
      Tensor<float> loss_gradient =
          TensorCrossEntropyLoss::compute_gradient(predictions, batch_labels);
      model.backward(loss_gradient);

      // Update parameters
      auto params = model.parameters();
      auto grads = model.gradients();
      optimizer.update(params, grads);

      // Print progress every 10 batches (more frequent for CNN)
      if (num_batches % 200 == 0) {
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
          TensorCrossEntropyLoss::compute_loss(predictions, batch_labels);
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
    if ((epoch + 1) % 3 == 0) {
      std::cout << "Current learning rate: " << optimizer.get_learning_rate()
                << std::endl;
      float new_lr = optimizer.get_learning_rate() * 0.8;
      optimizer.set_learning_rate(new_lr);
      std::cout << "Learning rate reduced to: " << new_lr << std::endl;
    }
  }
}

int main() {
  try {
    std::cout << "MNIST CNN Tensor<float> Neural Network Training" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Load data
    MNISTCNNDataLoader train_loader, test_loader;

    if (!train_loader.load_data("./data/mnist_train.csv")) {
      return -1;
    }

    if (!test_loader.load_data("./data/mnist_test.csv")) {
      return -1;
    }

    // Create CNN model architecture matching Mojo configuration
    std::cout << "\nBuilding CNN model architecture (matching Mojo config)..."
              << std::endl;

    auto model =
        tensor_layers::TensorSequentialBuilder<float>("mnist_cnn_classifier")
            // I1: input 28x28x1 (implicit - handled by data loader)

            // C1: convolution 5x5 kernel, stride 1, relu activation
            // Output size: 8x24x24 (padding 0) (28-5+1=24) (C H W order)
            .blas_conv2d(1, 8, 5, 5, 1, 1, 0, 0, "relu", true, "C1")

            // P1: max pool 3x3 blocks, stride 3
            // Output size: 8x8x8 (24/3=8) (C H W order)
            .maxpool2d(3, 3, 3, 3, 0, 0, "P1")

            // C2: inceptoin layer
            .blas_conv2d(8, 16, 1, 1, 1, 1, 0, 0, "relu", true,
                         "C2") // 1x1 conv

            // C3: convolution 5x5 kernel, stride 1, relu activation
            // Output size: 16x4x4 (8-5+1=4) (C H W order)
            .blas_conv2d(16, 48, 5, 5, 1, 1, 0, 0, "relu", true, "C2")

            // P2: max pool 2x2 blocks, stride 2
            // Output size: 48x2x2 (4/2=2) (C H W order)
            .maxpool2d(2, 2, 2, 2, 0, 0, "P2")

            // FC1: fully connected
            .blas_dense(48 * 4, 10, "linear", true,
                        "output") // Output layer with 10 classes

            .build();

    model.enable_profiling(true); // Enable profiling for performance analysis
    // Print model summary
    model.print_summary(std::vector<size_t>{
        1, 1, 28, 28}); // Show summary with single image input

    // Train the CNN model with appropriate hyperparameters
    std::cout << "\nStarting Mojo-style CNN training..." << std::endl;
    train_cnn_model(model, train_loader, test_loader,
                    5,  // epochs
                    64, // batch_size (moderate batch size)
                    0.01 // learning_rate (slightly higher for simpler model)
    );

    std::cout
        << "\nMNIST CNN Tensor<float> model training completed successfully!"
        << std::endl;

    model.save_to_file("model_snapshots/mnist_cnn_model");
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
