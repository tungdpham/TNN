#include <math.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "layers/activations.hpp"
#include "layers/layer.hpp"
#include "layers/sequential.hpp"
#include "matrix/matrix.hpp"

// Simple optimizer implementation
class SGDOptimizer {
private:
  float learning_rate_;
  float momentum_;
  std::vector<Matrix<float>> velocities_;
  bool initialized_;

public:
  SGDOptimizer(float learning_rate = 0.01, float momentum = 0.9)
      : learning_rate_(learning_rate), momentum_(momentum),
        initialized_(false) {}

  void update(std::vector<Matrix<float> *> &parameters,
              std::vector<Matrix<float> *> &gradients) {
    if (!initialized_) {
      velocities_.resize(parameters.size());
      for (size_t i = 0; i < parameters.size(); ++i) {
        velocities_[i] = Matrix<float>(parameters[i]->rows, parameters[i]->cols,
                                       parameters[i]->channels);
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

// Data loader for MNIST CSV format
class MNISTDataLoader {
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

  bool get_batch(int batch_size, Matrix<float> &batch_data,
                 Matrix<float> &batch_labels) {
    if (current_index_ >= data_.size()) {
      return false; // No more data
    }

    int actual_batch_size =
        std::min(batch_size, static_cast<int>(data_.size() - current_index_));

    // Create batch data matrix (batch_size x 784)
    batch_data = Matrix<float>(actual_batch_size, 784, 1);

    // Create batch labels matrix (batch_size x 10) - one-hot encoded
    batch_labels = Matrix<float>(actual_batch_size, 10, 1);
    batch_labels.fill(0.0);

    for (int i = 0; i < actual_batch_size; ++i) {
      // Copy pixel data
      for (int j = 0; j < 784; ++j) {
        batch_data(i, j, 0) = data_[current_index_ + i][j];
      }

      // Set one-hot label
      int label = labels_[current_index_ + i];
      batch_labels(i, label, 0) = 1.0;
    }

    current_index_ += actual_batch_size;
    return true;
  }

  void reset() { current_index_ = 0; }

  size_t size() const { return data_.size(); }
};

// Loss functions
class CrossEntropyLoss {
public:
  static float compute_loss(const Matrix<float> &predictions,
                            const Matrix<float> &targets) {
    float total_loss = 0.0;
    const float epsilon = 1e-15; // Small value to prevent log(0)

    for (int i = 0; i < predictions.rows; ++i) {
      for (int j = 0; j < predictions.cols; ++j) {
        if (targets(i, j, 0) > 0.5) { // This is the correct class
          float pred =
              std::max(epsilon, std::min(1.0f - epsilon, predictions(i, j, 0)));
          total_loss -= std::log(pred);
        }
      }
    }

    return total_loss / predictions.rows;
  }

  static Matrix<float> compute_gradient(const Matrix<float> &predictions,
                                        const Matrix<float> &targets) {
    Matrix<float> gradient = predictions;

    for (int i = 0; i < predictions.rows; ++i) {
      for (int j = 0; j < predictions.cols; ++j) {
        gradient(i, j, 0) = predictions(i, j, 0) - targets(i, j, 0);
      }
    }

    // Average over batch
    gradient /= static_cast<float>(predictions.rows);
    return gradient;
  }
};

// Softmax activation (separate from the activation system for simplicity)
void apply_softmax(Matrix<float> &matrix) {
  for (int batch = 0; batch < matrix.rows; ++batch) {
    // Find max for numerical stability
    float max_val = matrix(batch, 0, 0);
    for (int j = 1; j < matrix.cols; ++j) {
      max_val = std::max(max_val, matrix(batch, j, 0));
    }

    // Compute exponentials and sum
    float sum = 0.0;
    for (int j = 0; j < matrix.cols; ++j) {
      matrix(batch, j, 0) = std::exp(matrix(batch, j, 0) - max_val);
      sum += matrix(batch, j, 0);
    }

    // Normalize
    for (int j = 0; j < matrix.cols; ++j) {
      matrix(batch, j, 0) /= sum;
    }
  }
}

// Accuracy calculation
float calculate_accuracy(const Matrix<float> &predictions,
                         const Matrix<float> &targets) {
  int correct = 0;

  for (int i = 0; i < predictions.rows; ++i) {
    // Find predicted class (argmax)
    int pred_class = 0;
    float max_pred = predictions(i, 0, 0);
    for (int j = 1; j < predictions.cols; ++j) {
      if (predictions(i, j, 0) > max_pred) {
        max_pred = predictions(i, j, 0);
        pred_class = j;
      }
    }

    // Find true class
    int true_class = 0;
    for (int j = 0; j < targets.cols; ++j) {
      if (targets(i, j, 0) > 0.5) {
        true_class = j;
        break;
      }
    }

    if (pred_class == true_class) {
      correct++;
    }
  }

  return static_cast<float>(correct) / predictions.rows;
}

// Training function
void train_model(models::Sequential<float> &model,
                 MNISTDataLoader &train_loader, MNISTDataLoader &test_loader,
                 int epochs = 10, int batch_size = 32,
                 float learning_rate = 0.001) {
  SGDOptimizer optimizer(learning_rate, 0.9);

  std::cout << "Starting training..." << std::endl;
  std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size
            << ", Learning rate: " << learning_rate << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    auto epoch_start = std::chrono::high_resolution_clock::now();

    // Training phase
    model.train();
    train_loader.shuffle();
    train_loader.reset();

    float total_loss = 0.0;
    float total_accuracy = 0.0;
    int num_batches = 0;

    Matrix<float> batch_data, batch_labels;
    while (train_loader.get_batch(batch_size, batch_data, batch_labels)) {
      // Forward pass
      Matrix<float> predictions = model.forward(batch_data);
      apply_softmax(predictions);

      // Compute loss and accuracy
      float loss = CrossEntropyLoss::compute_loss(predictions, batch_labels);
      float accuracy = calculate_accuracy(predictions, batch_labels);

      total_loss += loss;
      total_accuracy += accuracy;
      num_batches++;

      // Backward pass
      Matrix<float> loss_gradient =
          CrossEntropyLoss::compute_gradient(predictions, batch_labels);
      model.backward(loss_gradient);

      // Update parameters
      auto params = model.parameters();
      auto grads = model.gradients();
      optimizer.update(params, grads);

      // Print progress every 100 batches
      if (num_batches % 100 == 0) {
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
      Matrix<float> predictions = model.forward(batch_data);
      apply_softmax(predictions);

      val_loss += CrossEntropyLoss::compute_loss(predictions, batch_labels);
      val_accuracy += calculate_accuracy(predictions, batch_labels);
      val_batches++;
    }

    float avg_val_loss = val_loss / val_batches;
    float avg_val_accuracy = val_accuracy / val_batches;

    auto epoch_end = std::chrono::high_resolution_clock::now();
    auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(
        epoch_end - epoch_start);

    // Print epoch summary
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed in "
              << epoch_duration.count() << "s" << std::endl;
    std::cout << "Training   - Loss: " << std::fixed << std::setprecision(4)
              << avg_train_loss << ", Accuracy: " << std::setprecision(2)
              << avg_train_accuracy * 100 << "%" << std::endl;
    std::cout << "Validation - Loss: " << std::fixed << std::setprecision(4)
              << avg_val_loss << ", Accuracy: " << std::setprecision(2)
              << avg_val_accuracy * 100 << "%" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Learning rate decay
    if ((epoch + 1) % 5 == 0) {
      std::cout << "Current learning rate: " << optimizer.get_learning_rate()
                << std::endl;
      float new_lr = optimizer.get_learning_rate() * 0.75;
      optimizer.set_learning_rate(new_lr);
      std::cout << "Learning rate reduced to: " << new_lr << std::endl;
    }
  }
}

int main() {
  try {
    std::cout << "MNIST Neural Network Training" << std::endl;
    std::cout << std::string(40, '=') << std::endl;

    // Load data
    MNISTDataLoader train_loader, test_loader;

    if (!train_loader.load_data("./data/mnist_train.csv")) {
      return -1;
    }

    if (!test_loader.load_data("./data/mnist_test.csv")) {
      return -1;
    }

    // Create model architecture
    auto model =
        std::make_unique<models::Sequential<float>>("mnist_classifier");

    // Build a simple feedforward network
    model->add(Layers::dense<float>(784, 256, "sigmoid", true, "hidden1"))
        .add(Layers::dense<float>(256, 10, "linear", true, "output"));

    // Compile the model
    model->compile(100, 784, 1); // batch_size x features x channels

    // Print model summary
    std::cout << model->summary() << std::endl;

    // Train the model
    train_model(*model, train_loader, test_loader,
                10,  // epochs
                100, // batch_size
                0.01 // learning_rate
    );

    std::cout << "Training completed successfully!" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
