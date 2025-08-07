#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <memory>
#include <numeric>
#include <string_view>
#include <atomic>
#include <omp.h>

#include "layers/layers.hpp"
#include "layers/sequential.hpp"
#include "tensor/tensor.hpp"
#include "layers/optimizers.hpp"

// Constants for MNIST dataset
namespace mnist_constants {
    constexpr size_t IMAGE_HEIGHT = 28;
    constexpr size_t IMAGE_WIDTH = 28;
    constexpr size_t IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
    constexpr size_t NUM_CLASSES = 10;
    constexpr float NORMALIZATION_FACTOR = 255.0f;
    constexpr float EPSILON = 1e-15f;
    constexpr int PROGRESS_PRINT_INTERVAL = 50;
    constexpr int LR_DECAY_INTERVAL = 2;
    constexpr float LR_DECAY_FACTOR = 0.8f;
}

// Enhanced data loader for MNIST CSV format adapted for CNN (2D images)
class MNISTCNNDataLoader {
private:
  std::vector<std::vector<float>> data_;
  std::vector<int> labels_;
  size_t current_index_;
  mutable std::mt19937 rng_{std::random_device{}()};
  
  // Pre-computed batch storage
  std::vector<Tensor<float>> batched_data_;
  std::vector<Tensor<float>> batched_labels_;
  size_t current_batch_index_;
  int batch_size_;
  bool batches_prepared_;

public:
  MNISTCNNDataLoader() : current_index_(0), current_batch_index_(0), batch_size_(32), batches_prepared_(false) {
    // Reserve space to reduce allocations
    data_.reserve(60000); // MNIST train set size
    labels_.reserve(60000);
  }

  bool load_data(std::string_view filename) {
    std::ifstream file{filename.data()};
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return false;
    }

    std::string line;
    line.reserve(3136); // Reserve space for MNIST line (784 pixels + label)
    
    // Skip header line
    if (!std::getline(file, line)) {
      std::cerr << "Error: Empty file " << filename << std::endl;
      return false;
    }

    data_.clear();
    labels_.clear();

    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string cell;

      // First column is label
      if (!std::getline(ss, cell, ',')) continue;
      labels_.push_back(std::stoi(cell));

      // Remaining columns are pixel values - reserve space for efficiency
      std::vector<float> row;
      row.reserve(mnist_constants::IMAGE_SIZE);
      
      while (std::getline(ss, cell, ',')) {
        row.push_back(std::stod(cell) / mnist_constants::NORMALIZATION_FACTOR);
      }
      
      // Validate image size
      if (row.size() != mnist_constants::IMAGE_SIZE) {
        std::cerr << "Warning: Invalid image size " << row.size() 
                  << " expected " << mnist_constants::IMAGE_SIZE << std::endl;
        continue;
      }
      
      data_.push_back(std::move(row));
    }

    current_index_ = 0;
    std::cout << "Loaded " << data_.size() << " samples from " << filename
              << std::endl;
    return !data_.empty();
  }

  void shuffle() {
    if (data_.empty()) return;
    
    std::vector<size_t> indices(data_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);

    // Use move semantics for better performance
    std::vector<std::vector<float>> shuffled_data;
    std::vector<int> shuffled_labels;
    shuffled_data.reserve(data_.size());
    shuffled_labels.reserve(labels_.size());

    for (const auto& idx : indices) {
      shuffled_data.emplace_back(std::move(data_[idx]));
      shuffled_labels.emplace_back(labels_[idx]);
    }

    data_ = std::move(shuffled_data);
    labels_ = std::move(shuffled_labels);
    current_index_ = 0;
    
    // Invalidate pre-computed batches since data order changed
    batches_prepared_ = false;
  }

  // Pre-compute all batches for efficient training
  void prepare_batches(int batch_size) {
    if (data_.empty()) {
      std::cerr << "Warning: No data loaded, cannot prepare batches!" << std::endl;
      return;
    }
    
    batch_size_ = batch_size;
    batched_data_.clear();
    batched_labels_.clear();
    
    const size_t num_samples = data_.size();
    const size_t num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division
    
    batched_data_.reserve(num_batches);
    batched_labels_.reserve(num_batches);
    
    std::cout << "Preparing " << num_batches << " batches of size " << batch_size << "..." << std::endl;
    
    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      const size_t start_idx = batch_idx * batch_size;
      const size_t end_idx = std::min(start_idx + batch_size, num_samples);
      const int actual_batch_size = static_cast<int>(end_idx - start_idx);
      
      // Create batch tensors
      Tensor<float> batch_data(
          std::vector<size_t>{static_cast<size_t>(actual_batch_size), 1, 
                             mnist_constants::IMAGE_HEIGHT, mnist_constants::IMAGE_WIDTH});
      
      Tensor<float> batch_labels(
          std::vector<size_t>{static_cast<size_t>(actual_batch_size), 
                             mnist_constants::NUM_CLASSES, 1, 1});
      batch_labels.fill(0.0f);
      
      // Fill batch data in parallel
      #pragma omp parallel for if(actual_batch_size > 16)
      for (int i = 0; i < actual_batch_size; ++i) {
        const size_t sample_idx = start_idx + i;
        const auto& image_data = data_[sample_idx];
        
        // Copy pixel data and reshape to 28x28 with better cache locality
        for (int h = 0; h < static_cast<int>(mnist_constants::IMAGE_HEIGHT); ++h) {
          for (int w = 0; w < static_cast<int>(mnist_constants::IMAGE_WIDTH); ++w) {
            batch_data(i, 0, h, w) = image_data[h * mnist_constants::IMAGE_WIDTH + w];
          }
        }
        
        // Set one-hot label
        const int label = labels_[sample_idx];
        if (label >= 0 && label < static_cast<int>(mnist_constants::NUM_CLASSES)) {
          batch_labels(i, label, 0, 0) = 1.0f;
        }
      }
      
      batched_data_.emplace_back(std::move(batch_data));
      batched_labels_.emplace_back(std::move(batch_labels));
    }
    
    current_batch_index_ = 0;
    batches_prepared_ = true;
    std::cout << "Batch preparation completed!" << std::endl;
  }

  // Fast batch retrieval from pre-computed batches
  bool get_next_batch(Tensor<float> &batch_data, Tensor<float> &batch_labels) {
    if (!batches_prepared_) {
      std::cerr << "Error: Batches not prepared! Call prepare_batches() first." << std::endl;
      return false;
    }
    
    if (current_batch_index_ >= batched_data_.size()) {
      return false; // No more batches
    }
    
    batch_data = batched_data_[current_batch_index_];
    batch_labels = batched_labels_[current_batch_index_];
    ++current_batch_index_;
    
    return true;
  }

  // Legacy method for backward compatibility - now uses pre-computed batches when available
  bool get_batch(int batch_size, Tensor<float> &batch_data,
                 Tensor<float> &batch_labels) {
    // If batches are prepared and batch size matches, use fast path
    if (batches_prepared_ && batch_size == batch_size_) {
      return get_next_batch(batch_data, batch_labels);
    }
    
    // Fallback to original implementation for different batch sizes
    if (current_index_ >= data_.size()) {
      return false; // No more data
    }

    const int actual_batch_size =
        std::min(batch_size, static_cast<int>(data_.size() - current_index_));

    // Create batch data tensor for CNN: (batch_size, channels=1, height=28, width=28)
    batch_data = Tensor<float>(
        std::vector<size_t>{static_cast<size_t>(actual_batch_size), 1, 
                           mnist_constants::IMAGE_HEIGHT, mnist_constants::IMAGE_WIDTH});

    // Create batch labels tensor (batch_size, 10, 1, 1) - one-hot encoded
    batch_labels = Tensor<float>(
        std::vector<size_t>{static_cast<size_t>(actual_batch_size), 
                           mnist_constants::NUM_CLASSES, 1, 1});
    batch_labels.fill(0.0f);

    // Parallelize batch processing for better performance
    #pragma omp parallel for if(actual_batch_size > 16)
    for (int i = 0; i < actual_batch_size; ++i) {
      const auto& image_data = data_[current_index_ + i];
      
      // Copy pixel data and reshape to 28x28 with better cache locality
      for (int h = 0; h < static_cast<int>(mnist_constants::IMAGE_HEIGHT); ++h) {
        for (int w = 0; w < static_cast<int>(mnist_constants::IMAGE_WIDTH); ++w) {
          batch_data(i, 0, h, w) = image_data[h * mnist_constants::IMAGE_WIDTH + w];
        }
      }

      // Set one-hot label
      const int label = labels_[current_index_ + i];
      if (label >= 0 && label < static_cast<int>(mnist_constants::NUM_CLASSES)) {
        batch_labels(i, label, 0, 0) = 1.0f;
      }
    }

    current_index_ += actual_batch_size;
    return true;
  }

  void reset() { 
    current_index_ = 0; 
    current_batch_index_ = 0;
  }

  size_t size() const { return data_.size(); }
  
  size_t num_batches() const { 
    return batches_prepared_ ? batched_data_.size() : 0; 
  }
  
  bool are_batches_prepared() const { return batches_prepared_; }
};

// Optimized loss functions for tensors
class TensorCrossEntropyLoss {
public:
  static float compute_loss(const Tensor<float> &predictions,
                            const Tensor<float> &targets) {
    const size_t batch_size = predictions.shape()[0];
    const size_t num_classes = predictions.shape()[1];
    
    double total_loss = 0.0; // Use double for better numerical precision
    
    // Parallelize loss computation for large batches
    #pragma omp parallel for reduction(+:total_loss) if(batch_size > 32)
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_classes; ++j) {
        if (targets(i, j, 0, 0) > 0.5f) { // This is the correct class
          const float pred = std::clamp(predictions(i, j, 0, 0), 
                                       mnist_constants::EPSILON, 
                                       1.0f - mnist_constants::EPSILON);
          total_loss -= std::log(pred);
          break; // Only one class is true per sample
        }
      }
    }

    return static_cast<float>(total_loss / batch_size);
  }

  static Tensor<float> compute_gradient(const Tensor<float> &predictions,
                                        const Tensor<float> &targets) {
    Tensor<float> gradient = predictions;
    const size_t batch_size = predictions.shape()[0];
    const size_t num_classes = predictions.shape()[1];
    const float inv_batch_size = 1.0f / static_cast<float>(batch_size);

    // Parallelize gradient computation
    #pragma omp parallel for if(batch_size > 32)
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_classes; ++j) {
        gradient(i, j, 0, 0) = (predictions(i, j, 0, 0) - targets(i, j, 0, 0)) * inv_batch_size;
      }
    }

    return gradient;
  }
};

// Optimized softmax activation for tensors with numerical stability
void apply_tensor_softmax(Tensor<float> &tensor) {
  const size_t batch_size = tensor.shape()[0];
  const size_t num_classes = tensor.shape()[1];

  // Parallelize softmax computation across batch
  #pragma omp parallel for if(batch_size > 16)
  for (size_t batch = 0; batch < batch_size; ++batch) {
    // Find max for numerical stability
    float max_val = tensor(batch, 0, 0, 0);
    for (size_t j = 1; j < num_classes; ++j) {
      max_val = std::max(max_val, tensor(batch, j, 0, 0));
    }

    // Compute exponentials and sum in one pass
    float sum = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      const float exp_val = std::exp(tensor(batch, j, 0, 0) - max_val);
      tensor(batch, j, 0, 0) = exp_val;
      sum += exp_val;
    }

    // Normalize with safety check
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
  #pragma omp parallel for reduction(+:total_correct) if(batch_size > 16)
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

    if (pred_class == true_class && true_class != -1) {
      total_correct++;
    }
  }

  return static_cast<float>(total_correct) / static_cast<float>(batch_size);
}

// Optimized training function for CNN tensor model with pre-computed batches
void train_cnn_model(layers::Sequential<float> &model,
                     MNISTCNNDataLoader &train_loader,
                     MNISTCNNDataLoader &test_loader, 
                     int epochs = 10,
                     int batch_size = 32, 
                     float learning_rate = 0.001f) {
  layers::SGD<float> optimizer(learning_rate, 0.9f);

  // Pre-allocate tensors to reduce memory allocations
  Tensor<float> batch_data, batch_labels, predictions;
  
  std::cout << "Starting optimized CNN tensor model training..." << std::endl;
  std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size
            << ", Learning rate: " << learning_rate << std::endl;
  std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
  std::cout << std::string(70, '=') << std::endl;

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

    // Use fast batch iteration with pre-computed batches
    while (train_loader.get_next_batch(batch_data, batch_labels)) {
      ++num_batches;
      
      // Forward pass
      predictions = model.forward(batch_data);
      apply_tensor_softmax(predictions);

      // Compute loss and accuracy
      const float loss = TensorCrossEntropyLoss::compute_loss(predictions, batch_labels);
      const float accuracy = calculate_tensor_accuracy(predictions, batch_labels);

      total_loss += loss;
      total_accuracy += accuracy;

      // Backward pass
      const Tensor<float> loss_gradient = 
          TensorCrossEntropyLoss::compute_gradient(predictions, batch_labels);
      model.backward(loss_gradient);

      // Update parameters
      auto params = model.parameters();
      auto grads = model.gradients();
      optimizer.update(params, grads);

      // Print progress at intervals
      if (num_batches % mnist_constants::PROGRESS_PRINT_INTERVAL == 0) {
        model.print_profiling_summary();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << ", Batch: " << num_batches 
                  << ", Loss: " << std::fixed << std::setprecision(4) << loss
                  << ", Acc: " << std::setprecision(2) << accuracy * 100.0f << "%"
                  << std::endl;
      }
    }

    const float avg_train_loss = static_cast<float>(total_loss / num_batches);
    const float avg_train_accuracy = static_cast<float>(total_accuracy / num_batches);

    // Optimized validation phase - use pre-computed batches
    model.eval();
    test_loader.reset();

    double val_loss = 0.0;
    double val_accuracy = 0.0;
    int val_batches = 0;

    // Use fast batch iteration with pre-computed validation batches
    while (test_loader.get_next_batch(batch_data, batch_labels)) {
      predictions = model.forward(batch_data);
      apply_tensor_softmax(predictions);

      val_loss += TensorCrossEntropyLoss::compute_loss(predictions, batch_labels);
      val_accuracy += calculate_tensor_accuracy(predictions, batch_labels);
      ++val_batches;
    }

    const float avg_val_loss = static_cast<float>(val_loss / val_batches);
    const float avg_val_accuracy = static_cast<float>(val_accuracy / val_batches);

    const auto epoch_end = std::chrono::high_resolution_clock::now();
    const auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(
        epoch_end - epoch_start);

    // Print epoch summary with improved formatting
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Epoch " << epoch + 1 << "/" << epochs 
              << " completed in " << epoch_duration.count() << "s" << std::endl;
    std::cout << "Training   - Loss: " << std::fixed << std::setprecision(4)
              << avg_train_loss << ", Accuracy: " << std::setprecision(2)
              << avg_train_accuracy * 100.0f << "%" << std::endl;
    std::cout << "Validation - Loss: " << std::fixed << std::setprecision(4)
              << avg_val_loss << ", Accuracy: " << std::setprecision(2)
              << avg_val_accuracy * 100.0f << "%" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Optimized learning rate decay with better scheduling
    if ((epoch + 1) % mnist_constants::LR_DECAY_INTERVAL == 0) {
      const float current_lr = optimizer.get_learning_rate();
      const float new_lr = current_lr * mnist_constants::LR_DECAY_FACTOR;
      optimizer.set_learning_rate(new_lr);
      std::cout << "Learning rate decayed: " << std::fixed << std::setprecision(6)
                << current_lr << " → " << new_lr << std::endl;
    }
  }
}

// Optimized main function with better error handling and resource management
int main() {
  try {
    std::cout << "Optimized MNIST CNN Tensor<float> Neural Network Training" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Set OpenMP configuration for optimal performance
    const int num_threads = omp_get_max_threads();
    omp_set_num_threads(std::min(num_threads, 8)); // Limit threads to avoid overhead
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads" << std::endl;

    // Initialize data loaders with improved error handling
    MNISTCNNDataLoader train_loader, test_loader;

    // Validate data loading
    if (!train_loader.load_data("./data/mnist_train.csv")) {
      std::cerr << "Failed to load training data!" << std::endl;
      return -1;
    }

    if (!test_loader.load_data("./data/mnist_test.csv")) {
      std::cerr << "Failed to load test data!" << std::endl;
      return -1;
    }

    std::cout << "Successfully loaded training data: " << train_loader.size() 
              << " samples" << std::endl;
    std::cout << "Successfully loaded test data: " << test_loader.size() 
              << " samples" << std::endl;

    // Create optimized CNN model architecture
    std::cout << "\nBuilding optimized CNN model architecture..." << std::endl;

    auto model =
        layers::SequentialBuilder<float>("optimized_mnist_cnn_classifier")
            // C1: First convolution layer - 5x5 kernel, stride 1, ReLU activation
            // Input: 1x28x28 → Output: 8x24x24 (28-5+1=24)
            .conv2d(1, 8, 5, 5, 1, 1, 0, 0, "relu", true, "conv1")

            // P1: Max pooling layer - 3x3 blocks, stride 3
            // Input: 8x24x24 → Output: 8x8x8 (24/3=8)
            .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")

            // C2: Inception-style 1x1 convolution for dimensionality reduction
            // Input: 8x8x8 → Output: 16x8x8
            .conv2d(8, 16, 1, 1, 1, 1, 0, 0, "relu", true, "conv2_1x1")

            // C3: Second convolution layer - 5x5 kernel, stride 1, ReLU activation
            // Input: 16x8x8 → Output: 48x4x4 (8-5+1=4)
            .conv2d(16, 48, 5, 5, 1, 1, 0, 0, "relu", true, "conv3")

            // P2: Second max pooling layer - 2x2 blocks, stride 2
            // Input: 48x4x4 → Output: 48x2x2 (4/2=2)
            .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")

            // FC1: Fully connected output layer
            // Input: 48x2x2 = 192 features → Output: 10 classes
            .dense(48 * 2 * 2, mnist_constants::NUM_CLASSES, "linear", true, "output")
            .build();
    
    // Enable profiling for performance analysis
    model.enable_profiling(true);
    
    // Print model summary with optimized batch size
    std::cout << "\nModel Architecture Summary:" << std::endl;
    model.print_summary(std::vector<size_t>{
        128, 1, mnist_constants::IMAGE_HEIGHT, mnist_constants::IMAGE_WIDTH});

    // Train the CNN model with optimized hyperparameters
    std::cout << "\nStarting optimized CNN training with improved hyperparameters..." << std::endl;
    
    // Use optimized training parameters
    constexpr int optimal_epochs = 5;
    constexpr int optimal_batch_size = 128; // Good balance between memory and convergence
    constexpr float optimal_learning_rate = 0.01f; // Slightly higher for faster convergence
    
    train_cnn_model(model, train_loader, test_loader,
                    optimal_epochs,
                    optimal_batch_size,
                    optimal_learning_rate);

    std::cout << "\nOptimized MNIST CNN model training completed successfully!" << std::endl;

    // Save model with error handling
    try {
      model.save_to_file("model_snapshots/mnist_cnn_model");
      std::cout << "Model saved to: model_snapshots/mnist_cnn_model" << std::endl;
    } catch (const std::exception& save_error) {
      std::cerr << "Warning: Failed to save model: " << save_error.what() << std::endl;
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
