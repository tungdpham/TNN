#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <fstream> // Required for std::ifstream and std::ofstream
#include <sstream> // Required for std::stringstream
#include <algorithm> // Required for std::shuffle, std::min
#include <numeric> // Required for std::iota
#include <random> // Required for std::random_device, std::mt19937
#include <limits> // Required for std::numeric_limits
#include "layers/layers.hpp"
#include "layers/optimizers.hpp"
#include "tensor/tensor.hpp"
#include "pipeline/pipeline_stage.hpp"
#include "pipeline/pipeline_orchestrator.hpp"
#include "pipeline/communication.hpp"

// Assuming the MNISTCNNDataLoader and loss functions are in a shared header
// For this example, I'll redefine them here for simplicity.
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
    std::getline(file, line); // Skip header

    data_.clear();
    labels_.clear();

    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string cell;
      std::getline(ss, cell, ',');
      labels_.push_back(std::stoi(cell));
      std::vector<float> row;
      while (std::getline(ss, cell, ',')) {
        row.push_back(std::stod(cell) / 255.0);
      }
      data_.push_back(row);
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
  }

  bool get_batch(int batch_size, Tensor<float> &batch_data, Tensor<float> &batch_labels) {
    if (current_index_ >= data_.size()) return false;
    int actual_batch_size = std::min(batch_size, static_cast<int>(data_.size() - current_index_));
    batch_data = Tensor<float>({(size_t)actual_batch_size, 1, 28, 28});
    batch_labels = Tensor<float>({(size_t)actual_batch_size, 10, 1, 1});
    batch_labels.fill(0.0f);
    for (int i = 0; i < actual_batch_size; ++i) {
      for (int h = 0; h < 28; ++h) {
        for (int w = 0; w < 28; ++w) {
          batch_data(i, 0, h, w) = data_[current_index_ + i][h * 28 + w];
        }
      }
      batch_labels(i, labels_[current_index_ + i], 0, 0) = 1.0f;
    }
    current_index_ += actual_batch_size;
    return true;
  }

  void reset() { current_index_ = 0; }
  size_t size() const { return data_.size(); }
};

class TensorCrossEntropyLoss {
public:
  static float compute_loss(const Tensor<float> &predictions, const Tensor<float> &targets) {
    float total_loss = 0.0;
    const float epsilon = 1e-15;
    for (size_t i = 0; i < predictions.batch_size(); ++i) {
      for (size_t j = 0; j < predictions.channels(); ++j) {
        if (targets(i, j, 0, 0) > 0.5) {
          total_loss -= std::log(std::max(epsilon, predictions(i, j, 0, 0)));
        }
      }
    }
    return total_loss / predictions.batch_size();
  }
};

void apply_tensor_softmax(Tensor<float> &tensor) {
  for (size_t i = 0; i < tensor.batch_size(); ++i) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < tensor.channels(); ++j) {
      max_val = std::max(max_val, tensor(i, j, 0, 0));
    }
    float sum = 0.0f;
    for (size_t j = 0; j < tensor.channels(); ++j) {
      tensor(i, j, 0, 0) = std::exp(tensor(i, j, 0, 0) - max_val);
      sum += tensor(i, j, 0, 0);
    }
    for (size_t j = 0; j < tensor.channels(); ++j) {
      tensor(i, j, 0, 0) /= sum;
    }
  }
}

float calculate_tensor_accuracy(const Tensor<float> &predictions, const Tensor<float> &targets) {
    int correct = 0;
    for (size_t i = 0; i < predictions.batch_size(); ++i) {
        int pred_class = 0, true_class = 0;
        float max_pred = -1.0f;
        for (size_t j = 0; j < predictions.channels(); ++j) {
            if (predictions(i, j, 0, 0) > max_pred) {
                max_pred = predictions(i, j, 0, 0);
                pred_class = j;
            }
            if (targets(i, j, 0, 0) > 0.5) {
                true_class = j;
            }
        }
        if (pred_class == true_class) correct++;
    }
    return (float)correct / predictions.batch_size();
}


void train_pipeline_model(
    pipeline::PipelineOrchestrator<float>& orchestrator,
    MNISTCNNDataLoader& train_loader,
    MNISTCNNDataLoader& test_loader,
    int epochs = 10,
    int batch_size = 32,
    float learning_rate = 0.001) {

    // Create and assign an optimizer for each stage
    for (auto& stage : orchestrator.get_stages()) {
        auto params = stage->parameters();
        if (!params.empty()) {
            stage->set_optimizer(std::make_unique<layers::Adam<float>>(params, learning_rate));
        }
    }

    std::cout << "Starting Pipeline CNN model training..." << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size
              << ", Learning rate: " << learning_rate << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // --- Training Loop ---
    for (int epoch = 0; epoch < epochs; ++epoch) {
        train_loader.reset();
        train_loader.shuffle();
        float running_loss = 0.0;
        float running_accuracy = 0.0;
        int batch_count = 0;

        auto epoch_start_time = std::chrono::high_resolution_clock::now();

        Tensor<float> batch_data, batch_labels;
        while (train_loader.get_batch(batch_size, batch_data, batch_labels)) {
            auto result = orchestrator.train_batch(batch_data, batch_labels);
            running_loss += result.first;
            running_accuracy += result.second;
            batch_count++;

            if (batch_count % 10 == 0) {
                std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "], Step [" << batch_count << "/" << train_loader.size() / batch_size
                          << "], Loss: " << std::fixed << std::setprecision(4) << running_loss / batch_count
                          << ", Accuracy: " << std::fixed << std::setprecision(4) << running_accuracy / batch_count << std::endl;
            }
        }
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end_time - epoch_start_time);

        std::cout << "Epoch " << epoch + 1 << " completed in " << epoch_duration.count() << " seconds." << std::endl;
        std::cout << "  Average Training Loss: " << running_loss / batch_count << std::endl;
        std::cout << "  Average Training Accuracy: " << running_accuracy / batch_count << std::endl;
    }

    // --- Print Profiling Info ---
    orchestrator.print_profiling_info();

    // --- Save Model ---
    // std::cout << "Saving model..." << std::endl;
    // std::ofstream file("model_snapshots/mnist_cnn_model.bin", std::ios::binary);
    // orchestrator.save_model(file);
    // file.close();
    std::cout << "Model saved successfully!" << std::endl;
}


int main() {
    try {
        std::cout << "MNIST CNN Pipeline Training" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        // Load data
        MNISTCNNDataLoader train_loader, test_loader;
        train_loader.load_data("./data/mnist_train.csv");
        test_loader.load_data("./data/mnist_test.csv");

        // --- Define Pipeline Stages ---
        // We'll split the model into two stages.

        // Stage 1
        auto stage1 = std::make_unique<pipeline::PipelineStage<float>>(0, 0, 8, 8); // 8 threads, 8 OMP threads each
        stage1->add_layer(Layers::blas_conv2d<float>(1, 8, 5, 5, 1, 1, 0, 0, "relu", true, "C1"));
        stage1->add_layer(Layers::maxpool2d<float>(3, 3, 3, 3, 0, 0, "P1"));
        stage1->add_layer(Layers::blas_conv2d<float>(8, 16, 1, 1, 1, 1, 0, 0, "relu", true, "C2_1x1"));
        stage1->add_layer(Layers::blas_conv2d<float>(16, 48, 5, 5, 1, 1, 0, 0, "relu", true, "C3"));

        // Stage 2
        auto stage2 = std::make_unique<pipeline::PipelineStage<float>>(1, 1, 8, 8); // 8 threads, 8 OMP threads each
        stage2->add_layer(Layers::maxpool2d<float>(2, 2, 2, 2, 0, 0, "P2"));
        stage2->add_layer(Layers::flatten<float>("flatten"));
        stage2->add_layer(Layers::blas_dense<float>(48 * 2 * 2, 10, "none", true, "output"));
        stage2->add_layer(Layers::activation<float>("softmax", "softmax_activation"));


        std::vector<std::unique_ptr<pipeline::PipelineStage<float>>> stages;
        stages.push_back(std::move(stage1));
        stages.push_back(std::move(stage2));

        auto communicator = std::make_unique<pipeline::InProcessCommunicator<float>>();
        
        // Using 4 micro-batches for the pipeline
        pipeline::PipelineOrchestrator<float> orchestrator(std::move(stages), std::move(communicator), 4);
        
        train_pipeline_model(orchestrator, train_loader, test_loader, 5, 128, 0.01);

        std::cout << "\nMNIST CNN Pipeline model training completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
