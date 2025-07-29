#include <iostream>
#include <vector>
#include <string>
#include "layers/sequential.hpp"
#include "tensor/tensor.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

// Data loader for MNIST CSV, adapted for testing
class MNISTCNNDataLoader {
private:
    std::vector<std::vector<float>> data_;
    std::vector<int> labels_;
    size_t current_index_;

public:
    bool load_data(const std::string& filename) {
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
                row.push_back(std::stof(cell) / 255.0f);
            }
            data_.push_back(row);
        }
        current_index_ = 0;
        return true;
    }

    bool get_batch(int batch_size, Tensor<float>& batch_data, Tensor<float>& batch_labels) {
        if (current_index_ >= data_.size()) {
            return false;
        }

        int actual_batch_size = std::min(batch_size, static_cast<int>(data_.size() - current_index_));

        batch_data = Tensor<float>(std::vector<size_t>{(size_t)actual_batch_size, 1, 28, 28});
        batch_labels = Tensor<float>(std::vector<size_t>{(size_t)actual_batch_size, 10, 1, 1});
        batch_labels.fill(0.0f);

        for (int i = 0; i < actual_batch_size; ++i) {
            for (int h = 0; h < 28; ++h) {
                for (int w = 0; w < 28; ++w) {
                    batch_data(i, 0, h, w) = data_[current_index_ + i][h * 28 + w];
                }
            }
            int label = labels_[current_index_ + i];
            batch_labels(i, label, 0, 0) = 1.0f;
        }

        current_index_ += actual_batch_size;
        return true;
    }
    
    void reset() { current_index_ = 0; }
    size_t size() const { return data_.size(); }
};

// Accuracy calculation
float calculate_tensor_accuracy(const Tensor<float>& predictions, const Tensor<float>& targets) {
    int correct = 0;
    size_t batch_size = predictions.shape()[0];
    for (size_t i = 0; i < batch_size; ++i) {
        int pred_class = 0;
        float max_pred = predictions(i, 0, 0, 0);
        for (size_t j = 1; j < 10; ++j) {
            if (predictions(i, j, 0, 0) > max_pred) {
                max_pred = predictions(i, j, 0, 0);
                pred_class = j;
            }
        }

        int true_class = 0;
        for (size_t j = 0; j < 10; ++j) {
            if (targets(i, j, 0, 0) > 0.5f) {
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


int main() {
    try {
        std::cout << "Loading MNIST CNN model and testing..." << std::endl;

        // Load the model from file
        auto model = tensor_layers::TensorSequential<float>::from_file("model_snapshots/mnist_cnn_model");
        std::cout << "Model loaded successfully." << std::endl;
        model.print_summary({1, 1, 28, 28});
        model.eval(); // Set model to evaluation mode

        // Load test data
        MNISTCNNDataLoader test_loader;
        if (!test_loader.load_data("./data/mnist_test.csv")) {
            return -1;
        }

        float total_accuracy = 0.0f;
        int num_batches = 0;
        int batch_size = 64;

        Tensor<float> batch_data, batch_labels;
        while (test_loader.get_batch(batch_size, batch_data, batch_labels)) {
            Tensor<float> predictions = model.predict(batch_data);
            total_accuracy += calculate_tensor_accuracy(predictions, batch_labels);
            num_batches++;
        }

        float avg_accuracy = total_accuracy / num_batches;
        std::cout << "\nTest Accuracy: " << std::fixed << std::setprecision(2) << avg_accuracy * 100 << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
