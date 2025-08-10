#include <iostream>
#include <vector>
#include <string>
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

// Data loader for MNIST CSV, adapted for testing
class MNISTDataLoader {
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

        int actual_batch_size = std::min(batch_size, (int)(data_.size() - current_index_));
        batch_data = Tensor<float>({(size_t)actual_batch_size, 1, 28, 28});
        batch_labels = Tensor<float>({(size_t)actual_batch_size, 10, 1, 1});
        batch_labels.fill(0.0f);

        for (int i = 0; i < actual_batch_size; ++i) {
            for (int j = 0; j < 784; ++j) {
                batch_data(i, 0, j / 28, j % 28) = data_[current_index_ + i][j];
            }
            batch_labels(i, labels_[current_index_ + i], 0, 0) = 1.0f;
        }

        current_index_ += actual_batch_size;
        return true;
    }

    void reset() {
        current_index_ = 0;
    }

    size_t get_num_samples() const {
        return data_.size();
    }
};

void run_test() {
    // Load the trained model
    tnn::Sequential<float> model;
    try {
        model = tnn::Sequential<float>::from_file("model_snapshots/mnist_cnn_model");
        std::cout << "Model loaded successfully from model_snapshots/mnist_cnn_model\n";
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return;
    }

    model.print_config();
    // Set the model to evaluation mode
    model.eval();

    // Load the test data
    MNISTDataLoader loader;
    if (!loader.load_data("data/mnist/test.csv")) {
        return;
    }

    int batch_size = 100;
    int correct_predictions = 0;
    int total_samples = 0;

    Tensor<float> batch_data, batch_labels;
    while (loader.get_batch(batch_size, batch_data, batch_labels)) {
        Tensor<float> predictions = model.predict(batch_data);

        for (size_t i = 0; i < predictions.batch_size(); ++i) {
            int predicted_label = predictions.argmax_channel(i, 0, 0);
            int true_label = batch_labels.argmax_channel(i, 0, 0);

            if (predicted_label == true_label) {
                correct_predictions++;
            }
        }
        total_samples += predictions.batch_size();
    }

    double accuracy = (double)correct_predictions / total_samples;
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100 << "%" << std::endl;
}

int main() {
    try {
        run_test();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred during testing: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
