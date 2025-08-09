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
#include "nn/layers.hpp"
#include "nn/optimizers.hpp"
#include "tensor/tensor.hpp"
#include "pipeline_experimental/pipeline_stage.hpp"
#include "pipeline_experimental/pipeline_orchestrator.hpp"
#include "pipeline_experimental/communication.hpp"
#include "utils/mnist_data_loader.hpp"
#include "nn/loss.hpp"  

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
    data_loading::MNISTDataLoader<float>& train_loader,
    data_loading::MNISTDataLoader<float>& test_loader,
    int epochs = 10,
    int batch_size = 32,
    float learning_rate = 0.001) {

    // Create a separate optimizer for each stage
    std::vector<std::unique_ptr<layers::Optimizer<float>>> optimizers;
    for (size_t i = 0; i < orchestrator.get_stages().size(); ++i) {
        optimizers.push_back(std::make_unique<layers::Adam<float>>(learning_rate));
    }

    std::cout << "Starting Pipeline CNN model training..." << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size
              << ", Learning rate: " << learning_rate << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        train_loader.shuffle();
        train_loader.reset();

        float total_loss = 0.0;
        float total_accuracy = 0.0;
        int num_batches = 0;

        Tensor<float> batch_data, batch_labels;
        while (train_loader.get_batch(batch_size, batch_data, batch_labels)) {
            auto [loss, acc] = orchestrator.train_batch(batch_data, batch_labels);
            total_loss += loss * batch_data.batch_size();
            total_accuracy += acc * batch_data.batch_size();
            
            // Apply gradients using the dedicated optimizer for each stage
            const auto& stages = orchestrator.get_stages();
            for (size_t i = 0; i < stages.size(); ++i) {
                stages[i]->apply_gradients(*optimizers[i]);
            }
            
            num_batches++;
            if(num_batches % 50 == 0) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Batch "
                  << num_batches << ", Loss: " << std::fixed
                  << std::setprecision(4) << loss
                  << ", Acc: " << std::setprecision(3) << acc * 100 << "%"
                  << std::endl;
            }
        }

        float avg_train_loss = total_loss / train_loader.size();
        float avg_train_accuracy = total_accuracy / train_loader.size();

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);

        std::cout << std::string(70, '-') << std::endl;
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed in "
                  << epoch_duration.count() << "s" << std::endl;
        std::cout << "Training   - Loss: " << std::fixed << std::setprecision(4)
              << avg_train_loss << ", Accuracy: " << std::setprecision(2)
              << avg_train_accuracy * 100 << "%" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }
}


int main() {
    try {
        omp_set_num_threads(4); // Set OpenMP threads if needed
        std::cout << "MNIST CNN Pipeline Training" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        // Load data
        data_loading::MNISTDataLoader<float> train_loader, test_loader;
        train_loader.load_data("./data/mnist/train.csv");
        test_loader.load_data("./data/mnist/test.csv");
        
        // Stage 1
        auto stage1 = std::make_unique<pipeline::PipelineStage<float>>(0, 0, 4); 
        stage1->add_layer(Layers::conv2d<float>(1, 8, 5, 5, 1, 1, 0, 0, "relu", true, "C1"));
        stage1->add_layer(Layers::maxpool2d<float>(3, 3, 3, 3, 0, 0, "P1"));
        stage1->add_layer(Layers::conv2d<float>(8, 16, 1, 1, 1, 1, 0, 0, "relu", true, "C2_1x1"));
        stage1->add_layer(Layers::conv2d<float>(16, 48, 5, 5, 1, 1, 0, 0, "relu", true, "C3"));

        // Stage 2
        auto stage2 = std::make_unique<pipeline::PipelineStage<float>>(1, 1, 4);
        stage2->add_layer(Layers::maxpool2d<float>(2, 2, 2, 2, 0, 0, "P2"));
        stage2->add_layer(Layers::dense<float>(48 * 2 * 2, 10, "linear", true, "output"));
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
