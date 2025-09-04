#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include "utils/cifar10_data_loader.hpp"
#include "utils/ops.hpp"
#include "utils/train.hpp"
#include "utils/misc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <sstream>
#include <vector>

namespace cifar10_constants {
constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 100;
constexpr int EPOCHS = 3;
constexpr size_t BATCH_SIZE = 32;
constexpr int LR_DECAY_INTERVAL = 10;
constexpr float LR_DECAY_FACTOR = 0.85f;
constexpr float LR_INITIAL = 0.005f;
} // namespace cifar10_constants

int main() {
  try {
    utils::set_num_threads(8);

    std::cout << "CIFAR-10 CNN Tensor<float> Neural Network Training"
              << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    data_loading::CIFAR10DataLoader<float> train_loader, test_loader;

    std::vector<std::string> train_files;
    for (int i = 1; i <= 5; ++i) {
      train_files.push_back("./data/cifar-10-batches-bin/data_batch_" +
                            std::to_string(i) + ".bin");
    }

    if (!train_loader.load_multiple_files(train_files)) {
      return -1;
    }

    if (!test_loader.load_multiple_files(
            {"./data/cifar-10-batches-bin/test_batch.bin"})) {
      return -1;
    }

    std::cout << "Successfully loaded training data: " << train_loader.size()
              << " samples" << std::endl;
    std::cout << "Successfully loaded test data: " << test_loader.size()
              << " samples" << std::endl;

    std::cout << "\nBuilding CNN model architecture for CIFAR-10..."
              << std::endl;

    auto model = tnn::SequentialBuilder<float>("cifar10_cnn_classifier")
                     .input({3, 32, 32})
                     .conv2d(16, 3, 3, 1, 1, 0, 0, "relu", true, "conv1")
                     .maxpool2d(3, 3, 3, 3, 0, 0, "maxpool1")
                     .conv2d(64, 3, 3, 1, 1, 0, 0, "relu", true, "conv2")
                     .maxpool2d(4, 4, 4, 4, 0, 0, "maxpool2")
                     .flatten("flatten")
                     .dense(10, "linear", true, "fc1")
                     .build();

    auto optimizer =
        std::make_unique<tnn::SGD<float>>(cifar10_constants::LR_INITIAL, 0.9f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = tnn::LossFactory<float>::create_crossentropy(
        cifar10_constants::EPSILON);
    model.set_loss_function(std::move(loss_function));

    model.enable_profiling(true);

    model.print_summary(std::vector<size_t>{
        cifar10_constants::BATCH_SIZE, 3, 32, 32});

    std::cout << "\nStarting CIFAR-10 CNN training..." << std::endl;
    train_classification_model(model, train_loader, test_loader, cifar10_constants::EPOCHS,
                    cifar10_constants::BATCH_SIZE,
                    cifar10_constants::LR_DECAY_FACTOR,
                    cifar10_constants::PROGRESS_PRINT_INTERVAL);

    std::cout
        << "\nCIFAR-10 CNN Tensor<float> model training completed successfully!"
        << std::endl;

    try {
      model.save_to_file("model_snapshots/cifar10_cnn_model");
      std::cout << "Model saved to: model_snapshots/cifar10_cnn_model"
                << std::endl;
    } catch (const std::exception &save_error) {
      std::cerr << "Warning: Failed to save model: " << save_error.what()
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
