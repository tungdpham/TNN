#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "data_loading/cifar100_data_loader.hpp"
#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "tensor/tensor.hpp"
#include "utils/env.hpp"
#include "utils/ops.hpp"

using namespace tnn;
using namespace data_loading;
using namespace ::utils;

namespace cifar100_constants {
constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 50;
constexpr int EPOCHS = 50;
constexpr size_t BATCH_SIZE = 64;
constexpr int LR_DECAY_INTERVAL = 15;
constexpr float LR_DECAY_FACTOR = 0.5f;
constexpr float LR_INITIAL = 0.001f;
} // namespace cifar100_constants

int main() {
  try {
    // Load environment variables from .env file
    std::cout << "Loading environment variables..." << std::endl;
    if (!load_env_file("./.env")) {
      std::cout << "No .env file found, using default training parameters." << std::endl;
    }

    // Get training parameters from environment or use defaults
    const int epochs = get_env<int>("EPOCHS", cifar100_constants::EPOCHS);
    const size_t batch_size = get_env<size_t>("BATCH_SIZE", cifar100_constants::BATCH_SIZE);
    const float lr_initial = get_env<float>("LR_INITIAL", cifar100_constants::LR_INITIAL);
    const float lr_decay_factor =
        get_env<float>("LR_DECAY_FACTOR", cifar100_constants::LR_DECAY_FACTOR);
    const size_t lr_decay_interval =
        get_env<size_t>("LR_DECAY_INTERVAL", cifar100_constants::LR_DECAY_INTERVAL);
    const int progress_print_interval =
        get_env<int>("PROGRESS_PRINT_INTERVAL", cifar100_constants::PROGRESS_PRINT_INTERVAL);

    TrainingConfig train_config{epochs,
                                batch_size,
                                lr_decay_factor,
                                lr_decay_interval,
                                progress_print_interval,
                                DEFAULT_NUM_THREADS,
                                ProfilerType::NORMAL};

    train_config.print_config();

    data_loading::CIFAR100DataLoader<float> train_loader, test_loader;

    if (!train_loader.load_data("./data/cifar-100-binary/train.bin")) {
      return -1;
    }

    if (!test_loader.load_data("./data/cifar-100-binary/test.bin")) {
      return -1;
    }

    std::cout << "Successfully loaded training data: " << train_loader.size() << " samples"
              << std::endl;
    std::cout << "Successfully loaded test data: " << test_loader.size() << " samples" << std::endl;

    std::cout << "\nBuilding CNN model architecture for CIFAR-100..." << std::endl;

    auto model = tnn::SequentialBuilder<float>("cifar100_cnn_classifier")
                     .input({3, 32, 32})
                     .conv2d(32, 3, 3, 1, 1, 0, 0, true, "conv1")
                     .activation("relu", "relu1")
                     .conv2d(64, 3, 3, 1, 1, 0, 0, true, "conv2")
                     .activation("relu", "relu2")
                     .conv2d(128, 5, 5, 1, 1, 0, 0, true, "conv2_1")
                     .activation("relu", "relu2_1")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool1")
                     .conv2d(256, 3, 3, 1, 1, 0, 0, true, "conv3")
                     .activation("relu", "relu3")
                     .conv2d(256, 3, 3, 1, 1, 0, 0, true, "conv4")
                     .activation("relu", "relu4")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                     .flatten("flatten")
                     .dense(512, "relu", true, "fc1")
                     .batchnorm(1e-5f, 0.1f, true, "bn1")
                     .dense(100, "linear", true, "output")
                     .activation("softmax", "softmax_output")
                     .build();

    auto optimizer = std::make_unique<tnn::Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-8f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = tnn::LossFactory<float>::create_crossentropy(cifar100_constants::EPSILON);
    model.set_loss_function(std::move(loss_function));

    model.enable_profiling(true);

    std::cout << "\nStarting CIFAR-100 CNN training..." << std::endl;
    train_classification_model(model, train_loader, test_loader, train_config);

    std::cout << "\nCIFAR-100 CNN Tensor<float> model training completed "
                 "successfully!"
              << std::endl;

    try {
      model.save_to_file("model_snapshots/cifar100_cnn_model");
      std::cout << "Model saved to: model_snapshots/cifar100_cnn_model" << std::endl;
    } catch (const std::exception &save_error) {
      std::cerr << "Warning: Failed to save model: " << save_error.what() << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
