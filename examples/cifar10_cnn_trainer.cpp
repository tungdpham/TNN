#include "data_augmentation/augmentation.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace tnn;

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
    // Load environment variables from .env file
    std::cout << "Loading environment variables..." << std::endl;
    if (!load_env_file("./.env")) {
      std::cout << "No .env file found, using default training parameters." << std::endl;
    }

    // Get training parameters from environment or use defaults
    const int epochs = get_env<int>("EPOCHS", cifar10_constants::EPOCHS);
    const size_t batch_size = get_env<size_t>("BATCH_SIZE", cifar10_constants::BATCH_SIZE);
    const float lr_initial = get_env<float>("LR_INITIAL", cifar10_constants::LR_INITIAL);
    const float lr_decay_factor =
        get_env<float>("LR_DECAY_FACTOR", cifar10_constants::LR_DECAY_FACTOR);
    const size_t lr_decay_interval =
        get_env<size_t>("LR_DECAY_INTERVAL", cifar10_constants::LR_DECAY_INTERVAL);
    const int progress_print_interval =
        get_env<int>("PROGRESS_PRINT_INTERVAL", cifar10_constants::PROGRESS_PRINT_INTERVAL);

    TrainingConfig train_config{epochs,
                                batch_size,
                                lr_decay_factor,
                                lr_decay_interval,
                                progress_print_interval,
                                DEFAULT_NUM_THREADS,
                                ProfilerType::NORMAL};

    train_config.print_config();

    CIFAR10DataLoader<float> train_loader, test_loader;

    std::vector<std::string> train_files;
    for (int i = 1; i <= 5; ++i) {
      train_files.push_back("./data/cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin");
    }

    if (!train_loader.load_multiple_files(train_files)) {
      return -1;
    }

    if (!test_loader.load_multiple_files({"./data/cifar-10-batches-bin/test_batch.bin"})) {
      return -1;
    }

    std::cout << "Successfully loaded training data: " << train_loader.size() << " samples"
              << std::endl;
    std::cout << "Successfully loaded test data: " << test_loader.size() << " samples" << std::endl;

    std::cout << "\nConfiguring data augmentation for training..." << std::endl;

    auto aug_strategy = AugmentationBuilder<float>()
                            .horizontal_flip(0.25f)
                            .rotation(0.4f, 10.0f)
                            .brightness(0.3f, 0.15f)
                            .contrast(0.3f, 0.15f)
                            .gaussian_noise(0.3f, 0.05f)
                            .build();
    train_loader.set_augmentation(std::move(aug_strategy));

    std::cout << "\nBuilding CNN model architecture for CIFAR-10..." << std::endl;

    auto model = SequentialBuilder<float>("cifar10_cnn_classifier_v1")
                     .input({3, 32, 32})
                     .conv2d(16, 3, 3, 1, 1, 0, 0, true, "conv1")
                     .batchnorm(1e-5f, 0.1f, true, "bn1")
                     .activation("relu", "relu1")
                     .maxpool2d(3, 3, 3, 3, 0, 0, "maxpool1")
                     .conv2d(64, 3, 3, 1, 1, 0, 0, true, "conv2")
                     .batchnorm(1e-5f, 0.1f, true, "bn2")
                     .activation("relu", "relu2")
                     .maxpool2d(4, 4, 4, 4, 0, 0, "maxpool2")
                     .flatten("flatten")
                     .dense(10, "linear", true, "fc1")
                     //  .activation("softmax", "softmax_output")
                     .build();

    model.initialize();

    auto optimizer = std::make_unique<SGD<float>>(lr_initial, 0.9f);
    model.set_optimizer(std::move(optimizer));

    // auto loss_function =
    // LossFactory<float>::create_crossentropy(cifar10_constants::EPSILON);
    auto loss_function = LossFactory<float>::create_softmax_crossentropy();
    model.set_loss_function(std::move(loss_function));

    model.enable_profiling(true);

    std::cout << "\nStarting CIFAR-10 CNN training..." << std::endl;
    train_classification_model(model, train_loader, test_loader, train_config);

    std::cout << "\nCIFAR-10 CNN Tensor<float> model training completed successfully!" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
