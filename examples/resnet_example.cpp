/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "data_augmentation/augmentation.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "nn/blocks.hpp"
#include "nn/example_models.hpp"
#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"
#include <cmath>
#include <iostream>
#include <vector>

using namespace tnn;
using namespace std;

namespace resnet_constants {
constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 100;
constexpr int EPOCHS = 10;
constexpr size_t BATCH_SIZE = 32;
constexpr int LR_DECAY_INTERVAL = 5;
constexpr float LR_DECAY_FACTOR = 0.85f;
constexpr float LR_INITIAL = 0.01f;
} // namespace resnet_constants

int main() {
  try {
    // Load environment variables from .env file
    cout << "Loading environment variables..." << endl;
    if (!load_env_file("./.env")) {
      cout << "No .env file found, using default training parameters." << endl;
    }

    // Get training parameters from environment or use defaults
    const int epochs = get_env<int>("EPOCHS", resnet_constants::EPOCHS);
    const size_t batch_size = get_env<size_t>("BATCH_SIZE", resnet_constants::BATCH_SIZE);
    const float lr_initial = get_env<float>("LR_INITIAL", resnet_constants::LR_INITIAL);
    const float lr_decay_factor =
        get_env<float>("LR_DECAY_FACTOR", resnet_constants::LR_DECAY_FACTOR);
    const size_t lr_decay_interval =
        get_env<size_t>("LR_DECAY_INTERVAL", resnet_constants::LR_DECAY_INTERVAL);
    const int progress_print_interval =
        get_env<int>("PROGRESS_PRINT_INTERVAL", resnet_constants::PROGRESS_PRINT_INTERVAL);

    TrainingConfig train_config{epochs,
                                batch_size,
                                lr_decay_factor,
                                lr_decay_interval,
                                progress_print_interval,
                                DEFAULT_NUM_THREADS,
                                ProfilerType::NORMAL};

    train_config.print_config();

    // Load CIFAR-10 dataset
    CIFAR10DataLoader<float> train_loader, test_loader;

    vector<string> train_files;
    for (int i = 1; i <= 5; ++i) {
      train_files.push_back("./data/cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin");
    }

    if (!train_loader.load_multiple_files(train_files)) {
      cerr << "Failed to load training data!" << endl;
      return -1;
    }

    if (!test_loader.load_multiple_files({"./data/cifar-10-batches-bin/test_batch.bin"})) {
      cerr << "Failed to load test data!" << endl;
      return -1;
    }

    cout << "Successfully loaded training data: " << train_loader.size() << " samples" << endl;
    cout << "Successfully loaded test data: " << test_loader.size() << " samples" << endl;

    // Configure data augmentation for training
    cout << "\nConfiguring data augmentation for training..." << endl;
    auto aug_strategy = AugmentationBuilder<float>()
                            .horizontal_flip(0.25f)
                            .rotation(0.4f, 10.0f)
                            .brightness(0.3f, 0.15f)
                            .contrast(0.3f, 0.15f)
                            .gaussian_noise(0.3f, 0.05f)
                            .random_crop(0.4f, 4)
                            .build();
    train_loader.set_augmentation(std::move(aug_strategy));

    // Build ResNet model
    cout << "\nBuilding ResNet model architecture for CIFAR-10..." << endl;
    auto model = create_resnet18_cifar10(); // Use smaller ResNet - ResNet-18 is too deep

    model.print_summary({1, 3, 32, 32});
    model.initialize();

    // Set optimizer and loss function
    auto optimizer = make_unique<SGD<float>>(lr_initial, 0.9f);
    // auto optimizer = make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-7f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = LossFactory<float>::create_softmax_crossentropy();
    model.set_loss_function(std::move(loss_function));

    model.enable_profiling(true);

    // Train the model
    cout << "\nStarting ResNet training on CIFAR-10..." << endl;
    train_classification_model(model, train_loader, test_loader, train_config);

    cout << "\nResNet CIFAR-10 training completed successfully!" << endl;

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return -1;
  }

  return 0;
}
