/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "data_augmentation/augmentation.hpp"
#include "data_loading/cifar10_data_loader.hpp"
#include "nn/example_models.hpp"
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

constexpr float LR_INITIAL = 0.001f;

int main() {
  try {
    // Load environment variables from .env file
    cout << "Loading environment variables..." << endl;
    if (!load_env_file("./.env")) {
      cout << "No .env file found, using default training parameters." << endl;
    }

    string device_type_str = get_env<string>("DEVICE_TYPE", "CPU");

    float lr_initial = get_env<float>("LR_INITIAL", LR_INITIAL);
    DeviceType device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;

    TrainingConfig train_config;
    train_config.load_from_env();

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

    auto model = create_resnet18_cifar10();

    model.print_summary({1, 3, 32, 32});

    model.set_device(device_type);
    model.initialize();

    // Set optimizer and loss function
    auto optimizer = make_unique<SGD<float>>(lr_initial, 0.9f);
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
