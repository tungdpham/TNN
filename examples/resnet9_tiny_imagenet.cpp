/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "data_augmentation/augmentation.hpp"
#include "data_loading/tiny_imagenet_data_loader.hpp"
#include "device/device_type.hpp"
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

constexpr float LR_INITIAL = 0.01f; // Higher initial LR for smaller model

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
    std::cout << "Using device type: " << (device_type == DeviceType::CPU ? "CPU" : "GPU")
              << std::endl;

    TrainingConfig train_config;
    train_config.load_from_env();

    train_config.print_config();

    // Create data loaders
    TinyImageNetDataLoader<float> train_loader;
    TinyImageNetDataLoader<float> val_loader;

    // Path to Tiny ImageNet dataset
    std::string dataset_path = "data/tiny-imagenet-200";

    // Load training data
    std::cout << "\nLoading training data..." << std::endl;
    if (!train_loader.load_data(dataset_path, true)) {
      std::cerr << "Failed to load training data!" << std::endl;
      return 1;
    }

    // Load validation data
    std::cout << "\nLoading validation data..." << std::endl;
    if (!val_loader.load_data(dataset_path, false)) {
      std::cerr << "Failed to load validation data!" << std::endl;
      return 1;
    }

    // Configure data augmentation for training (lighter augmentation)
    cout << "\nConfiguring data augmentation for training..." << endl;
    auto aug_strategy =
        AugmentationBuilder<float>().horizontal_flip(0.5f).random_crop(0.3f, 4).build();
    train_loader.set_augmentation(std::move(aug_strategy));

    // Use ResNet-9 instead of ResNet-18 for faster training
    auto model = create_resnet9_tiny_imagenet();

    model.print_summary({64, 3, 64, 64});

    model.set_device(device_type);
    model.initialize();

    // Set optimizer and loss function
    auto optimizer = make_unique<SGD<float>>(lr_initial, 0.9f);
    // auto optimizer = make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-8f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = LossFactory<float>::create_softmax_crossentropy();
    model.set_loss_function(std::move(loss_function));

    model.enable_profiling(true);

    // Train the model
    cout << "\nStarting ResNet-9 training on Tiny ImageNet..." << endl;
    train_classification_model(model, train_loader, val_loader, train_config);

    cout << "\nResNet-9 Tiny ImageNet training completed successfully!" << endl;

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return -1;
  }

  return 0;
}
