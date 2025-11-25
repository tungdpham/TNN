#include "data_loading/tiny_imagenet_data_loader.hpp"
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

constexpr float LR_INITIAL = 0.0001f;

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

    cout << "Using learning rate: " << lr_initial << endl;

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

    cout << "Successfully loaded training data: " << train_loader.size() << " samples" << endl;
    cout << "Successfully loaded validation data: " << val_loader.size() << " samples" << endl;

    auto train_aug = AugmentationBuilder<float>()
                         .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                         .horizontal_flip(0.2)
                         .rotation(0.2f, 5.0f)
                         .random_crop(0.25f, 4)
                         .build();
    cout << "Configuring data augmentation for training." << endl;
    train_loader.set_augmentation(std::move(train_aug));

    auto val_aug = AugmentationBuilder<float>()
                       .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                       .build();
    cout << "Configuring data normalization for validation." << endl;
    val_loader.set_augmentation(std::move(val_aug));

    cout << "\nBuilding CNN model architecture for Tiny ImageNet..." << endl;

    auto model = create_resnet18_tiny_imagenet();

    model.set_device(device_type);
    model.initialize();

    // Use slightly higher epsilon for better numerical stability
    auto optimizer = make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-7f);
    // auto optimizer = make_unique<SGD<float>>(lr_initial, 0.9f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();
    model.set_loss_function(std::move(loss_function));

    model.enable_profiling(true);

    cout << "\nStarting Tiny ImageNet CNN training..." << endl;
    train_classification_model(model, train_loader, val_loader, train_config);

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return -1;
  }

  return 0;
}
