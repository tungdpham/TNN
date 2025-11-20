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
using namespace std;

constexpr float LR_INITIAL = 0.005f;

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

    CIFAR10DataLoader<float> train_loader, test_loader;

    vector<string> train_files;
    for (int i = 1; i <= 5; ++i) {
      train_files.push_back("./data/cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin");
    }

    if (!train_loader.load_multiple_files(train_files)) {
      return -1;
    }

    if (!test_loader.load_multiple_files({"./data/cifar-10-batches-bin/test_batch.bin"})) {
      return -1;
    }

    cout << "Successfully loaded training data: " << train_loader.size() << " samples" << endl;
    cout << "Successfully loaded test data: " << test_loader.size() << " samples" << endl;

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

    cout << "\nBuilding CNN model architecture for CIFAR-10..." << endl;

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
                     .build();

    model.set_device(device_type);
    model.initialize();

    auto optimizer = make_unique<SGD<float>>(lr_initial, 0.9f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = LossFactory<float>::create_softmax_crossentropy();
    model.set_loss_function(std::move(loss_function));

    model.enable_profiling(true);

    cout << "\nStarting CIFAR-10 CNN training..." << endl;
    train_classification_model(model, train_loader, test_loader, train_config);

    cout << "\nCIFAR-10 CNN Tensor<float> model training completed successfully!" << endl;
  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return -1;
  }

  return 0;
}
