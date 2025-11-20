#include "data_loading/cifar10_data_loader.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

#include <cmath>
#include <fstream>
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

    auto aug_strategy = AugmentationBuilder<float>()
                            .horizontal_flip(0.25f)
                            .rotation(0.3f, 10.0f)
                            .brightness(0.3f, 0.15f)
                            .contrast(0.3f, 0.15f)
                            .gaussian_noise(0.3f, 0.05f)
                            .random_crop(0.4f, 4)
                            .build();
    cout << "Configuring data augmentation for training." << endl;
    train_loader.set_augmentation(std::move(aug_strategy));

    cout << "\nBuilding CNN model architecture for CIFAR-10..." << endl;

    auto model = SequentialBuilder<float>("cifar10_cnn_classifier_v2")
                     .input({3, 32, 32})
                     .conv2d(64, 3, 3, 1, 1, 1, 1, false, "conv0")
                     .batchnorm(1e-5f, 0.1f, true, "bn0")
                     .activation("relu", "relu0")
                     .conv2d(64, 3, 3, 1, 1, 1, 1, false, "conv1")
                     .batchnorm(1e-5f, 0.1f, true, "bn1")
                     .activation("relu", "relu1")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool0")
                     .conv2d(128, 3, 3, 1, 1, 1, 1, false, "conv2")
                     .batchnorm(1e-5f, 0.1f, true, "bn2")
                     .activation("relu", "relu2")
                     .conv2d(128, 3, 3, 1, 1, 1, 1, false, "conv3")
                     .batchnorm(1e-5f, 0.1f, true, "bn3")
                     .activation("relu", "relu3")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool1")
                     .conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv4")
                     .batchnorm(1e-5f, 0.1f, true, "bn5")
                     .activation("relu", "relu5")
                     .conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv5")
                     .activation("relu", "relu6")
                     .conv2d(256, 3, 3, 1, 1, 1, 1, false, "conv6")
                     .batchnorm(1e-5f, 0.1f, true, "bn6")
                     .activation("relu", "relu6")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                     .conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv7")
                     .batchnorm(1e-5f, 0.1f, true, "bn8")
                     .activation("relu", "relu7")
                     .conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv8")
                     .batchnorm(1e-5f, 0.1f, true, "bn9")
                     .activation("relu", "relu8")
                     .conv2d(512, 3, 3, 1, 1, 1, 1, false, "conv9")
                     .batchnorm(1e-5f, 0.1f, true, "bn10")
                     .activation("relu", "relu9")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool3")
                     .flatten("flatten")
                     .dense(512, "linear", true, "fc0")
                     .activation("relu", "relu10")
                     .dense(10, "linear", true, "fc1")
                     .build();
    model.set_device(device_type);
    model.initialize();

    auto optimizer = make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-8f);
    model.set_optimizer(std::move(optimizer));

    // auto loss_function =
    // LossFactory<float>::create_crossentropy(cifar10_constants::EPSILON);
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
