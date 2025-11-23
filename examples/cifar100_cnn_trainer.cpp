#include <cmath>
#include <iostream>
#include <vector>

#include "data_loading/cifar100_data_loader.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 50;
constexpr int EPOCHS = 50;
constexpr size_t BATCH_SIZE = 64;
constexpr int LR_DECAY_INTERVAL = 15;
constexpr float LR_DECAY_FACTOR = 0.5f;
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

    CIFAR100DataLoader<float> train_loader, test_loader;

    create_cifar100_dataloader("./data", train_loader, test_loader);

    auto model = SequentialBuilder<float>("cifar100_cnn_classifier")
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
                     .dense(512, true, "fc0")
                     .activation("relu", "relu10")
                     .dense(10, true, "fc1")
                     .build();

    model.set_device(device_type);
    model.initialize();

    auto optimizer = make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-8f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = LossFactory<float>::create_crossentropy(::EPSILON);
    model.set_loss_function(std::move(loss_function));

    model.enable_profiling(true);

    cout << "\nStarting CIFAR-100 CNN training..." << endl;
    train_classification_model(model, train_loader, test_loader, train_config);

    cout << "\nCIFAR-100 CNN Tensor<float> model training completed "
            "successfully!"
         << endl;

    try {
      model.save_to_file("model_snapshots/cifar100_cnn_model");
      cout << "Model saved to: model_snapshots/cifar100_cnn_model" << endl;
    } catch (exception &save_error) {
      cerr << "Warning: Failed to save model: " << save_error.what() << endl;
    }
  } catch (exception &e) {
    cerr << "Error: " << e.what() << endl;
    return -1;
  }

  return 0;
}
