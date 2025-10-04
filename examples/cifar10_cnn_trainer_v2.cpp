#include "data_loading/cifar10_data_loader.hpp"
#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "tensor/tensor.hpp"
#include "utils/ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using namespace tnn;
using namespace data_loading;
using namespace data_augmentation;

namespace cifar10_constants {
constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 100;
constexpr int EPOCHS = 40;
constexpr size_t BATCH_SIZE = 32;
constexpr int LR_DECAY_INTERVAL = 3;
constexpr float LR_DECAY_FACTOR = 0.85f;
constexpr float LR_INITIAL = 0.001f;
} // namespace cifar10_constants

int main() {
  try {
    std::cout << "CIFAR-10 CNN Tensor<float> Neural Network Training" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

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

    auto aug_strategy = AugmentationBuilder<float>()
                            .horizontal_flip(0.25f)
                            .rotation(0.3f, 10.0f)
                            .brightness(0.3f, 0.15f)
                            .contrast(0.3f, 0.15f)
                            .gaussian_noise(0.3f, 0.05f)
                            .build();
    std::cout << "Configuring data augmentation for training." << std::endl;
    train_loader.set_augmentation(std::move(aug_strategy));

    std::cout << "\nBuilding CNN model architecture for CIFAR-10..." << std::endl;

    auto model = SequentialBuilder<float>("cifar10_cnn_classifier_v2")
                     .input({3, 32, 32})
                     .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv0")
                     .batchnorm(1e-5f, 0.1f, true, "bn0")
                     .activation("relu", "relu0")
                     .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv1")
                     .batchnorm(1e-5f, 0.1f, true, "bn1")
                     .activation("relu", "relu1")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool0")
                     .conv2d(128, 3, 3, 1, 1, 1, 1, true, "conv2")
                     .batchnorm(1e-5f, 0.1f, true, "bn2")
                     .activation("relu", "relu2")
                     .conv2d(128, 3, 3, 1, 1, 1, 1, true, "conv3")
                     .batchnorm(1e-5f, 0.1f, true, "bn3")
                     .activation("relu", "relu3")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool1")
                     .conv2d(256, 3, 3, 1, 1, 1, 1, true, "conv4")
                     .batchnorm(1e-5f, 0.1f, true, "bn5")
                     .activation("relu", "relu5")
                     .conv2d(256, 3, 3, 1, 1, 1, 1, true, "conv5")
                     .activation("relu", "relu6")
                     .conv2d(256, 3, 3, 1, 1, 1, 1, true, "conv6")
                     .batchnorm(1e-5f, 0.1f, true, "bn6")
                     .activation("relu", "relu6")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                     .conv2d(512, 3, 3, 1, 1, 1, 1, true, "conv7")
                     .batchnorm(1e-5f, 0.1f, true, "bn8")
                     .activation("relu", "relu7")
                     .conv2d(512, 3, 3, 1, 1, 1, 1, true, "conv8")
                     .batchnorm(1e-5f, 0.1f, true, "bn9")
                     .activation("relu", "relu8")
                     .conv2d(512, 3, 3, 1, 1, 1, 1, true, "conv9")
                     .batchnorm(1e-5f, 0.1f, true, "bn10")
                     .activation("relu", "relu9")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool3")
                     .flatten("flatten")
                     .dense(512, "linear", true, "fc0")
                     .activation("relu", "relu10")
                     .dense(10, "linear", true, "fc1")
                     .build();
    auto optimizer =
        std::make_unique<tnn::Adam<float>>(cifar10_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = tnn::LossFactory<float>::create_crossentropy(cifar10_constants::EPSILON);
    model.set_loss_function(std::move(loss_function));

    // Load pre-trained weights if available
    const std::string pretrained_weights_file = "./model_snapshots/" + model.name() + ".bin";
    if (std::ifstream(pretrained_weights_file)) {
      std::cout << "\nLoading pre-trained model weights from " << pretrained_weights_file << " ..."
                << std::endl;
      model.load_weights_file(pretrained_weights_file);
      std::cout << "Successfully loaded pre-trained model weights." << std::endl;
    } else {
      std::cout << "\nNo pre-trained weights file found. Training model from scratch." << std::endl;
    }

    model.enable_profiling(true);

    std::cout << "\nStarting CIFAR-10 CNN training..." << std::endl;
    train_classification_model(
        model, train_loader, test_loader,
        {cifar10_constants::EPOCHS, cifar10_constants::BATCH_SIZE,
         cifar10_constants::LR_DECAY_FACTOR, cifar10_constants::LR_DECAY_INTERVAL,
         cifar10_constants::PROGRESS_PRINT_INTERVAL, DEFAULT_NUM_THREADS, ProfilerType::NORMAL});

    std::cout << "\nCIFAR-10 CNN Tensor<float> model training completed successfully!" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
