#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string_view>
#include <vector>

#include "data_augmentation/augmentation.hpp"
#include "data_loading/mnist_data_loader.hpp"
#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "tensor/tensor.hpp"
#include "utils/misc.hpp"
#include "utils/ops.hpp"

namespace mnist_constants {

constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 100;
constexpr int EPOCHS = 5;
constexpr size_t BATCH_SIZE = 64;
constexpr int LR_DECAY_INTERVAL = 2;
constexpr float LR_DECAY_FACTOR = 0.8f;
constexpr float LR_INITIAL = 0.01f;

} // namespace mnist_constants

int main() {
  std::cout.tie(nullptr);
  std::cin.tie(nullptr);
  std::ios::sync_with_stdio(false);
  try {

    data_loading::MNISTDataLoader<float> train_loader, test_loader;

    if (!train_loader.load_data("./data/mnist/train.csv")) {
      std::cerr << "Failed to load training data!" << std::endl;
      return -1;
    }

    if (!test_loader.load_data("./data/mnist/test.csv")) {
      std::cerr << "Failed to load test data!" << std::endl;
      return -1;
    }

    std::cout << "Successfully loaded training data: " << train_loader.size() << " samples"
              << std::endl;
    std::cout << "Successfully loaded test data: " << test_loader.size() << " samples" << std::endl;

    std::cout << "\nBuilding CNN model architecture with automatic shape inference..." << std::endl;

    auto aug_strategy = data_augmentation::AugmentationBuilder<float>()
                            .contrast(0.3f, 0.15f)
                            .gaussian_noise(0.3f, 0.05f)
                            .build();
    train_loader.set_augmentation(std::move(aug_strategy));

    auto model = tnn::SequentialBuilder<float>("mnist_cnn_model")
                     .input({1, ::mnist_constants::IMAGE_HEIGHT, ::mnist_constants::IMAGE_WIDTH})
                     .conv2d(8, 5, 5, 1, 1, 0, 0, true, "conv1")
                     .batchnorm(1e-5f, 0.1f, true, "bn1")
                     .activation("relu", "relu1")
                     .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")
                     .conv2d(16, 1, 1, 1, 1, 0, 0, true, "conv2_1x1")
                     .batchnorm(1e-5f, 0.1f, true, "bn2")
                     .activation("relu", "relu2")
                     .conv2d(48, 5, 5, 1, 1, 0, 0, true, "conv3")
                     .batchnorm(1e-5f, 0.1f, true, "bn3")
                     .activation("relu", "relu3")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                     .flatten("flatten")
                     .dense(::mnist_constants::NUM_CLASSES, "linear", true, "output")
                     .activation("softmax", "softmax_output")
                     .build();

    auto optimizer =
        std::make_unique<tnn::Adam<float>>(mnist_constants::LR_INITIAL, 0.9f, 0.999f, 1e-8f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = tnn::LossFactory<float>::create_crossentropy(mnist_constants::EPSILON);
    model.set_loss_function(std::move(loss_function));

    model.print_config();
    train_classification_model(
        model, train_loader, test_loader,
        {mnist_constants::EPOCHS, mnist_constants::BATCH_SIZE, mnist_constants::LR_DECAY_FACTOR,
         mnist_constants::LR_DECAY_INTERVAL, mnist_constants::PROGRESS_PRINT_INTERVAL,
         DEFAULT_NUM_THREADS, ProfilerType::NORMAL});
  } catch (const std::exception &e) {
    std::cerr << "Error during training: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "Unknown error occurred during training!" << std::endl;
    return -1;
  }

  return 0;
}
