#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "data_augmentation/augmentation.hpp"
#include "data_loading/mnist_data_loader.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

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
  cin.tie(nullptr);
  try {

    // Load environment variables from .env file
    cout << "Loading environment variables..." << endl;
    if (!load_env_file("./.env")) {
      cout << "No .env file found, using default training parameters." << endl;
    }

    // Get training parameters from environment or use defaults
    const int epochs = get_env<int>("EPOCHS", mnist_constants::EPOCHS);
    const size_t batch_size = get_env<size_t>("BATCH_SIZE", mnist_constants::BATCH_SIZE);
    const float lr_initial = get_env<float>("LR_INITIAL", mnist_constants::LR_INITIAL);
    const float lr_decay_factor =
        get_env<float>("LR_DECAY_FACTOR", mnist_constants::LR_DECAY_FACTOR);
    const size_t lr_decay_interval =
        get_env<size_t>("LR_DECAY_INTERVAL", mnist_constants::LR_DECAY_INTERVAL);
    const int progress_print_interval =
        get_env<int>("PROGRESS_PRINT_INTERVAL", mnist_constants::PROGRESS_PRINT_INTERVAL);

    TrainingConfig train_config{epochs,
                                batch_size,
                                lr_decay_factor,
                                lr_decay_interval,
                                progress_print_interval,
                                DEFAULT_NUM_THREADS,
                                ProfilerType::NORMAL};

    train_config.print_config();

    MNISTDataLoader<float> train_loader, test_loader;

    if (!train_loader.load_data("./data/mnist/train.csv")) {
      cerr << "Failed to load training data!" << endl;
      return -1;
    }

    if (!test_loader.load_data("./data/mnist/test.csv")) {
      cerr << "Failed to load test data!" << endl;
      return -1;
    }

    cout << "Successfully loaded training data: " << train_loader.size() << " samples" << endl;
    cout << "Successfully loaded test data: " << test_loader.size() << " samples" << endl;

    cout << "\nBuilding CNN model architecture" << endl;

    auto aug_strategy =
        AugmentationBuilder<float>().contrast(0.3f, 0.15f).gaussian_noise(0.3f, 0.05f).build();
    train_loader.set_augmentation(std::move(aug_strategy));

    auto model = SequentialBuilder<float>("mnist_cnn_model")
                     .input({1, ::mnist_constants::IMAGE_HEIGHT, ::mnist_constants::IMAGE_WIDTH})
                     .conv2d(8, 3, 3, 1, 1, 0, 0, true, "conv1")
                     .batchnorm(1e-5f, 0.1f, true, "bn1")
                     .activation("relu", "relu1")
                     .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")
                     //  .avgpool2d(3, 3, 3, 3, 0, 0, "pool1")
                     .conv2d(16, 1, 1, 1, 1, 0, 0, true, "conv2_1x1")
                     .batchnorm(1e-5f, 0.1f, true, "bn2")
                     .activation("relu", "relu2")
                     .conv2d(48, 3, 3, 1, 1, 0, 0, true, "conv3")
                     .batchnorm(1e-5f, 0.1f, true, "bn3")
                     .activation("relu", "relu3")
                     .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                     .flatten("flatten")
                     .dense(::mnist_constants::NUM_CLASSES, "linear", true, "output")
                     .build();

    model.set_device("GPU:0");
    model.initialize();

    // auto optimizer = make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-8f);
    auto optimizer = make_unique<SGD<float>>(lr_initial, 0.9f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = LossFactory<float>::create_softmax_crossentropy();
    model.set_loss_function(std::move(loss_function));

    model.print_config();
    train_classification_model(model, train_loader, test_loader, train_config);
  } catch (const exception &e) {
    cerr << "Error during training: " << e.what() << endl;
    return -1;
  } catch (...) {
    cerr << "Unknown error occurred during training!" << endl;
    return -1;
  }

  return 0;
}
