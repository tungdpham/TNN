#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "data_augmentation/augmentation.hpp"
#include "data_loading/mnist_data_loader.hpp"
#include "nn/example_models.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

constexpr float LR_INITIAL = 0.01f;

int main() {
  cin.tie(nullptr);
  try {

    // Load environment variables from .env file
    cout << "Loading environment variables..." << endl;
    if (!load_env_file("./.env")) {
      cout << "No .env file found, using default training parameters." << endl;
    }

    string device_type_str = get_env<string>("DEVICE_TYPE", "CPU");

    float lr_initial = get_env<float>("LR_INITIAL", LR_INITIAL);
    DeviceType device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;

    cout << "Using device type: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;

    TrainingConfig train_config;
    train_config.load_from_env();

    train_config.print_config();

    MNISTDataLoader<float> train_loader, test_loader;

    create_mnist_data_loaders("./data", train_loader, test_loader);

    cout << "Successfully loaded training data: " << train_loader.size() << " samples" << endl;
    cout << "Successfully loaded test data: " << test_loader.size() << " samples" << endl;

    cout << "\nBuilding CNN model architecture" << endl;

    auto aug_strategy =
        AugmentationBuilder<float>().contrast(0.3f, 0.15f).gaussian_noise(0.3f, 0.05f).build();
    train_loader.set_augmentation(std::move(aug_strategy));

    auto model = create_mnist_trainer();

    model.set_device(device_type);
    model.initialize();

    auto optimizer = make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-8f);

    auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();

    train_classification_model(model, train_loader, test_loader, std::move(optimizer),
                               std::move(loss_function), train_config);
  } catch (const exception &e) {
    cerr << "Error during training: " << e.what() << endl;
    return -1;
  } catch (...) {
    cerr << "Unknown error occurred during training!" << endl;
    return -1;
  }

  return 0;
}
