#include "data_loading/cifar10_data_loader.hpp"
#include "nn/example_models.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

#include <cmath>
#include <iostream>

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

    create_cifar10_dataloader("./data", train_loader, test_loader);

    auto train_transform = AugmentationBuilder<float>()
                               .cutout(1.0f, 16)
                               .random_crop(1.0f, 4)
                               .horizontal_flip(0.5f)
                               //  .vertical_flip(0.5f)
                               //  .rotation(0.3f, 10.0f)
                               //  .brightness(0.3f, 0.15f)
                               //  .contrast(0.3f, 0.15f)
                               //  .gaussian_noise(1.0f, 0.1f)
                               .normalize({0.49139968, 0.48215827, 0.44653124},
                                          {0.24703233f, 0.24348505f, 0.26158768f})
                               .build();
    cout << "Configuring data transformation for training." << endl;
    train_loader.set_augmentation(std::move(train_transform));

    auto val_transform = AugmentationBuilder<float>()
                             .normalize({0.49139968, 0.48215827, 0.44653124},
                                        {0.24703233f, 0.24348505f, 0.26158768f})
                             .build();
    cout << "Configuring data normalization for test." << endl;
    test_loader.set_augmentation(std::move(val_transform));

    cout << "\nBuilding CNN model architecture for CIFAR-10..." << endl;

    auto model = create_resnet9_cifar10();

    model.set_device(device_type);
    model.initialize();

    auto optimizer = make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-7f, 1e-3f, true);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();
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
