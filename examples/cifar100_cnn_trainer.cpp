#include <cmath>
#include <iostream>
#include <vector>

#include "data_loading/cifar100_data_loader.hpp"
#include "nn/example_models.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

constexpr float EPSILON = 1e-15f;
constexpr float LR_INITIAL = 0.001f;

int main() {
  try {
    string device_type_str = Env::get<string>("DEVICE_TYPE", "CPU");

    float lr_initial = Env::get<float>("LR_INITIAL", LR_INITIAL);
    DeviceType device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;

    TrainingConfig train_config;
    train_config.load_from_env();

    train_config.print_config();

    CIFAR100DataLoader<float> train_loader, test_loader;

    create_cifar100_dataloader("./data", train_loader, test_loader);

    auto train_transform = AugmentationBuilder<float>()
                               .random_crop(0.5f, 4)
                               .horizontal_flip(0.5f)
                               .cutout(0.5f, 4)
                               .gaussian_noise(0.5f, 0.05f)
                               .build();
    cout << "Configuring data transformation for training." << endl;
    train_loader.set_augmentation(std::move(train_transform));

    auto val_transform = AugmentationBuilder<float>().build();
    cout << "Configuring data normalization for test." << endl;
    test_loader.set_augmentation(std::move(val_transform));

    auto model = create_wrn16_8_cifar100();

    model.set_device(device_type);
    model.initialize();

    auto optimizer = OptimizerFactory<float>::create_adam(lr_initial, 0.9f, 0.999f, 1e-8f);

    auto loss_function = LossFactory<float>::create_crossentropy(::EPSILON);

    model.enable_profiling(true);

    auto scheduler = SchedulerFactory<float>::create_no_op(optimizer.get());

    cout << "Starting CIFAR-100 CNN training..." << endl;
    train_classification_model(model, train_loader, test_loader, std::move(optimizer),
                               std::move(loss_function), std::move(scheduler), train_config);

    cout << "CIFAR-100 CNN Tensor<float> model training completed "
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
