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
    string device_type_str = Env::get<string>("DEVICE_TYPE", "CPU");

    float lr_initial = Env::get<float>("LR_INITIAL", LR_INITIAL);
    DeviceType device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;

    TrainingConfig train_config;
    train_config.load_from_env();
    train_config.print_config();

    CIFAR10DataLoader<float> train_loader, test_loader;

    CIFAR10DataLoader<float>::create("./data", train_loader, test_loader);

    auto train_transform = AugmentationBuilder<float>()
                               .random_crop(0.5f, 4)
                               .horizontal_flip(0.5f)
                               .cutout(0.5f, 8)
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

    cout << "Building CNN model architecture for CIFAR-10..." << endl;

    auto model = create_resnet9_cifar10();

    model.set_device(device_type);
    model.initialize();

    auto optimizer = OptimizerFactory<float>::create_adam(lr_initial, 0.9f, 0.999f, 1e-8f);

    auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();

    model.enable_profiling(true);

    auto scheduler = SchedulerFactory<float>::create_no_op(optimizer.get());

    cout << "Starting CIFAR-10 CNN training..." << endl;
    train_classification_model(model, train_loader, test_loader, std::move(optimizer),
                               std::move(loss_function), std::move(scheduler), train_config);

    cout << "CIFAR-10 CNN Tensor<float> model training completed successfully!" << endl;
  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return -1;
  }

  return 0;
}
