#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "data_augmentation/augmentation.hpp"
#include "data_loading/mnist_data_loader.hpp"
#include "nn/example_models.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

constexpr float LR_INITIAL = 0.01f;

int main() {
  cin.tie(nullptr);
  try {
    string device_type_str = Env::get<string>("DEVICE_TYPE", "CPU");

    float lr_initial = Env::get<float>("LR_INITIAL", LR_INITIAL);
    DeviceType device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;

    cout << "Using device type: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;

    TrainingConfig train_config;
    train_config.load_from_env();

    train_config.print_config();

    MNISTDataLoader<float> train_loader, test_loader;

    MNISTDataLoader<float>::create("./data", train_loader, test_loader);

    auto aug_strategy = AugmentationBuilder<float>()
                            .rotation(0.5f, 15.0f)
                            .contrast(0.3f, 0.15f)
                            .gaussian_noise(0.3f, 0.05f)
                            .build();
    train_loader.set_augmentation(std::move(aug_strategy));

    auto model = create_mnist_trainer();

    model.set_device(device_type);
    model.init();

    auto optimizer = OptimizerFactory<float>::create_adam(lr_initial, 0.9f, 0.999f, 1e-8f, 5e-4f);

    auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();

    auto scheduler = SchedulerFactory<float>::create_step_lr(optimizer.get(), 1, 0.95f);

    train_model(model, train_loader, test_loader, std::move(optimizer), std::move(loss_function),
                std::move(scheduler), train_config);
  } catch (const exception &e) {
    cerr << "Error during training: " << e.what() << endl;
    return -1;
  } catch (...) {
    cerr << "Unknown error occurred during training!" << endl;
    return -1;
  }

  return 0;
}
