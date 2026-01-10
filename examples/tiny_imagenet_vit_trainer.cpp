#include <cmath>
#include <iostream>
#include <memory>

#include "data_loading/tiny_imagenet_data_loader.hpp"
#include "nn/example_models.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

constexpr float LR_INITIAL = 0.0005f;

int main() {
  cin.tie(nullptr);
  try {
    cout << "Loading environment variables..." << endl;
    string device_type_str = Env::get<string>("DEVICE_TYPE", "CPU");
    float lr_initial = Env::get<float>("LR_INITIAL", LR_INITIAL);
    DeviceType device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;
    cout << "Using device type: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;

    TrainingConfig train_config;
    train_config.load_from_env();
    train_config.print_config();

    TinyImageNetDataLoader<float> train_loader, val_loader;
    TinyImageNetDataLoader<float>::create("./data/tiny-imagenet-200", train_loader, val_loader);

    auto train_aug = AugmentationBuilder<float>()
                         .random_crop(1.0f, 4)
                         .rotation(0.25f, 5.0f)
                         .horizontal_flip(0.5)
                         .brightness(0.2f)
                         .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                         .build();
    train_loader.set_augmentation(std::move(train_aug));

    auto val_aug = AugmentationBuilder<float>()
                       .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
                       .build();
    val_loader.set_augmentation(std::move(val_aug));

    auto model = create_vit_tiny_imagenet();

    model.set_device(device_type);
    model.initialize();

    auto optimizer = OptimizerFactory<float>::create_adam(lr_initial, 0.9f, 0.999f, 1e-8f, 1e-4f);
    auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();
    auto scheduler = SchedulerFactory<float>::create_step_lr(optimizer.get(), 1, 0.9f);

    train_model(model, train_loader, val_loader, std::move(optimizer), std::move(loss_function),
                std::move(scheduler), train_config);
  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }
  return 0;
}
