/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "data_loading/tiny_imagenet_data_loader.hpp"
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

constexpr float LR_INITIAL = 0.0001f;

int main() {
  try {
    // Load environment variables from .env file
    cout << "Loading environment variables..." << endl;

    string device_type_str = Env::get<string>("DEVICE_TYPE", "CPU");

    float lr_initial = Env::get<float>("LR_INITIAL", LR_INITIAL);
    DeviceType device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;

    cout << "Using learning rate: " << lr_initial << endl;

    TrainingConfig train_config;
    train_config.load_from_env();
    train_config.print_config();

    TinyImageNetDataLoader<float> train_loader, val_loader;
    TinyImageNetDataLoader<float>::create("data/tiny-imagenet-200", train_loader, val_loader);

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

    auto model = create_resnet34_tiny_imagenet();

    model.set_device(device_type);
    model.init();

    auto optimizer = OptimizerFactory<float>::create_adam(lr_initial, 0.9f, 0.999f, 1e-8f);

    auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();

    model.enable_profiling(true);

    auto scheduler = SchedulerFactory<float>::create_no_op(optimizer.get());

    train_model(model, train_loader, val_loader, std::move(optimizer), std::move(loss_function),
                std::move(scheduler), train_config);

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return -1;
  }

  return 0;
}
