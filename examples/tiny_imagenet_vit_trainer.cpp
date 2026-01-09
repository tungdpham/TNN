#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "data_loading/tiny_imagenet_data_loader.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "nn/sequential.hpp"
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

    size_t patch_size = 4;
    size_t embed_dim = 256;
    size_t num_heads = 4;
    size_t mlp_ratio = 4;
    size_t depth = 1;
    size_t num_classes = 200;
    size_t num_patches = (64 / patch_size) * (64 / patch_size);
    size_t seq_len = num_patches + 1;

    SequentialBuilder<float> builder("ViT_TinyImageNet");
    builder.input({3, 64, 64})
        .conv2d(embed_dim, patch_size, patch_size, patch_size, patch_size, 0, 0, true,
                "patch_embed")
        .flatten(2, "flatten_patches")
        .class_token(embed_dim)
        .positional_embedding(embed_dim, seq_len)
        .dropout(0.1f);

    for (size_t i = 0; i < depth; ++i) {
      builder.residual(LayerBuilder<float>()
                           .input({embed_dim, seq_len, 1})
                           .layernorm(1e-5f, true, "ln_attn")
                           .full_attention(embed_dim, num_heads)
                           .dropout(0.1f)
                           .build(),
                       {}, "linear", "encoder_" + to_string(i) + "_attn");

      builder.residual(LayerBuilder<float>()
                           .input({embed_dim, seq_len, 1})
                           .layernorm(1e-5f, true, "ln_mlp")
                           .conv2d(embed_dim * mlp_ratio, 1, 1)
                           .activation("gelu")
                           .dropout(0.1f)
                           .conv2d(embed_dim, 1, 1)
                           .dropout(0.1f)
                           .build(),
                       {}, "linear", "encoder_" + to_string(i) + "_mlp");
    }

    builder.layernorm(1e-5f, true, "ln_final");
    builder.slice(2, 0, 1, "extract_cls_token");
    builder.dense(num_classes, true, "head");

    auto model = builder.build();

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
