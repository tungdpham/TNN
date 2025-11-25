#include "data_loading/cifar10_data_loader.hpp"
#include "data_loading/tiny_imagenet_data_loader.hpp"
#include "nn/example_models.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "tensor/tensor_ops.hpp"
#include "utils/env.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace tnn;
using namespace std;

constexpr float LR_INITIAL = 0.0001f;

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

    cout << "Using learning rate: " << lr_initial << endl;

    TrainingConfig train_config;
    train_config.load_from_env();
    train_config.print_config();

    CIFAR10DataLoader<float> train_loader, val_loader;

    create_cifar10_dataloader("./data", train_loader, val_loader);

    auto model = SequentialBuilder<float>("ResNet-18-CIFAR10")
                     .input({3, 32, 32})
                     .conv2d(32, 3, 3, 1, 1, 1, 1, true, "conv1")
                     .batchnorm(1e-5f, 0.1f, true, "bn1")
                     .activation("relu", "relu1")
                     //  // Layer 1: 64 channels
                     //  .basic_residual_block(32, 32, 1, "layer1_block1")
                     //  .basic_residual_block(32, 32, 1, "layer1_block2")
                     //  // Layer 2: 128 channels with stride 2
                     //  .basic_residual_block(32, 64, 2, "layer2_block1")
                     //  .basic_residual_block(64, 64, 1, "layer2_block2")
                     //  // Layer 3: 256 channels with stride 2
                     //  .basic_residual_block(64, 128, 2, "layer3_block1")
                     //  .basic_residual_block(128, 128, 1, "layer3_block2")
                     //  // Layer 4: 512 channels with stride 2
                     //  .basic_residual_block(128, 256, 2, "layer4_block1")
                     //  .basic_residual_block(256, 256, 1, "layer4_block2")
                     //  // Global average pooling and classifier
                     .avgpool2d(2, 2, 1, 1, 0, 0, "avgpool")
                     .flatten("flatten")
                     .dense(256, true, "fc")
                     .dense(10, true, "output")
                     .build();

    model.set_seed(1234);
    model.set_device(device_type);
    model.initialize();

    // Use slightly higher epsilon for better numerical stability
    auto optimizer = make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-7f);
    // auto optimizer = make_unique<SGD<float>>(lr_initial, 0.9f);
    model.set_optimizer(std::move(optimizer));

    auto loss_function = LossFactory<float>::create_logsoftmax_crossentropy();

    train_loader.prepare_batches(64);

    Tensor<float> batch_data, batch_labels;

    // test forward
    train_loader.get_next_batch(batch_data, batch_labels);

    cout << "original batch data: " << endl;
    batch_data.print_data();
    cout << endl;

    auto preds = model.forward(batch_data, 0);

    cout << "Predictions data: " << endl;
    preds.print_data();
    cout << endl;

    Tensor<float> grad;
    loss_function->compute_gradient(preds, batch_labels, grad);
    // model.backward(grad);

    // model.update_parameters();

    vector<Tensor<float>> slices;
    split(batch_data, slices, 4);
    vector<Tensor<float>> label_slices;
    split(batch_labels, label_slices, 4);

    cout << "Split predictions into " << slices.size() << " micro-batches:" << endl;

    for (size_t i = 0; i < slices.size(); ++i) {
      cout << "Micro-batch " << i << " data: " << endl;
      slices[i].print_data();
    }

    for (size_t i = 0; i < slices.size(); ++i) {
      auto micro_preds = model.forward(slices[i], i);
      cout << "Micro-batch " << i << " forward output: " << endl;
      micro_preds.print_data();

      Tensor<float> microbatch_grad;
      loss_function->compute_gradient(micro_preds, label_slices[i], microbatch_grad);
      microbatch_grad /= static_cast<float>(slices.size()); // scale gradient
      model.backward(microbatch_grad, i);
    }

    auto layer_ptrs = model.get_layers();
    for (auto &layer : layer_ptrs) {
      cout << "DEBUG: Layer " << layer->name() << " parameters:" << endl;
      auto params = layer->parameters();
      for (auto &param : params) {
        param->head(1000);
      }
      cout << endl;
      auto grads = layer->gradients();
      cout << "DEBUG: Layer " << layer->name() << " gradients:" << endl;
      for (auto &grad : grads) {
        grad->head(1000);
      }
      cout << endl;
    }

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << endl;
    return -1;
  }

  return 0;
}
