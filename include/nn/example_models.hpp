/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"

namespace tnn {
Sequential<float> create_mnist_trainer() {
  auto model = SequentialBuilder<float>("mnist_cnn_model")
                   .input({1, 28, 28})
                   .conv2d(8, 5, 5, 1, 1, 0, 0, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(3, 3, 3, 3, 0, 0, "pool1")
                   .conv2d(16, 1, 1, 1, 1, 0, 0, true, "conv2_1x1")
                   .batchnorm(1e-5f, 0.1f, true, "bn2")
                   .activation("relu", "relu2")
                   .conv2d(48, 5, 5, 1, 1, 0, 0, true, "conv3")
                   .batchnorm(1e-5f, 0.1f, true, "bn3")
                   .activation("relu", "relu3")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                   .flatten("flatten")
                   .dense(10, "linear", true, "output")
                   .build();
  return model;
}

Sequential<float> create_cifar10_trainer_v1() {
  auto model = SequentialBuilder<float>("cifar10_cnn_classifier_v1")
                   .input({3, 32, 32})
                   .conv2d(16, 3, 3, 1, 1, 0, 0, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(3, 3, 3, 3, 0, 0, "maxpool1")
                   .conv2d(64, 3, 3, 1, 1, 0, 0, true, "conv2")
                   .batchnorm(1e-5f, 0.1f, true, "bn2")
                   .activation("relu", "relu2")
                   .maxpool2d(4, 4, 4, 4, 0, 0, "maxpool2")
                   .flatten("flatten")
                   .dense(10, "linear", true, "fc1")
                   .build();
  return model;
}

Sequential<float> create_cifar10_trainer_v2() {
  auto model = SequentialBuilder<float>("cifar10_cnn_classifier")
                   .input({3, 32, 32})
                   .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv0")
                   .batchnorm(1e-5f, 0.1f, true, "bn0")
                   .activation("relu", "relu0")
                   .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool0")
                   .conv2d(128, 3, 3, 1, 1, 1, 1, true, "conv2")
                   .batchnorm(1e-5f, 0.1f, true, "bn2")
                   .activation("relu", "relu2")
                   .conv2d(128, 3, 3, 1, 1, 1, 1, true, "conv3")
                   .batchnorm(1e-5f, 0.1f, true, "bn3")
                   .activation("relu", "relu3")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool1")
                   .conv2d(256, 3, 3, 1, 1, 1, 1, true, "conv4")
                   .batchnorm(1e-5f, 0.1f, true, "bn5")
                   .activation("relu", "relu5")
                   .conv2d(256, 3, 3, 1, 1, 1, 1, true, "conv5")
                   .activation("relu", "relu6")
                   .conv2d(256, 3, 3, 1, 1, 1, 1, true, "conv6")
                   .batchnorm(1e-5f, 0.1f, true, "bn6")
                   .activation("relu", "relu6")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool2")
                   .conv2d(512, 3, 3, 1, 1, 1, 1, true, "conv7")
                   .batchnorm(1e-5f, 0.1f, true, "bn8")
                   .activation("relu", "relu7")
                   .conv2d(512, 3, 3, 1, 1, 1, 1, true, "conv8")
                   .batchnorm(1e-5f, 0.1f, true, "bn9")
                   .activation("relu", "relu8")
                   .conv2d(512, 3, 3, 1, 1, 1, 1, true, "conv9")
                   .batchnorm(1e-5f, 0.1f, true, "bn10")
                   .activation("relu", "relu9")
                   .maxpool2d(2, 2, 2, 2, 0, 0, "pool3")
                   .flatten("flatten")
                   .dense(512, "linear", true, "fc0")
                   .activation("relu", "relu10")
                   .dense(10, "linear", true, "fc1")
                   .build();

  return model;
}

Sequential<float> create_resnet18_cifar10() {
  auto model = SequentialBuilder<float>("ResNet-18-CIFAR10")
                   .input({3, 32, 32})
                   .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   // Layer 1: 64 channels
                   .basic_residual_block(64, 64, 1, "layer1_block1")
                   .basic_residual_block(64, 64, 1, "layer1_block2")
                   // Layer 2: 128 channels with stride 2
                   .basic_residual_block(64, 128, 2, "layer2_block1")
                   .basic_residual_block(128, 128, 1, "layer2_block2")
                   // Layer 3: 256 channels with stride 2
                   .basic_residual_block(128, 256, 2, "layer3_block1")
                   .basic_residual_block(256, 256, 1, "layer3_block2")
                   // Layer 4: 512 channels with stride 2
                   .basic_residual_block(256, 512, 2, "layer4_block1")
                   .basic_residual_block(512, 512, 1, "layer4_block2")
                   // Global average pooling and classifier
                   .avgpool2d(4, 4, 1, 1, 0, 0, "avgpool")
                   .flatten("flatten")
                   .dense(10, "linear", true, "fc")
                   .build();
  return model;
}

Sequential<float> create_resnet50_cifar10() {
  auto model = SequentialBuilder<float>("ResNet-50-CIFAR10")
                   .input({3, 32, 32})
                   .conv2d(64, 3, 3, 1, 1, 1, 1, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   // Layer 1: 64 channels, 3 bottleneck blocks
                   .bottleneck_residual_block(64, 64, 256, 1, "layer1_block1")
                   .bottleneck_residual_block(256, 64, 256, 1, "layer1_block2")
                   .bottleneck_residual_block(256, 64, 256, 1, "layer1_block3")
                   // Layer 2: 128 channels, 4 bottleneck blocks with stride 2
                   .bottleneck_residual_block(256, 128, 512, 2, "layer2_block1")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block2")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block3")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block4")
                   // Layer 3: 256 channels, 6 bottleneck blocks with stride 2
                   .bottleneck_residual_block(512, 256, 1024, 2, "layer3_block1")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block2")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block3")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block4")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block5")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block6")
                   // Layer 4: 512 channels, 3 bottleneck blocks with stride 2
                   .bottleneck_residual_block(1024, 512, 2048, 2, "layer4_block1")
                   .bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block2")
                   .bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block3")
                   // Global average pooling and classifier
                   .flatten("flatten")
                   .dense(10, "linear", true, "fc")
                   .build();
  return model;
}

Sequential<float> create_resnet18_imagenet() {
  auto model = SequentialBuilder<float>("ResNet-18-ImageNet")
                   .input({3, 224, 224})
                   .conv2d(64, 7, 7, 2, 2, 3, 3, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(3, 3, 2, 2, 1, 1, "maxpool")
                   // Layer 1: 64 channels
                   .basic_residual_block(64, 64, 1, "layer1_block1")
                   .basic_residual_block(64, 64, 1, "layer1_block2")
                   // Layer 2: 128 channels with stride 2
                   .basic_residual_block(64, 128, 2, "layer2_block1")
                   .basic_residual_block(128, 128, 1, "layer2_block2")
                   // Layer 3: 256 channels with stride 2
                   .basic_residual_block(128, 256, 2, "layer3_block1")
                   .basic_residual_block(256, 256, 1, "layer3_block2")
                   // Layer 4: 512 channels with stride 2
                   .basic_residual_block(256, 512, 2, "layer4_block1")
                   .basic_residual_block(512, 512, 1, "layer4_block2")
                   // Global average pooling and classifier
                   .avgpool2d(7, 7, 1, 1, 0, 0, "avgpool")
                   .flatten("flatten")
                   .dense(1000, "linear", true, "fc")
                   .build();
  return model;
}

Sequential<float> create_resnet50_imagenet() {
  auto model = SequentialBuilder<float>("ResNet-50-ImageNet")
                   .input({3, 224, 224})
                   .conv2d(64, 7, 7, 2, 2, 3, 3, true, "conv1")
                   .batchnorm(1e-5f, 0.1f, true, "bn1")
                   .activation("relu", "relu1")
                   .maxpool2d(3, 3, 2, 2, 1, 1, "maxpool")
                   // Layer 1: 64 channels, 3 bottleneck blocks
                   .bottleneck_residual_block(64, 64, 256, 1, "layer1_block1")
                   .bottleneck_residual_block(256, 64, 256, 1, "layer1_block2")
                   .bottleneck_residual_block(256, 64, 256, 1, "layer1_block3")
                   // Layer 2: 128 channels, 4 bottleneck blocks with stride 2
                   .bottleneck_residual_block(256, 128, 512, 2, "layer2_block1")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block2")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block3")
                   .bottleneck_residual_block(512, 128, 512, 1, "layer2_block4")
                   // Layer 3: 256 channels, 6 bottleneck blocks with stride 2
                   .bottleneck_residual_block(512, 256, 1024, 2, "layer3_block1")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block2")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block3")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block4")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block5")
                   .bottleneck_residual_block(1024, 256, 1024, 1, "layer3_block6")
                   // Layer 4: 512 channels, 3 bottleneck blocks with stride 2
                   .bottleneck_residual_block(1024, 512, 2048, 2, "layer4_block1")
                   .bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block2")
                   .bottleneck_residual_block(2048, 512, 2048, 1, "layer4_block3")
                   // Global average pooling and classifier
                   .avgpool2d(7, 7, 1, 1, 0, 0, "avgpool")
                   .flatten("flatten")
                   .dense(1000, "linear", true, "fc")
                   .build();
  return model;
}

} // namespace tnn