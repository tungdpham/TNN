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

} // namespace tnn