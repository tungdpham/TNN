#pragma once

#include "data_loader.hpp"
#include "nn/sequential.hpp"

template <typename T = float>
void train_classification(
  tnn::Sequential<T>& model,
  data_loading::BaseDataLoader<T>& train_loader,
  data_loading::BaseDataLoader<T>& test_loader,
  int epochs = 10,
  int batch_size = 32,
  float learning_rate = 0.001f
) {
  Tensor<T> batch_data, batch_labels, predictions;

  // Prepare batches for training and validation data
  train_loader.prepare_batches(batch_size);
  test_loader.prepare_batches(batch_size);

}