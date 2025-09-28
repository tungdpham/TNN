#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "data_loading/mnist_data_loader.hpp"

void run_test() {

  tnn::Sequential<float> model;
  try {
    model = tnn::Sequential<float>::from_file("model_snapshots/mnist_cnn_model");
    std::cout << "Model loaded successfully from model_snapshots/mnist_cnn_model\n";
  } catch (const std::exception &e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
    return;
  }

  model.print_config();

  model.set_training(false);

  data_loading::MNISTDataLoader<float> loader;
  if (!loader.load_data("data/mnist/test.csv")) {
    return;
  }

  size_t batch_size = 100;
  size_t correct_predictions = 0;
  size_t total_samples = 0;

  Tensor<float> batch_data, batch_labels;
  while (loader.get_batch(batch_size, batch_data, batch_labels)) {
    Tensor<float> predictions = model.forward(batch_data);

    for (size_t i = 0; i < predictions.batch_size(); ++i) {
      size_t predicted_label = std::distance(
          predictions.data() + i * 10,
          std::max_element(predictions.data() + i * 10, predictions.data() + (i + 1) * 10));
      size_t true_label = static_cast<size_t>(batch_labels.data()[i * 10]);

      if (predicted_label == true_label) {
        correct_predictions++;
      }
    }
    total_samples += predictions.batch_size();
  }

  double accuracy = (double)correct_predictions / total_samples;
  std::cout << "Test Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100 << "%"
            << std::endl;
}

int main() {
  try {
    run_test();
  } catch (const std::exception &e) {
    std::cerr << "An error occurred during testing: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
