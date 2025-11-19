#include "data_loading/mnist_data_loader.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include "utils/utils_extended.hpp"
#include <iomanip>
#include <iostream>
#include <string>

using namespace tnn;
using namespace std;

void run_test() {

  Sequential<float> model;
  try {
    model = Sequential<float>::from_file("model_snapshots/mnist_cnn_model");
    cout << "Model loaded successfully from model_snapshots/mnist_cnn_model\n";
  } catch (const exception &e) {
    cerr << "Error loading model: " << e.what() << endl;
    return;
  }

  model.print_config();

  model.set_training(false);

  MNISTDataLoader<float> loader;

  if (!loader.load_data("data/mnist/test.csv")) {
    return;
  }

  size_t batch_size = 100;
  size_t correct_predictions = 0;

  Tensor<float> batch_data, batch_labels;
  while (loader.get_batch(batch_size, batch_data, batch_labels)) {
    Tensor<float> predictions = model.forward(batch_data);

    correct_predictions += compute_class_corrects(predictions, batch_labels);
  }

  double accuracy = (double)correct_predictions / loader.size();
  cout << "Test Accuracy: " << fixed << setprecision(4) << accuracy * 100 << "%" << endl;
}

int main() {
  try {
    run_test();
  } catch (const exception &e) {
    cerr << "An error occurred during testing: " << e.what() << endl;
    return 1;
  }
  return 0;
}
