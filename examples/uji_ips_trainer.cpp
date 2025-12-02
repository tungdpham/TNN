#include "data_loading/wifi_data_loader.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include "utils/env.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

#include <vector>

using namespace tnn;
using namespace std;

namespace ips_constants {
constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 200;
constexpr int LR_DECAY_INTERVAL = 10;
constexpr float LR_DECAY_FACTOR = 0.85f;
constexpr size_t MAX_BATCH_SIZE = 32;
constexpr size_t MAX_EPOCHS = 100;
constexpr float learning_rate = 0.01f;
} // namespace ips_constants

class DistanceLoss {
public:
  static float compute_loss(const Tensor<float> &predictions, const Tensor<float> &targets,
                            const WiFiDataLoader &data_loader) {
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    if (output_size < 2)
      return 0.0f;

    double total_loss = 0.0;

    for (size_t i = 0; i < batch_size; ++i) {

      vector<float> pred_coords(output_size), target_coords(output_size);

      for (size_t j = 0; j < output_size; ++j) {
        pred_coords[j] = predictions(i, j, 0, 0);
        target_coords[j] = targets(i, j, 0, 0);
      }

      if (data_loader.is_normalized()) {
        pred_coords = data_loader.denormalize_targets(pred_coords);
        target_coords = data_loader.denormalize_targets(target_coords);
      }

      float distance_sq = 0.0f;
      for (size_t j = 0; j < min(size_t(2), output_size); ++j) {
        const float diff = pred_coords[j] - target_coords[j];
        distance_sq += diff * diff;
      }

      total_loss += distance_sq;
    }

    return static_cast<float>(total_loss / batch_size);
  }

  static Tensor<float> compute_gradient(const Tensor<float> &predictions,
                                        const Tensor<float> &targets,
                                        const WiFiDataLoader &data_loader) {
    Tensor<float> gradient = predictions.clone();
    gradient.fill(0.0f);

    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    if (output_size < 2)
      return gradient;

    auto target_stds = data_loader.get_target_stds();
    if (target_stds.size() < 2)
      return gradient;

    const float scale = 2.0f / static_cast<float>(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {

      vector<float> pred_coords(output_size), target_coords(output_size);

      for (size_t j = 0; j < output_size; ++j) {
        pred_coords[j] = predictions(i, j, 0, 0);
        target_coords[j] = targets(i, j, 0, 0);
      }

      if (data_loader.is_normalized()) {
        pred_coords = data_loader.denormalize_targets(pred_coords);
        target_coords = data_loader.denormalize_targets(target_coords);
      }

      for (size_t j = 0; j < min(size_t(2), output_size); ++j) {
        const float real_diff = pred_coords[j] - target_coords[j];

        gradient(i, j, 0, 0) = scale * real_diff * target_stds[j];
      }
    }

    return gradient;
  }
};

float calculate_positioning_accuracy(const Tensor<float> &predictions, const Tensor<float> &targets,
                                     const WiFiDataLoader &data_loader,
                                     float threshold_meters = 5.0f) {
  const size_t batch_size = predictions.shape()[0];
  const size_t output_size = predictions.shape()[1];

  if (output_size < 2)
    return 0.0f;

  int accurate_predictions = 0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : accurate_predictions) if (batch_size > 16)
#endif
  for (size_t i = 0; i < batch_size; ++i) {

    vector<float> pred_coords(output_size), target_coords(output_size);

    for (size_t j = 0; j < output_size; ++j) {
      pred_coords[j] = predictions(i, j, 0, 0);
      target_coords[j] = targets(i, j, 0, 0);
    }

    if (data_loader.is_normalized()) {
      pred_coords = data_loader.denormalize_targets(pred_coords);
      target_coords = data_loader.denormalize_targets(target_coords);
    }

    float distance = 0.0f;
    for (size_t j = 0; j < min(size_t(2), output_size); ++j) {
      const float diff = pred_coords[j] - target_coords[j];
      distance += diff * diff;
    }
    distance = sqrt(distance);

    if (distance <= threshold_meters) {
      accurate_predictions++;
    }
  }

  return static_cast<float>(accurate_predictions) / static_cast<float>(batch_size);
}

float calculate_average_positioning_error(const Tensor<float> &predictions,
                                          const Tensor<float> &targets,
                                          const WiFiDataLoader &data_loader, bool debug = false) {
  const size_t batch_size = predictions.shape()[0];
  const size_t output_size = predictions.shape()[1];

  if (output_size < 2)
    return 0.0f;

  double total_error = 0.0;

  if (debug && batch_size > 0) {
    cout << "\nDEBUG: First 3 samples:" << endl;
    for (size_t i = 0; i < min(size_t(3), batch_size); ++i) {
      vector<float> pred_coords(output_size), target_coords(output_size);

      for (size_t j = 0; j < output_size; ++j) {
        pred_coords[j] = predictions(i, j, 0, 0);
        target_coords[j] = targets(i, j, 0, 0);
      }

      cout << "Sample " << i << ":" << endl;
      cout << "  Raw pred: (" << pred_coords[0] << ", " << pred_coords[1] << ")" << endl;
      cout << "  Raw target: (" << target_coords[0] << ", " << target_coords[1] << ")" << endl;

      if (data_loader.is_normalized()) {
        auto denorm_pred = data_loader.denormalize_targets(pred_coords);
        auto denorm_target = data_loader.denormalize_targets(target_coords);
        cout << "  Denorm pred: (" << denorm_pred[0] << ", " << denorm_pred[1] << ")" << endl;
        cout << "  Denorm target: (" << denorm_target[0] << ", " << denorm_target[1] << ")" << endl;

        float distance = 0.0f;
        for (size_t j = 0; j < min(size_t(2), output_size); ++j) {
          const float diff = denorm_pred[j] - denorm_target[j];
          distance += diff * diff;
        }
        distance = sqrt(distance);
        cout << "  Distance: " << distance << "m" << endl;
      }
    }
  }
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total_error) if (batch_size > 16)
#endif
  for (size_t i = 0; i < batch_size; ++i) {

    vector<float> pred_coords(output_size), target_coords(output_size);

    for (size_t j = 0; j < output_size; ++j) {
      pred_coords[j] = predictions(i, j, 0, 0);
      target_coords[j] = targets(i, j, 0, 0);
    }

    if (data_loader.is_normalized()) {
      pred_coords = data_loader.denormalize_targets(pred_coords);
      target_coords = data_loader.denormalize_targets(target_coords);
    }

    float distance = 0.0f;
    for (size_t j = 0; j < min(size_t(2), output_size); ++j) {
      const float diff = pred_coords[j] - target_coords[j];
      distance += diff * diff;
    }
    distance = sqrt(distance);

    total_error += distance;
  }

  return static_cast<float>(total_error / batch_size);
}

float calculate_classification_accuracy(const Tensor<float> &predictions,
                                        const Tensor<float> &targets) {
  const size_t batch_size = predictions.shape()[0];
  const size_t num_classes = predictions.shape()[1];

  int total_correct = 0;

  for (size_t i = 0; i < batch_size; ++i) {

    int pred_class = 0;
    float max_pred = predictions(i, 0, 0, 0);
    for (size_t j = 1; j < num_classes; ++j) {
      const float pred_val = predictions(i, j, 0, 0);
      if (pred_val > max_pred) {
        max_pred = pred_val;
        pred_class = static_cast<int>(j);
      }
    }

    int true_class = -1;
    for (size_t j = 0; j < num_classes; ++j) {
      if (targets(i, j, 0, 0) > 0.5f) {
        true_class = static_cast<int>(j);
        break;
      }
    }

    if (pred_class == true_class && true_class != -1) {
      total_correct++;
    }
  }

  return static_cast<float>(total_correct) / static_cast<float>(batch_size);
}

void train_ips_model(Sequential<float> &model, WiFiDataLoader &train_loader,
                     WiFiDataLoader &test_loader, Optimizer<float> &optimizer,
                     Loss<float> &classification_loss, int epochs = 50, int batch_size = 64,
                     float learning_rate = 0.001f) {

  const bool is_regression = train_loader.is_regression();
  const string job_type = is_regression ? "Coordinate Prediction" : "Classification";

  cout << "Starting IPS model training..." << endl;
  cout << "Job: " << job_type << endl;
  cout << "Epochs: " << epochs << ", Batch size: " << batch_size
       << ", Learning rate: " << learning_rate << endl;
  cout << "Features: " << train_loader.num_features() << ", Outputs: " << train_loader.num_outputs()
       << endl;
  cout << string(80, '=') << endl;

  cout << "\nPreparing training batches..." << endl;
  train_loader.prepare_batches(batch_size);

  cout << "Preparing validation batches..." << endl;
  test_loader.prepare_batches(batch_size);

  cout << "Training batches: " << train_loader.num_batches() << endl;
  cout << "Validation batches: " << test_loader.num_batches() << endl;
  cout << string(80, '=') << endl;

  Tensor<float> batch_features, batch_targets, predictions;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    const auto epoch_start = chrono::high_resolution_clock::now();

    model.set_training(true);
    train_loader.shuffle();
    train_loader.prepare_batches(batch_size);
    train_loader.reset();

    double total_loss = 0.0;
    double total_accuracy = 0.0;
    double total_positioning_error = 0.0;
    int num_batches = 0;

    cout << "Epoch " << epoch + 1 << "/" << epochs << endl;

    while (train_loader.get_next_batch(batch_features, batch_targets)) {
      ++num_batches;

      model.forward(batch_features);

      float loss, accuracy, positioning_error = 0.0f;
      Tensor<float> loss_gradient;

      if (is_regression) {

        loss = DistanceLoss::compute_loss(predictions, batch_targets, train_loader);
        accuracy = calculate_positioning_accuracy(predictions, batch_targets, train_loader);
        positioning_error =
            calculate_average_positioning_error(predictions, batch_targets, train_loader);
        loss_gradient = DistanceLoss::compute_gradient(predictions, batch_targets, train_loader);
      } else {
        apply_softmax(predictions);
        classification_loss.compute_loss(predictions, batch_targets, loss);
        accuracy = calculate_classification_accuracy(predictions, batch_targets);
        classification_loss.compute_gradient(predictions, batch_targets, loss_gradient);
      }

      total_loss += loss;
      total_accuracy += accuracy;
      if (is_regression) {
        total_positioning_error += positioning_error;
      }

      model.backward(loss_gradient);

      optimizer.update();

      if (num_batches % ips_constants::PROGRESS_PRINT_INTERVAL == 0) {
        cout << "Batch " << num_batches << " - Loss: " << fixed << setprecision(4) << loss;
        if (is_regression) {
          cout << "m², Accuracy (<5m): " << setprecision(4) << accuracy * 100.0f << "%"
               << ", Avg Error: " << setprecision(4) << positioning_error << "m";
        } else {
          cout << ", Accuracy: " << setprecision(4) << accuracy * 100.0f << "%";
        }
        cout << endl;
      }
    }

    const float avg_train_loss = static_cast<float>(total_loss / num_batches);
    const float avg_train_accuracy = static_cast<float>(total_accuracy / num_batches);
    const float avg_train_positioning_error =
        is_regression ? static_cast<float>(total_positioning_error / num_batches) : 0.0f;

    model.set_training(false);
    test_loader.reset();

    double val_loss = 0.0;
    double val_accuracy = 0.0;
    double val_positioning_error = 0.0;
    int val_batches = 0;

    while (test_loader.get_next_batch(batch_features, batch_targets)) {
      model.forward(batch_features);

      if (is_regression) {
        val_loss += DistanceLoss::compute_loss(predictions, batch_targets, test_loader);
        val_accuracy += calculate_positioning_accuracy(predictions, batch_targets, test_loader);

        if (val_batches == 0) {
          cout << "\nDEBUG: First validation batch analysis:" << endl;
          val_positioning_error +=
              calculate_average_positioning_error(predictions, batch_targets, test_loader, true);
        } else {
          val_positioning_error +=
              calculate_average_positioning_error(predictions, batch_targets, test_loader);
        }
      } else {
        apply_softmax(predictions);
        float loss;
        classification_loss.compute_loss(predictions, batch_targets, loss);
        val_loss += loss;
        val_accuracy += calculate_classification_accuracy(predictions, batch_targets);
      }
      ++val_batches;
    }

    const float avg_val_loss = static_cast<float>(val_loss / val_batches);
    const float avg_val_accuracy = static_cast<float>(val_accuracy / val_batches);
    const float avg_val_positioning_error =
        is_regression ? static_cast<float>(val_positioning_error / val_batches) : 0.0f;

    const auto epoch_end = chrono::high_resolution_clock::now();
    const auto epoch_duration = chrono::duration_cast<chrono::seconds>(epoch_end - epoch_start);

    cout << string(80, '-') << endl;
    cout << "Epoch " << epoch + 1 << "/" << epochs << " completed in " << epoch_duration.count()
         << "s" << endl;

    if (is_regression) {
      cout << "Training   - Distance Loss: " << fixed << setprecision(2) << avg_train_loss
           << "m², Accuracy (<5m): " << setprecision(2) << avg_train_accuracy * 100.0f
           << "%, Avg Error: " << setprecision(2) << avg_train_positioning_error << "m" << endl;
      cout << "Validation - Distance Loss: " << fixed << setprecision(2) << avg_val_loss
           << "m², Accuracy (<5m): " << setprecision(2) << avg_val_accuracy * 100.0f
           << "%, Avg Error: " << setprecision(2) << avg_val_positioning_error << "m" << endl;
    } else {
      cout << "Training   - CE Loss: " << fixed << setprecision(6) << avg_train_loss
           << ", Accuracy: " << setprecision(2) << avg_train_accuracy * 100.0f << "%" << endl;
      cout << "Validation - CE Loss: " << fixed << setprecision(6) << avg_val_loss
           << ", Accuracy: " << setprecision(2) << avg_val_accuracy * 100.0f << "%" << endl;
    }
    cout << string(80, '=') << endl;

    if ((epoch + 1) % ips_constants::LR_DECAY_INTERVAL == 0) {
      const float current_lr = optimizer.get_learning_rate();
      const float new_lr = current_lr * ips_constants::LR_DECAY_FACTOR;
      optimizer.set_learning_rate(new_lr);
      cout << "Learning rate decayed: " << fixed << setprecision(8) << current_lr << " → " << new_lr
           << endl;
    }
  }
}

int main() {
  try {
    // Load environment variables from .env file
    cout << "Loading environment variables..." << endl;
    if (!load_env_file("./.env")) {
      cout << "No .env file found, using default training parameters." << endl;
    }

    // Get training parameters from environment or use defaults
    const size_t max_epochs = get_env<size_t>("EPOCHS", ips_constants::MAX_EPOCHS);
    const size_t batch_size = get_env<size_t>("BATCH_SIZE", ips_constants::MAX_BATCH_SIZE);
    const float lr_initial = get_env<float>("LR_INITIAL", ips_constants::learning_rate);
    const float lr_decay_factor = get_env<float>("LR_DECAY_FACTOR", ips_constants::LR_DECAY_FACTOR);
    const int lr_decay_interval =
        get_env<int>("LR_DECAY_INTERVAL", ips_constants::LR_DECAY_INTERVAL);

    cout << "Indoor Positioning System (IPS) Neural Network Training" << endl;
    cout << "Supports UTS, UJI and other WiFi fingerprinting datasets" << endl;
    cout << string(70, '=') << endl;
    cout << "Training Parameters:" << endl;
    cout << "  Max Epochs: " << max_epochs << endl;
    cout << "  Batch Size: " << batch_size << endl;
    cout << "  Initial Learning Rate: " << lr_initial << endl;
    cout << "  LR Decay Factor: " << lr_decay_factor << endl;
    cout << "  LR Decay Interval: " << lr_decay_interval << endl;
    cout << string(70, '=') << endl;
    bool is_regression = true;
    WiFiDataLoader train_loader(is_regression), test_loader(is_regression);

    string train_file = "./data/uji/TrainingData.csv";
    string test_file = "./data/uji/ValidationData.csv";

    cout << "\nLoading training data from: " << train_file << endl;

    if (!train_loader.load_data(train_file, 0, 520, 520, 522, true)) {
      cerr << "Failed to load training data!" << endl;
      cerr << "Please ensure the data file exists and adjust column "
              "indices if needed."
           << endl;
      return -1;
    }

    cout << "Loading test data from: " << test_file << endl;

    if (!test_loader.load_data(test_file, 0, 520, 520, 522, true)) {
      cerr << "Failed to load test data!" << endl;
      cerr << "Please ensure the data file exists and adjust column "
              "indices if needed."
           << endl;
      return -1;
    }

    train_loader.print_statistics();
    test_loader.print_statistics();

    cout << "\nNormalizing training data..." << endl;
    train_loader.normalize_data();

    auto feature_means = train_loader.get_feature_means();
    auto feature_stds = train_loader.get_feature_stds();
    auto target_means = train_loader.get_target_means();
    auto target_stds = train_loader.get_target_stds();

    cout << "Normalizing test data using training statistics..." << endl;
    test_loader.apply_normalization(feature_means, feature_stds, target_means, target_stds);

    cout << "\nNormalization Statistics:" << endl;
    cout << "Target means: ";
    for (size_t i = 0; i < min(target_means.size(), size_t(2)); ++i) {
      cout << target_means[i] << " ";
    }
    cout << endl;
    cout << "Target stds: ";
    for (size_t i = 0; i < min(target_stds.size(), size_t(2)); ++i) {
      cout << target_stds[i] << " ";
    }
    cout << endl;

    cout << "\nBuilding IPS model architecture..." << endl;

    const size_t input_features = train_loader.num_features();
    const size_t output_size = train_loader.num_outputs();

    auto model = SequentialBuilder<float>("ips_classifier")
                     .input({input_features, 1, 1})
                     .dense(192, true, "hidden1")
                     .batchnorm(1e-5f, 0.1f, true, "batchnorm1")
                     .activation("relu", "hidden1_relu")
                     .dropout(0.25f, "dropout1")

                     .dense(64, true, "hidden2")
                     .batchnorm(1e-5f, 0.1f, true, "batchnorm2")
                     .activation("relu", "hidden2_relu")

                     .dense(32, true, "hidden3")
                     .batchnorm(1e-5f, 0.1f, true, "batchnorm3")
                     .activation("relu", "hidden3_relu")

                     .dropout(0.25f, "dropout3")

                     .dense(16, true, "hidden4")
                     .batchnorm(1e-5f, 0.1f, true, "batchnorm4")
                     .activation("relu", "hidden4_relu")

                     .dense(output_size, true, "output")
                     .build();

    auto optimizer = make_unique<Adam<float>>(lr_initial, 0.9f, 0.999f, 1e-8f);
    auto loss_function = LossFactory<float>::create_crossentropy(ips_constants::EPSILON);

    // Attach optimizer to model parameters
    auto params = model.parameters();
    auto grads = model.gradients();
    optimizer->attach(params, grads);

    cout << "\nModel Architecture Summary:" << endl;

    cout << "\nStarting IPS model training..." << endl;
    train_ips_model(model, train_loader, test_loader, *optimizer, *loss_function, max_epochs,
                    batch_size, lr_initial);

    cout << "\nIPS model training completed successfully!" << endl;

    try {
      const string model_name = is_regression ? "ips_regression_model" : "ips_classification_model";
      model.save_to_file("model_snapshots/" + model_name);
      cout << "Model saved to: model_snapshots/" << model_name << endl;
    } catch (const exception &save_error) {
      cerr << "Warning: Failed to save model: " << save_error.what() << endl;
    }

  } catch (const exception &e) {
    cerr << "Error during training: " << e.what() << endl;
    return -1;
  } catch (...) {
    cerr << "Unknown error occurred during training!" << endl;
    return -1;
  }

  return 0;
}
