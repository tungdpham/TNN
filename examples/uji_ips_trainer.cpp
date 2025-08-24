#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string_view>
#include <vector>

#include "nn/layers.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include "utils/wifi_data_loader.hpp"

namespace ips_constants {
constexpr float EPSILON = 1e-15f;
constexpr int PROGRESS_PRINT_INTERVAL = 50;
constexpr int LR_DECAY_INTERVAL = 10;
constexpr float LR_DECAY_FACTOR = 0.85f;
constexpr float POSITIONING_ERROR_THRESHOLD = 5.0f;
constexpr size_t MAX_BATCH_SIZE = 32;
constexpr size_t MAX_EPOCHS = 100;
constexpr float learning_rate = 0.01f;
} // namespace ips_constants

class DistanceLoss {
public:
  static float compute_loss(const Tensor<float> &predictions,
                            const Tensor<float> &targets,
                            const WiFiDataLoader &data_loader) {
    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    if (output_size < 2)
      return 0.0f;

    double total_loss = 0.0;

#pragma omp parallel for reduction(+ : total_loss) if (batch_size > 32)
    for (size_t i = 0; i < batch_size; ++i) {

      std::vector<float> pred_coords(output_size), target_coords(output_size);

      for (size_t j = 0; j < output_size; ++j) {
        pred_coords[j] = predictions(i, j, 0, 0);
        target_coords[j] = targets(i, j, 0, 0);
      }

      if (data_loader.is_normalized()) {
        pred_coords = data_loader.denormalize_targets(pred_coords);
        target_coords = data_loader.denormalize_targets(target_coords);
      }

      float distance_sq = 0.0f;
      for (size_t j = 0; j < std::min(size_t(2), output_size); ++j) {
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
    Tensor<float> gradient = predictions;
    gradient.fill(0.0f);

    const size_t batch_size = predictions.shape()[0];
    const size_t output_size = predictions.shape()[1];

    if (output_size < 2)
      return gradient;

    auto target_stds = data_loader.get_target_stds();
    if (target_stds.size() < 2)
      return gradient;

    const float scale = 2.0f / static_cast<float>(batch_size);

#pragma omp parallel for if (batch_size > 32)
    for (size_t i = 0; i < batch_size; ++i) {

      std::vector<float> pred_coords(output_size), target_coords(output_size);

      for (size_t j = 0; j < output_size; ++j) {
        pred_coords[j] = predictions(i, j, 0, 0);
        target_coords[j] = targets(i, j, 0, 0);
      }

      if (data_loader.is_normalized()) {
        pred_coords = data_loader.denormalize_targets(pred_coords);
        target_coords = data_loader.denormalize_targets(target_coords);
      }

      for (size_t j = 0; j < std::min(size_t(2), output_size); ++j) {
        const float real_diff = pred_coords[j] - target_coords[j];

        gradient(i, j, 0, 0) = scale * real_diff * target_stds[j];
      }
    }

    return gradient;
  }
};

void apply_tensor_softmax(Tensor<float> &tensor) {
  const size_t batch_size = tensor.shape()[0];
  const size_t num_classes = tensor.shape()[1];

#pragma omp parallel for if (batch_size > 16)
  for (size_t batch = 0; batch < batch_size; ++batch) {

    float max_val = tensor(batch, 0, 0, 0);
    for (size_t j = 1; j < num_classes; ++j) {
      max_val = std::max(max_val, tensor(batch, j, 0, 0));
    }

    float sum = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      const float exp_val = std::exp(tensor(batch, j, 0, 0) - max_val);
      tensor(batch, j, 0, 0) = exp_val;
      sum += exp_val;
    }

    const float inv_sum = 1.0f / std::max(sum, ips_constants::EPSILON);
    for (size_t j = 0; j < num_classes; ++j) {
      tensor(batch, j, 0, 0) *= inv_sum;
    }
  }
}

float calculate_positioning_accuracy(const Tensor<float> &predictions,
                                     const Tensor<float> &targets,
                                     const WiFiDataLoader &data_loader,
                                     float threshold_meters = 5.0f) {
  const size_t batch_size = predictions.shape()[0];
  const size_t output_size = predictions.shape()[1];

  if (output_size < 2)
    return 0.0f;

  int accurate_predictions = 0;

#pragma omp parallel for reduction(+ : accurate_predictions) if (batch_size >  \
                                                                     16)
  for (size_t i = 0; i < batch_size; ++i) {

    std::vector<float> pred_coords(output_size), target_coords(output_size);

    for (size_t j = 0; j < output_size; ++j) {
      pred_coords[j] = predictions(i, j, 0, 0);
      target_coords[j] = targets(i, j, 0, 0);
    }

    if (data_loader.is_normalized()) {
      pred_coords = data_loader.denormalize_targets(pred_coords);
      target_coords = data_loader.denormalize_targets(target_coords);
    }

    float distance = 0.0f;
    for (size_t j = 0; j < std::min(size_t(2), output_size); ++j) {
      const float diff = pred_coords[j] - target_coords[j];
      distance += diff * diff;
    }
    distance = std::sqrt(distance);

    if (distance <= threshold_meters) {
      accurate_predictions++;
    }
  }

  return static_cast<float>(accurate_predictions) /
         static_cast<float>(batch_size);
}

float calculate_average_positioning_error(const Tensor<float> &predictions,
                                          const Tensor<float> &targets,
                                          const WiFiDataLoader &data_loader,
                                          bool debug = false) {
  const size_t batch_size = predictions.shape()[0];
  const size_t output_size = predictions.shape()[1];

  if (output_size < 2)
    return 0.0f;

  double total_error = 0.0;

  if (debug && batch_size > 0) {
    std::cout << "\nDEBUG: First 3 samples:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), batch_size); ++i) {
      std::vector<float> pred_coords(output_size), target_coords(output_size);

      for (size_t j = 0; j < output_size; ++j) {
        pred_coords[j] = predictions(i, j, 0, 0);
        target_coords[j] = targets(i, j, 0, 0);
      }

      std::cout << "Sample " << i << ":" << std::endl;
      std::cout << "  Raw pred: (" << pred_coords[0] << ", " << pred_coords[1]
                << ")" << std::endl;
      std::cout << "  Raw target: (" << target_coords[0] << ", "
                << target_coords[1] << ")" << std::endl;

      if (data_loader.is_normalized()) {
        auto denorm_pred = data_loader.denormalize_targets(pred_coords);
        auto denorm_target = data_loader.denormalize_targets(target_coords);
        std::cout << "  Denorm pred: (" << denorm_pred[0] << ", "
                  << denorm_pred[1] << ")" << std::endl;
        std::cout << "  Denorm target: (" << denorm_target[0] << ", "
                  << denorm_target[1] << ")" << std::endl;

        float distance = 0.0f;
        for (size_t j = 0; j < std::min(size_t(2), output_size); ++j) {
          const float diff = denorm_pred[j] - denorm_target[j];
          distance += diff * diff;
        }
        distance = std::sqrt(distance);
        std::cout << "  Distance: " << distance << "m" << std::endl;
      }
    }
  }

#pragma omp parallel for reduction(+ : total_error) if (batch_size > 16)
  for (size_t i = 0; i < batch_size; ++i) {

    std::vector<float> pred_coords(output_size), target_coords(output_size);

    for (size_t j = 0; j < output_size; ++j) {
      pred_coords[j] = predictions(i, j, 0, 0);
      target_coords[j] = targets(i, j, 0, 0);
    }

    if (data_loader.is_normalized()) {
      pred_coords = data_loader.denormalize_targets(pred_coords);
      target_coords = data_loader.denormalize_targets(target_coords);
    }

    float distance = 0.0f;
    for (size_t j = 0; j < std::min(size_t(2), output_size); ++j) {
      const float diff = pred_coords[j] - target_coords[j];
      distance += diff * diff;
    }
    distance = std::sqrt(distance);

    total_error += distance;
  }

  return static_cast<float>(total_error / batch_size);
}

float calculate_classification_accuracy(const Tensor<float> &predictions,
                                        const Tensor<float> &targets) {
  const size_t batch_size = predictions.shape()[0];
  const size_t num_classes = predictions.shape()[1];

  int total_correct = 0;

#pragma omp parallel for reduction(+ : total_correct) if (batch_size > 16)
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

void train_ips_model(tnn::Sequential<float> &model,
                     WiFiDataLoader &train_loader, WiFiDataLoader &test_loader,
                     int epochs = 50, int batch_size = 64,
                     float learning_rate = 0.001f) {

  tnn::Adam<float> optimizer(learning_rate, 0.9f, 0.999f, 1e-8f);

  auto classification_loss =
      tnn::LossFactory<float>::create_crossentropy(ips_constants::EPSILON);

  const bool is_regression = train_loader.is_regression();
  const std::string task_type =
      is_regression ? "Coordinate Prediction" : "Classification";

  std::cout << "Starting IPS model training..." << std::endl;
  std::cout << "Task: " << task_type << std::endl;
  std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size
            << ", Learning rate: " << learning_rate << std::endl;
  std::cout << "Features: " << train_loader.num_features()
            << ", Outputs: " << train_loader.num_outputs() << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  std::cout << "\nPreparing training batches..." << std::endl;
  train_loader.prepare_batches(batch_size);

  std::cout << "Preparing validation batches..." << std::endl;
  test_loader.prepare_batches(batch_size);

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  std::cout << "Validation batches: " << test_loader.num_batches() << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  Tensor<float> batch_features, batch_targets, predictions;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    const auto epoch_start = std::chrono::high_resolution_clock::now();

    model.train();
    train_loader.shuffle();
    train_loader.prepare_batches(batch_size);
    train_loader.reset();

    double total_loss = 0.0;
    double total_accuracy = 0.0;
    double total_positioning_error = 0.0;
    int num_batches = 0;

    std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;

    while (train_loader.get_next_batch(batch_features, batch_targets)) {
      ++num_batches;

      predictions = model.forward(batch_features);

      float loss, accuracy, positioning_error = 0.0f;
      Tensor<float> loss_gradient;

      if (is_regression) {

        loss = DistanceLoss::compute_loss(predictions, batch_targets,
                                          train_loader);
        accuracy = calculate_positioning_accuracy(predictions, batch_targets,
                                                  train_loader);
        positioning_error = calculate_average_positioning_error(
            predictions, batch_targets, train_loader);
        loss_gradient = DistanceLoss::compute_gradient(
            predictions, batch_targets, train_loader);
      } else {

        apply_tensor_softmax(predictions);
        loss = classification_loss->compute_loss(predictions, batch_targets);
        accuracy =
            calculate_classification_accuracy(predictions, batch_targets);
        loss_gradient =
            classification_loss->compute_gradient(predictions, batch_targets);
      }

      total_loss += loss;
      total_accuracy += accuracy;
      if (is_regression) {
        total_positioning_error += positioning_error;
      }

      model.backward(loss_gradient);

      auto params = model.parameters();
      auto grads = model.gradients();
      optimizer.update(params, grads);

      if (num_batches % ips_constants::PROGRESS_PRINT_INTERVAL == 0) {
        std::cout << "Batch " << num_batches << " - Loss: " << std::fixed
                  << std::setprecision(4) << loss;
        if (is_regression) {
          std::cout << "m², Accuracy (<5m): " << std::setprecision(4)
                    << accuracy * 100.0f << "%"
                    << ", Avg Error: " << std::setprecision(4)
                    << positioning_error << "m";
        } else {
          std::cout << ", Accuracy: " << std::setprecision(4)
                    << accuracy * 100.0f << "%";
        }
        std::cout << std::endl;
      }
    }

    const float avg_train_loss = static_cast<float>(total_loss / num_batches);
    const float avg_train_accuracy =
        static_cast<float>(total_accuracy / num_batches);
    const float avg_train_positioning_error =
        is_regression
            ? static_cast<float>(total_positioning_error / num_batches)
            : 0.0f;

    model.eval();
    test_loader.reset();

    double val_loss = 0.0;
    double val_accuracy = 0.0;
    double val_positioning_error = 0.0;
    int val_batches = 0;

    while (test_loader.get_next_batch(batch_features, batch_targets)) {
      predictions = model.forward(batch_features);

      if (is_regression) {
        val_loss +=
            DistanceLoss::compute_loss(predictions, batch_targets, test_loader);
        val_accuracy += calculate_positioning_accuracy(
            predictions, batch_targets, test_loader);

        if (val_batches == 0) {
          std::cout << "\nDEBUG: First validation batch analysis:" << std::endl;
          val_positioning_error += calculate_average_positioning_error(
              predictions, batch_targets, test_loader, true);
        } else {
          val_positioning_error += calculate_average_positioning_error(
              predictions, batch_targets, test_loader);
        }
      } else {
        apply_tensor_softmax(predictions);
        val_loss +=
            classification_loss->compute_loss(predictions, batch_targets);
        val_accuracy +=
            calculate_classification_accuracy(predictions, batch_targets);
      }
      ++val_batches;
    }

    const float avg_val_loss = static_cast<float>(val_loss / val_batches);
    const float avg_val_accuracy =
        static_cast<float>(val_accuracy / val_batches);
    const float avg_val_positioning_error =
        is_regression ? static_cast<float>(val_positioning_error / val_batches)
                      : 0.0f;

    const auto epoch_end = std::chrono::high_resolution_clock::now();
    const auto epoch_duration =
        std::chrono::duration_cast<std::chrono::seconds>(epoch_end -
                                                         epoch_start);

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed in "
              << epoch_duration.count() << "s" << std::endl;

    if (is_regression) {
      std::cout << "Training   - Distance Loss: " << std::fixed
                << std::setprecision(2) << avg_train_loss
                << "m², Accuracy (<5m): " << std::setprecision(2)
                << avg_train_accuracy * 100.0f
                << "%, Avg Error: " << std::setprecision(2)
                << avg_train_positioning_error << "m" << std::endl;
      std::cout << "Validation - Distance Loss: " << std::fixed
                << std::setprecision(2) << avg_val_loss
                << "m², Accuracy (<5m): " << std::setprecision(2)
                << avg_val_accuracy * 100.0f
                << "%, Avg Error: " << std::setprecision(2)
                << avg_val_positioning_error << "m" << std::endl;
    } else {
      std::cout << "Training   - CE Loss: " << std::fixed
                << std::setprecision(6) << avg_train_loss
                << ", Accuracy: " << std::setprecision(2)
                << avg_train_accuracy * 100.0f << "%" << std::endl;
      std::cout << "Validation - CE Loss: " << std::fixed
                << std::setprecision(6) << avg_val_loss
                << ", Accuracy: " << std::setprecision(2)
                << avg_val_accuracy * 100.0f << "%" << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;

    if ((epoch + 1) % ips_constants::LR_DECAY_INTERVAL == 0) {
      const float current_lr = optimizer.get_learning_rate();
      const float new_lr = current_lr * ips_constants::LR_DECAY_FACTOR;
      optimizer.set_learning_rate(new_lr);
      std::cout << "Learning rate decayed: " << std::fixed
                << std::setprecision(8) << current_lr << " → " << new_lr
                << std::endl;
    }
  }
}

int main() {
  try {
    std::cout << "Indoor Positioning System (IPS) Neural Network Training"
              << std::endl;
    std::cout << "Supports UTS, UJI and other WiFi fingerprinting datasets"
              << std::endl;
    std::cout << std::string(70, '=') << std::endl;
#ifdef _OPENMP

    const int num_threads = omp_get_max_threads();
    omp_set_num_threads(std::min(num_threads, 8));
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads"
              << std::endl;
#endif

    bool is_regression = true;
    WiFiDataLoader train_loader(is_regression), test_loader(is_regression);

    std::string train_file = "./data/uji/train.csv";
    std::string test_file = "./data/uji/validation.csv";

    std::cout << "\nLoading training data from: " << train_file << std::endl;

    if (!train_loader.load_data(train_file, 0, 520, 520, 522, true)) {
      std::cerr << "Failed to load training data!" << std::endl;
      std::cerr << "Please ensure the data file exists and adjust column "
                   "indices if needed."
                << std::endl;
      return -1;
    }

    std::cout << "Loading test data from: " << test_file << std::endl;

    if (!test_loader.load_data(test_file, 0, 520, 520, 522, true)) {
      std::cerr << "Failed to load test data!" << std::endl;
      std::cerr << "Please ensure the data file exists and adjust column "
                   "indices if needed."
                << std::endl;
      return -1;
    }

    train_loader.print_statistics();
    test_loader.print_statistics();

    std::cout << "\nNormalizing training data..." << std::endl;
    train_loader.normalize_data();

    auto feature_means = train_loader.get_feature_means();
    auto feature_stds = train_loader.get_feature_stds();
    auto target_means = train_loader.get_target_means();
    auto target_stds = train_loader.get_target_stds();

    std::cout << "Normalizing test data using training statistics..."
              << std::endl;
    test_loader.apply_normalization(feature_means, feature_stds, target_means,
                                    target_stds);

    std::cout << "\nNormalization Statistics:" << std::endl;
    std::cout << "Target means: ";
    for (size_t i = 0; i < std::min(target_means.size(), size_t(2)); ++i) {
      std::cout << target_means[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Target stds: ";
    for (size_t i = 0; i < std::min(target_stds.size(), size_t(2)); ++i) {
      std::cout << target_stds[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\nBuilding IPS model architecture..." << std::endl;

    const size_t input_features = train_loader.num_features();
    const size_t output_size = train_loader.num_outputs();
    const std::string output_activation = is_regression ? "linear" : "linear";

    auto model = tnn::SequentialBuilder<float>("ips_classifier")

                     .dense(192, "linear", true, "hidden1")
                     .batchnorm(1e-5, 0.1, true, "batchnorm1")
                     .activation("relu", "hidden1_relu")
                     .dropout(0.25f, "dropout1")

                     .dense(64, "linear", true, "hidden2")
                     .batchnorm(1e-5, 0.1, true, "batchnorm2")
                     .activation("relu", "hidden2_relu")

                     .dense(32, "linear", true, "hidden3")
                     .batchnorm(1e-5, 0.1, true, "batchnorm3")
                     .activation("relu", "hidden3_relu")

                     .dropout(0.25f, "dropout3")

                     .dense(16, "linear", true, "hidden4")
                     .batchnorm(1e-5, 0.1, true, "batchnorm4")
                     .activation("relu", "hidden4_relu")

                     .dense(output_size, output_activation, true, "output")
                     .build();

    std::cout << "\nModel Architecture Summary:" << std::endl;
    model.print_summary(std::vector<size_t>{ips_constants::MAX_BATCH_SIZE,
                                            input_features, 1, 1});

    constexpr int epochs = ips_constants::MAX_EPOCHS;
    constexpr int batch_size = ips_constants::MAX_BATCH_SIZE;
    constexpr float learning_rate = ips_constants::learning_rate;

    std::cout << "\nStarting IPS model training..." << std::endl;
    train_ips_model(model, train_loader, test_loader, epochs, batch_size,
                    learning_rate);

    std::cout << "\nIPS model training completed successfully!" << std::endl;

    try {
      const std::string model_name =
          is_regression ? "ips_regression_model" : "ips_classification_model";
      model.save_to_file("model_snapshots/" + model_name);
      std::cout << "Model saved to: model_snapshots/" << model_name
                << std::endl;
    } catch (const std::exception &save_error) {
      std::cerr << "Warning: Failed to save model: " << save_error.what()
                << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error during training: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "Unknown error occurred during training!" << std::endl;
    return -1;
  }

  return 0;
}
