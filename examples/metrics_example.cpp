/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include <iostream>

#include "nn/metrics.hpp"
#include "tensor/tensor.hpp"

using namespace tnn;

int main() {
  // Example: Classification metrics
  std::cout << "=== Classification Metrics Example ===" << std::endl;

  // Create dummy predictions (probabilities) - batch_size=4, num_classes=3
  std::vector<float> pred_data = {
      0.7f, 0.2f, 0.1f,  // Sample 1: predicts class 0
      0.1f, 0.8f, 0.1f,  // Sample 2: predicts class 1
      0.2f, 0.3f, 0.5f,  // Sample 3: predicts class 2
      0.6f, 0.3f, 0.1f   // Sample 4: predicts class 0
  };

  // Ground truth labels
  std::vector<int> target_data = {0, 1, 2, 1};  // True classes

  // Create tensors
  Tensor predictions = make_tensor<float>({4, 3});
  Tensor targets = make_tensor<int>({4});

  // Copy data
  std::copy(pred_data.begin(), pred_data.end(), predictions->data_as<float>());
  std::copy(target_data.begin(), target_data.end(), targets->data_as<int>());

  // Compute various metrics
  int correct = compute_class_corrects(predictions, targets);
  float accuracy = static_cast<float>(correct) / 4.0f;

  float precision = compute_precision(predictions, targets);
  float recall = compute_recall(predictions, targets);
  float f1 = compute_f1_score(predictions, targets);
  float perplexity = compute_perplexity(predictions, targets);
  float top2_acc = compute_top_k_accuracy(predictions, targets, 2);

  std::cout << "Accuracy: " << accuracy << std::endl;
  std::cout << "Precision (macro-avg): " << precision << std::endl;
  std::cout << "Recall (macro-avg): " << recall << std::endl;
  std::cout << "F1 Score (macro-avg): " << f1 << std::endl;
  std::cout << "Perplexity: " << perplexity << std::endl;
  std::cout << "Top-2 Accuracy: " << top2_acc << std::endl;

  // Per-class metrics
  std::cout << "Per-class metrics:" << std::endl;
  for (int c = 0; c < 3; ++c) {
    float class_precision = compute_precision(predictions, targets, c);
    float class_recall = compute_recall(predictions, targets, c);
    float class_f1 = compute_f1_score(predictions, targets, c);
    std::cout << "Class " << c << " - Precision: " << class_precision
              << ", Recall: " << class_recall << ", F1: " << class_f1 << std::endl;
  }

  // Example: Regression metrics
  std::cout << "Regression Metrics Example" << std::endl;

  std::vector<float> pred_reg = {2.5f, 0.0f, 2.1f, 7.8f};
  std::vector<float> target_reg = {3.0f, -0.5f, 2.0f, 8.0f};

  Tensor reg_predictions = make_tensor<float>({4});
  Tensor reg_targets = make_tensor<float>({4});

  std::copy(pred_reg.begin(), pred_reg.end(), reg_predictions->data_as<float>());
  std::copy(target_reg.begin(), target_reg.end(), reg_targets->data_as<float>());

  float mae = compute_mae(reg_predictions, reg_targets);
  float mse = compute_mse(reg_predictions, reg_targets);
  float rmse = compute_rmse(reg_predictions, reg_targets);

  std::cout << "MAE: " << mae << std::endl;
  std::cout << "MSE: " << mse << std::endl;
  std::cout << "RMSE: " << rmse << std::endl;

  return 0;
}
