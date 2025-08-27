#pragma once

#include "data_loader.hpp"
#include "nn/sequential.hpp"

void train_cnn_model(tnn::Sequential<float> &model,
                     data_loading::BaseDataLoader<float> &train_loader,
                     data_loading::BaseDataLoader<float> &test_loader,
                     int epochs = 10, int batch_size = 32,
                     float lr_decay_factor = 0.9f, int progress_print_interval = 100) {

  Tensor<float> batch_data, batch_labels, predictions;

  
  std::cout << "\nPreparing training batches..." << std::endl;
  train_loader.prepare_batches(batch_size);

  std::cout << "Preparing validation batches..." << std::endl;
  test_loader.prepare_batches(batch_size);

  std::cout << "Training batches: " << train_loader.num_batches() << std::endl;
  std::cout << "Validation batches: " << test_loader.num_batches() << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    const auto epoch_start = std::chrono::high_resolution_clock::now();

    
    model.train();
    train_loader.shuffle();
    train_loader.prepare_batches(batch_size); 
    train_loader.reset();

    double total_loss = 0.0; 
    double total_accuracy = 0.0;
    int num_batches = 0;
    std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
    
    while (train_loader.get_next_batch(batch_data, batch_labels)) {
      ++num_batches;
      
      predictions = model.forward(batch_data);
      utils::apply_softmax<float>(predictions);

      
      const float loss =
          model.loss_function()->compute_loss(predictions, batch_labels);
      const float accuracy =
          utils::compute_class_accuracy<float>(predictions, batch_labels);

      total_loss += loss;
      total_accuracy += accuracy;

      
      const Tensor<float> loss_gradient =
          model.loss_function()->compute_gradient(predictions, batch_labels);
      model.backward(loss_gradient);

      
      model.update_parameters();

      
      if (num_batches % progress_print_interval == 0) {
        model.print_profiling_summary();
        std::cout << "Batch ID: " << num_batches
                  << ", Batch's Loss: " << std::fixed << std::setprecision(4)
                  << loss << ", Batch's Accuracy: " << std::setprecision(2)
                  << accuracy * 100.0f << "%" << std::endl;
      }
      model.clear_profiling_data();
    }
    std::cout << std::endl; 

    const float avg_train_loss = static_cast<float>(total_loss / num_batches);
    const float avg_train_accuracy =
        static_cast<float>(total_accuracy / num_batches);

    const auto epoch_end = std::chrono::high_resolution_clock::now();
    const auto epoch_duration =
        std::chrono::duration_cast<std::chrono::seconds>(epoch_end -
                                                         epoch_start);
    
    model.eval();
    test_loader.reset();

    std::cout << "Starting validation..." << std::endl;
    double val_loss = 0.0;
    double val_accuracy = 0.0;
    int val_batches = 0;

    
    while (test_loader.get_next_batch(batch_data, batch_labels)) {
      predictions = model.forward(batch_data);
      utils::apply_softmax<float>(predictions);

      val_loss +=
          model.loss_function()->compute_loss(predictions, batch_labels);
      val_accuracy +=
          utils::compute_class_accuracy<float>(predictions, batch_labels);
      ++val_batches;
    }

    const float avg_val_loss = static_cast<float>(val_loss / val_batches);
    const float avg_val_accuracy =
        static_cast<float>(val_accuracy / val_batches);

    
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed in "
              << epoch_duration.count() << "s" << std::endl;
    std::cout << "Training   - Loss: " << std::fixed << std::setprecision(4)
              << avg_train_loss << ", Accuracy: " << std::setprecision(2)
              << avg_train_accuracy * 100.0f << "%" << std::endl;
    std::cout << "Validation - Loss: " << std::fixed << std::setprecision(4)
              << avg_val_loss << ", Accuracy: " << std::setprecision(2)
              << avg_val_accuracy * 100.0f << "%" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    
    if ((epoch + 1) % progress_print_interval == 0) {
      const float current_lr = model.optimizer()->get_learning_rate();
      const float new_lr = current_lr * lr_decay_factor;
      model.optimizer()->set_learning_rate(new_lr);
      std::cout << "Learning rate decayed: " << std::fixed
                << std::setprecision(6) << current_lr << " → " << new_lr
                << std::endl;
    }
  }
}