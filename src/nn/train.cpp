/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/train.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <chrono>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "device/del_allocator_v2.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "nn/csv_logger.hpp"
#include "nn/graph_executor.hpp"
#include "nn/metrics.hpp"
#include "threading/thread_wrapper.hpp"
#include "type/type.hpp"
#include "utils/env.hpp"

using namespace std;

namespace tnn {

void TrainingConfig::print_config() const {
  cout << "Training Configuration:" << endl;
  cout << "  Epochs: " << epochs << endl;
  cout << "  Batch Size: " << batch_size << endl;
  cout << "  Max Steps: " << max_steps << endl;
  cout << "  Initial Learning Rate: " << lr_initial << endl;
  cout << "  Gradient Accumulation Steps: " << gradient_accumulation_steps << endl;
  cout << "  Progress Print Interval (batches): " << progress_print_interval << endl;
  cout << "  Number of Threads: " << num_threads << endl;
  cout << "  Profiler Type: "
       << (profiler_type == ProfilerType::NONE
               ? "None"
               : (profiler_type == ProfilerType::NORMAL ? "Normal" : "Cumulative"))
       << endl;
  cout << "  Print Layer Profiling Info: " << (print_layer_profiling ? "Yes" : "No") << endl;
  cout << "  Print Layer Memory Usage: " << (print_layer_memory_usage ? "Yes" : "No") << endl;
  cout << "  Number of Microbatches: " << num_microbatches << endl;
  cout << "  Device Type: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;
}

void TrainingConfig::load_from_env() {
  // Get training parameters from environment or use defaults
  Env::get("EPOCHS", epochs);
  Env::get("BATCH_SIZE", batch_size);
  Env::get("MAX_STEPS", max_steps);
  Env::get("LR_INITIAL", lr_initial);
  Env::get("GRADIENT_ACCUMULATION_STEPS", gradient_accumulation_steps);
  Env::get("PROGRESS_PRINT_INTERVAL", progress_print_interval);
  string profiler_type_str = "NONE";
  Env::get("PROFILER_TYPE", profiler_type_str);
  if (profiler_type_str == "NORMAL") {
    profiler_type = ProfilerType::NORMAL;
  } else if (profiler_type_str == "CUMULATIVE") {
    profiler_type = ProfilerType::CUMULATIVE;
  } else {
    profiler_type = ProfilerType::NONE;
  }
  Env::get("NUM_THREADS", num_threads);
  Env::get("PRINT_LAYER_PROFILING", print_layer_profiling);
  Env::get("PRINT_LAYER_MEMORY_USAGE", print_layer_memory_usage);
  Env::get("NUM_MICROBATCHES", num_microbatches);
  string device_type_str = "CPU";
  Env::get("DEVICE_TYPE", device_type_str);
  device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;
  Env::get("MODEL_NAME", model_name);
  Env::get("MODEL_PATH", model_path);
  Env::get("DATASET_NAME", dataset_name);
  Env::get("DATASET_PATH", dataset_path);
  string io_dtype_str = dtype_to_string(io_dtype);
  Env::get("IO_DTYPE", io_dtype_str);
  io_dtype = string_to_dtype(io_dtype_str);
  string param_dtype_str = dtype_to_string(param_dtype);
  Env::get("PARAM_DTYPE", param_dtype_str);
  param_dtype = string_to_dtype(param_dtype_str);
  string compute_dtype_str = dtype_to_string(compute_dtype);
  Env::get("COMPUTE_DTYPE", compute_dtype_str);
  compute_dtype = string_to_dtype(compute_dtype_str);

  // Parse LogMode settings
  Env::get("LOG_LOSS", log_mode.log_loss);
  Env::get("LOG_ACCURACY", log_mode.log_accuracy);
  Env::get("LOG_PRECISION", log_mode.log_precision);
  Env::get("LOG_RECALL", log_mode.log_recall);
  Env::get("LOG_F1_SCORE", log_mode.log_f1_score);
  Env::get("LOG_PERPLEXITY", log_mode.log_perplexity);
  Env::get("LOG_TOP_K_ACCURACY", log_mode.log_top_k_accuracy);
  Env::get("LOG_MAE", log_mode.log_mae);
  Env::get("LOG_MSE", log_mode.log_mse);
  Env::get("LOG_RMSE", log_mode.log_rmse);
}

void TrainingConfig::load_from_json(const string &config_path) {
  ifstream file(config_path);
  if (!file.is_open()) {
    throw runtime_error("Failed to open config file: " + config_path);
  }

  nlohmann::json config;
  file >> config;
  file.close();

  epochs = config.value("epochs", epochs);
  batch_size = config.value("batch_size", batch_size);
  max_steps = config.value("max_steps", max_steps);
  lr_initial = config.value("lr_initial", lr_initial);
  gradient_accumulation_steps =
      config.value("gradient_accumulation_steps", gradient_accumulation_steps);
  progress_print_interval = config.value("progress_print_interval", progress_print_interval);
  num_threads = config.value("num_threads", num_threads);
  string profiler_type_str = config.value("profiler_type", "NONE");
  if (profiler_type_str == "NORMAL") {
    profiler_type = ProfilerType::NORMAL;
  } else if (profiler_type_str == "CUMULATIVE") {
    profiler_type = ProfilerType::CUMULATIVE;
  } else {
    profiler_type = ProfilerType::NONE;
  }
  print_layer_profiling = config.value("print_layer_profiling", print_layer_profiling);
  print_layer_memory_usage = config.value("print_layer_memory_usage", print_layer_memory_usage);
  num_microbatches = config.value("num_microbatches", num_microbatches);
  if (config.contains("device_type")) {
    string device_str = config["device_type"];
    device_type = (device_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;
  }
  model_name = config.value("model_name", model_name);
  model_path = config.value("model_path", model_path);
  dataset_name = config.value("dataset_name", dataset_name);
  dataset_path = config.value("dataset_path", dataset_path);
  string io_dtype_str = config.value("io_dtype", dtype_to_string(io_dtype));
  io_dtype = string_to_dtype(io_dtype_str);
  string param_dtype_str = config.value("param_dtype", dtype_to_string(param_dtype));
  param_dtype = string_to_dtype(param_dtype_str);
  string compute_dtype_str = config.value("compute_dtype", dtype_to_string(compute_dtype));
  compute_dtype = string_to_dtype(compute_dtype_str);

  // Parse LogMode settings from JSON
  if (config.contains("log_mode")) {
    auto log_config = config["log_mode"];
    log_mode.log_loss = log_config.value("log_loss", log_mode.log_loss);
    log_mode.log_accuracy = log_config.value("log_accuracy", log_mode.log_accuracy);
    log_mode.log_precision = log_config.value("log_precision", log_mode.log_precision);
    log_mode.log_recall = log_config.value("log_recall", log_mode.log_recall);
    log_mode.log_f1_score = log_config.value("log_f1_score", log_mode.log_f1_score);
    log_mode.log_perplexity = log_config.value("log_perplexity", log_mode.log_perplexity);
    log_mode.log_top_k_accuracy =
        log_config.value("log_top_k_accuracy", log_mode.log_top_k_accuracy);
    log_mode.log_mae = log_config.value("log_mae", log_mode.log_mae);
    log_mode.log_mse = log_config.value("log_mse", log_mode.log_mse);
    log_mode.log_rmse = log_config.value("log_rmse", log_mode.log_rmse);
  }
}

static Result train_epoch(Graph &graph, unique_ptr<BaseDataLoader> &train_loader,
                          unique_ptr<Optimizer> &optimizer, const unique_ptr<Loss> &criterion,
                          unique_ptr<Scheduler> &scheduler, const TrainingConfig &config,
                          CsvLogger &logger, int epoch) {
  auto train_start = chrono::high_resolution_clock::now();
  Tensor batch_data, batch_labels;
  const Device &model_device = graph.device();
  auto &mem_pool = PoolAllocator::instance(model_device, defaultFlowHandle);
  auto ws_allocator = DELAllocatorV2::instance(model_device, defaultFlowHandle);
  GraphExecutor executor(graph, ws_allocator);

  cout << "Starting training epoch..." << endl;
  graph.set_training(true);
  train_loader->shuffle();
  train_loader->reset();

  float total_loss = 0.0;
  int total_corrects = 0;
  size_t total_class_num = 0;
  int num_batches = 0;
  int grad_accum_counter = 0;

  cout << "Training batches: " << train_loader->size() << endl;
  while (train_loader->get_batch(config.batch_size, batch_data, batch_labels) &&
         (config.max_steps == -1 || num_batches < config.max_steps)) {
    auto batch_start = chrono::high_resolution_clock::now();
    ++num_batches;
    auto device_labels = batch_labels->to_device(model_device);

    Tensor predictions = make_tensor(mem_pool, batch_data->data_type());

    const InputPack inputs{
        {"input", &batch_data},
    };
    OutputPack outputs{
        {"output", &predictions},
    };

    executor.forward(inputs, outputs);

    size_t batch_size = 1;
    for (size_t i = 0; i < predictions->dims() - 1; ++i) {
      batch_size *= predictions->shape()[i];
    }
    total_class_num += batch_size;

    float loss;
    criterion->compute_loss(predictions, device_labels, loss);
    total_loss += loss;

    int batch_corrects = compute_class_corrects(predictions, device_labels);
    total_corrects += batch_corrects;

    // Compute additional metrics before freeing predictions
    std::unordered_map<std::string, double> batch_metrics;
    if (config.log_mode.log_precision) {
      batch_metrics["precision"] = compute_precision(predictions, device_labels);
    }
    if (config.log_mode.log_recall) {
      batch_metrics["recall"] = compute_recall(predictions, device_labels);
    }
    if (config.log_mode.log_f1_score) {
      batch_metrics["f1_score"] = compute_f1_score(predictions, device_labels);
    }
    if (config.log_mode.log_perplexity) {
      batch_metrics["perplexity"] = compute_perplexity(predictions, device_labels);
    }
    if (config.log_mode.log_top_k_accuracy) {
      batch_metrics["top_k_accuracy"] = compute_top_k_accuracy(predictions, device_labels, 5);
    }

    Tensor loss_gradient = make_tensor(mem_pool, batch_data->data_type(), predictions->shape());
    criterion->compute_gradient(predictions, device_labels, loss_gradient);

    predictions = nullptr;  // free prediction buffer early

    Tensor backward_output = make_tensor(mem_pool, batch_data->data_type(), batch_data->shape());

    const InputPack grad_inputs{
        {"output", &loss_gradient},
    };
    OutputPack grad_outputs{
        {"input", &backward_output},
    };
    executor.backward(grad_inputs, grad_outputs);

    auto batch_end = chrono::high_resolution_clock::now();
    auto batch_duration = chrono::duration_cast<chrono::milliseconds>(batch_end - batch_start);

    if (++grad_accum_counter == config.gradient_accumulation_steps) {
      grad_accum_counter = 0;
      optimizer->update();
      optimizer->zero_grads();
      if (scheduler) {
        scheduler->step();
      }
    }
    model_device.getFlow(defaultFlowHandle)->synchronize();

    // Log batch metrics
    {
      double batch_acc_pct = total_class_num > 0 ? (total_corrects * 100.0 / total_class_num) : 0.0;

      if (config.log_mode.log_loss) {
        batch_metrics["loss"] = loss;
      }
      if (config.log_mode.log_accuracy) {
        batch_metrics["accuracy_pct"] = batch_acc_pct;
      }
      batch_metrics["time_ms"] = batch_duration.count();

      logger.log_batch(epoch, num_batches, batch_metrics);
    }

    if (num_batches % config.progress_print_interval == 0) {
      cout << "Batch ID: " << num_batches << ", Batch's Loss: " << fixed << setprecision(4) << loss
           << ", Cumulative Accuracy: " << setprecision(2)
           << (total_corrects * 100.0 / total_class_num) << "%";
      if (config.log_mode.log_f1_score && batch_metrics.count("f1_score")) {
        cout << ", F1: " << setprecision(4) << batch_metrics["f1_score"];
      }
      if (config.log_mode.log_perplexity && batch_metrics.count("perplexity")) {
        cout << ", PPL: " << setprecision(2) << batch_metrics["perplexity"];
      }
      cout << ", Batch Time: " << batch_duration.count() << "ms" << endl;
    }
  }
  cout << endl;

  const double avg_train_loss = total_loss / num_batches;
  const double avg_train_accuracy = static_cast<double>(total_corrects) / total_class_num;

  auto train_end = chrono::high_resolution_clock::now();
  auto train_epoch_duration = chrono::duration_cast<chrono::milliseconds>(train_end - train_start);
  cout << "Training epoch completed in " << train_epoch_duration.count() << "ms" << endl;
  return {avg_train_loss, avg_train_accuracy};
}

static void train_val(Graph &graph, unique_ptr<BaseDataLoader> &train_loader,
                      unique_ptr<BaseDataLoader> &val_loader, unique_ptr<Optimizer> &optimizer,
                      const unique_ptr<Loss> &criterion, unique_ptr<Scheduler> &scheduler,
                      const TrainingConfig &config) {
  ThreadWrapper thread_wrapper({config.num_threads});

  double best_val_accuracy = 0.0;
  CsvLogger logger("tnn_" + graph.name(), config.log_dir, &config.log_mode);

  thread_wrapper.execute([&]() -> void {
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      cout << "Epoch " << epoch + 1 << "/" << config.epochs << endl;

      // train phrase
      auto [avg_train_loss, avg_train_accuracy] = train_epoch(
          graph, train_loader, optimizer, criterion, scheduler, config, logger, epoch + 1);

      // validation phrase
      auto [avg_val_loss, avg_val_accuracy] =
          validate_model(graph, val_loader, criterion, config, &logger, epoch + 1);

      if (avg_val_accuracy > best_val_accuracy) {
        best_val_accuracy = avg_val_accuracy;
        cout << "New best validation accuracy: " << fixed << setprecision(2)
             << best_val_accuracy * 100.0 << "%" << endl;
        try {
          filesystem::create_directories("model_snapshots");
          string filepath = "model_snapshots/" + graph.name();
          ofstream file(filepath, ios::binary);
          if (!file.is_open()) {
            throw runtime_error("Failed to open file: " + filepath);
          }
          graph.save_state(file);
          file.close();
          cout << "Model saved to " << filepath << endl;
        } catch (const exception &e) {
          cerr << "Error saving model: " << e.what() << endl;
        }
      }

      cout << string(60, '-') << endl;
      cout << "Epoch " << epoch + 1 << "/" << config.epochs << endl;
      cout << "Training   - Loss: " << fixed << setprecision(4) << avg_train_loss
           << ", Accuracy: " << setprecision(2) << avg_train_accuracy * 100.0 << "%" << endl;
      cout << "Validation - Loss: " << fixed << setprecision(4) << avg_val_loss
           << ", Accuracy: " << setprecision(2) << avg_val_accuracy * 100.0 << "%" << endl;
      cout << string(60, '=') << endl;

      if ((epoch + 1) % 5 == 0) {
        thread_wrapper.clean_buffers();
      }

      // Log epoch metrics
      {
        std::unordered_map<std::string, double> metrics;
        if (config.log_mode.log_loss) {
          metrics["train_loss"] = avg_train_loss;
          metrics["val_loss"] = avg_val_loss;
        }
        if (config.log_mode.log_accuracy) {
          metrics["train_accuracy_pct"] = avg_train_accuracy * 100.0;
          metrics["val_accuracy_pct"] = avg_val_accuracy * 100.0;
        }
        logger.log_epoch(epoch + 1, metrics);
      }
    }
  });
}

static void train_step(Graph &graph, unique_ptr<BaseDataLoader> &train_loader,
                       const unique_ptr<Optimizer> &optimizer, const unique_ptr<Loss> &criterion,
                       const unique_ptr<Scheduler> &scheduler, const TrainingConfig &config) {
  ThreadWrapper thread_wrapper({config.num_threads});

  Tensor batch_data, batch_labels;
  cout << "Starting training epoch..." << endl;
  graph.set_training(true);
  train_loader->shuffle();
  train_loader->reset();

  const Device &model_device = graph.device();
  auto &mem_pool = PoolAllocator::instance(model_device, defaultFlowHandle);
  auto ws_allocator = DELAllocatorV2::instance(model_device, defaultFlowHandle);
  GraphExecutor executor(graph, ws_allocator);

  int grad_accum_counter = 0;
  CsvLogger logger("tnn_" + graph.name(), config.log_dir, &config.log_mode);

  train_loader->reset();
  auto start_time = chrono::high_resolution_clock::now();

  thread_wrapper.execute([&]() -> void {
    for (int steps = 0; steps < config.max_steps; ++steps) {
      if (!train_loader->get_batch(config.batch_size, batch_data, batch_labels)) {
        break;
      }
      auto batch_start = chrono::high_resolution_clock::now();
      Tensor predictions;
      Tensor device_labels = batch_labels->to_device(model_device);
      const InputPack inputs{
          {"input", &batch_data},
      };
      OutputPack outputs{
          {"output", &predictions},
      };
      executor.forward(inputs, outputs);
      float loss;
      criterion->compute_loss(predictions, device_labels, loss);

      int corrects = compute_class_corrects(predictions, device_labels);

      // Compute additional metrics before freeing predictions
      std::unordered_map<std::string, double> batch_metrics;
      if (config.log_mode.log_precision) {
        batch_metrics["precision"] = compute_precision(predictions, device_labels);
      }
      if (config.log_mode.log_recall) {
        batch_metrics["recall"] = compute_recall(predictions, device_labels);
      }
      if (config.log_mode.log_f1_score) {
        batch_metrics["f1_score"] = compute_f1_score(predictions, device_labels);
      }
      if (config.log_mode.log_perplexity) {
        batch_metrics["perplexity"] = compute_perplexity(predictions, device_labels);
      }
      if (config.log_mode.log_top_k_accuracy) {
        batch_metrics["top_k_accuracy"] = compute_top_k_accuracy(predictions, device_labels, 5);
      }

      Tensor loss_gradient = make_tensor(mem_pool, batch_data->data_type(), predictions->shape());
      criterion->compute_gradient(predictions, device_labels, loss_gradient);

      Tensor backward_output = make_tensor(mem_pool, batch_data->data_type(), batch_data->shape());
      const InputPack grad_outputs{
          {"output", &loss_gradient},
      };
      OutputPack grad_inputs{
          {"input", &backward_output},
      };
      executor.backward(grad_outputs, grad_inputs);

      backward_output = nullptr;  // free backward output buffer early

      auto batch_end = chrono::high_resolution_clock::now();
      auto batch_duration = chrono::duration_cast<chrono::milliseconds>(batch_end - batch_start);
      if (++grad_accum_counter == config.gradient_accumulation_steps) {
        grad_accum_counter = 0;
        optimizer->update();
        optimizer->zero_grads();
        if (scheduler) {
          scheduler->step();
        }
      }

      size_t num_samples = 1;
      for (size_t i = 0; i < predictions->dims() - 1; ++i) {
        num_samples *= predictions->shape()[i];
      }

      double batch_acc_pct = corrects * 100.0 / num_samples;

      // Log batch metrics for benchmarking.
      {
        if (config.log_mode.log_loss) {
          batch_metrics["loss"] = loss;
        }
        if (config.log_mode.log_accuracy) {
          batch_metrics["accuracy_pct"] = batch_acc_pct;
        }
        batch_metrics["time_ms"] = batch_duration.count();

        logger.log_batch(1, steps, batch_metrics);
      }

      if (steps % config.progress_print_interval == 0) {
        cout << "Batch ID: " << steps << ", Batch's Loss: " << fixed << setprecision(4) << loss
             << ", Batch's Accuracy: " << setprecision(2) << batch_acc_pct << "%";
        if (config.log_mode.log_f1_score && batch_metrics.count("f1_score")) {
          cout << ", F1: " << setprecision(4) << batch_metrics["f1_score"];
        }
        if (config.log_mode.log_perplexity && batch_metrics.count("perplexity")) {
          cout << ", PPL: " << setprecision(2) << batch_metrics["perplexity"];
        }
        cout << ", Batch Time: " << batch_duration.count() << "ms" << endl;
      }
    }

    // training epoch done, print time taken
    auto end_time = chrono::high_resolution_clock::now();
    auto epoch_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Training completed in " << epoch_duration.count() << "ms" << endl;

    // save model
    try {
      filesystem::create_directories("model_snapshots");
      string filepath = "model_snapshots/" + graph.name();
      ofstream file(filepath, ios::binary);
      if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filepath);
      }
      graph.save_state(file);
      file.close();
      cout << "Model saved to " << filepath << endl;
    } catch (const exception &e) {
      cerr << "Error saving model: " << e.what() << endl;
    }
  });
}

void train_model(Graph &graph, unique_ptr<BaseDataLoader> &train_loader,
                 unique_ptr<BaseDataLoader> &val_loader, unique_ptr<Optimizer> &optimizer,
                 const unique_ptr<Loss> &criterion, unique_ptr<Scheduler> &scheduler,
                 const TrainingConfig &config) {
  optimizer->attach(graph.context());

  cout << "Training batches: " << train_loader->size() / config.batch_size << endl;
  cout << "Validation batches: " << val_loader->size() / config.batch_size << endl;

  vector<size_t> data_shape = train_loader->get_data_shape();
  data_shape.insert(data_shape.begin(), config.batch_size);  // add batch dimension

  bool is_val = config.max_steps == -1;

  if (is_val) {
    train_val(graph, train_loader, val_loader, optimizer, criterion, scheduler, config);
  } else {
    train_step(graph, train_loader, optimizer, criterion, scheduler, config);
  }
}

Result validate_model(Graph &graph, unique_ptr<BaseDataLoader> &val_loader,
                      const unique_ptr<Loss> &criterion, const TrainingConfig &config,
                      CsvLogger *logger, int epoch) {
  auto &mem_pool = PoolAllocator::instance(graph.device(), defaultFlowHandle);
  auto ws_allocator = DELAllocatorV2::instance(graph.device(), defaultFlowHandle);
  GraphExecutor executor(graph, ws_allocator);
  Tensor batch_data, batch_labels;

  graph.set_training(false);
  val_loader->reset();

  cout << "Starting validation..." << endl;
  double val_loss = 0.0;
  double val_corrects = 0.0;
  int val_batches = 0;
  csref<Device> model_device = graph.device();

  Tensor device_batch_labels;

  while (val_loader->get_batch(config.batch_size, batch_data, batch_labels)) {
    Tensor device_input = batch_data->to_device(model_device);
    const InputPack inputs{
        {"input", &device_input},
    };
    Tensor predictions = make_tensor<float>(mem_pool, {});
    OutputPack outputs{
        {"output", &predictions},
    };
    executor.forward(inputs, outputs);

    device_batch_labels = batch_labels->to_device(model_device);
    float loss;
    criterion->compute_loss(predictions, device_batch_labels, loss);
    val_loss += loss;
    int batch_corrects = compute_class_corrects(predictions, device_batch_labels);
    val_corrects += batch_corrects;
    ++val_batches;

    if (logger) {
      std::unordered_map<std::string, double> metrics;
      double batch_acc_pct = batch_corrects / static_cast<double>(config.batch_size) * 100.0;

      if (config.log_mode.log_loss) {
        metrics["loss"] = loss;
      }
      if (config.log_mode.log_accuracy) {
        metrics["accuracy_pct"] = batch_acc_pct;
      }
      if (config.log_mode.log_precision) {
        metrics["precision"] = compute_precision(predictions, device_batch_labels);
      }
      if (config.log_mode.log_recall) {
        metrics["recall"] = compute_recall(predictions, device_batch_labels);
      }
      if (config.log_mode.log_f1_score) {
        metrics["f1_score"] = compute_f1_score(predictions, device_batch_labels);
      }
      if (config.log_mode.log_perplexity) {
        metrics["perplexity"] = compute_perplexity(predictions, device_batch_labels);
      }
      if (config.log_mode.log_top_k_accuracy) {
        metrics["top_k_accuracy"] = compute_top_k_accuracy(predictions, device_batch_labels, 5);
      }

      logger->log_val_batch(epoch, val_batches, metrics);
    }
  }

  double avg_val_loss = val_loss / val_batches;
  double avg_val_accuracy = val_corrects / val_loader->size();

  return {avg_val_loss, avg_val_accuracy};
}

}  // namespace tnn
