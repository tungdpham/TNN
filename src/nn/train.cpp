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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "device/del_allocator_v2.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "nn/accuracy.hpp"
#include "nn/graph_executor.hpp"
#include "threading/thread_wrapper.hpp"
#include "type/type.hpp"
#include "utils/env.hpp"
#include "utils/memory.hpp"

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
  epochs = Env::get<int>("EPOCHS", DEFAULT_EPOCH);
  batch_size = Env::get<size_t>("BATCH_SIZE", DEFAULT_BATCH_SIZE);
  max_steps = Env::get<uint64_t>("MAX_STEPS", -1);  // -1 for no limit
  lr_initial = Env::get<float>("LR_INITIAL", 0.001f);
  gradient_accumulation_steps = Env::get<int>("GRADIENT_ACCUMULATION_STEPS", 1);
  progress_print_interval = Env::get<int>("PROGRESS_PRINT_INTERVAL", DEFAULT_PRINT_INTERVAL);
  string profiler_type_str = Env::get<string>("PROFILER_TYPE", "NONE");
  if (profiler_type_str == "NORMAL") {
    profiler_type = ProfilerType::NORMAL;
  } else if (profiler_type_str == "CUMULATIVE") {
    profiler_type = ProfilerType::CUMULATIVE;
  } else {
    profiler_type = ProfilerType::NONE;
  }
  num_threads = Env::get<int64_t>("NUM_THREADS", DEFAULT_NUM_THREADS);
  print_layer_profiling = Env::get<bool>("PRINT_LAYER_PROFILING", false);
  print_layer_memory_usage = Env::get<bool>("PRINT_LAYER_MEMORY_USAGE", false);
  num_microbatches = Env::get<size_t>("NUM_MICROBATCHES", 2);
  string device_type_str = Env::get<string>("DEVICE_TYPE", "CPU");
  device_type = (device_type_str == "CPU") ? DeviceType::CPU : DeviceType::GPU;
  model_name = Env::get<string>("MODEL_NAME", model_name);
  model_path = Env::get<string>("MODEL_PATH", model_path);
  dataset_name = Env::get<string>("DATASET_NAME", dataset_name);
  dataset_path = Env::get<string>("DATASET_PATH", dataset_path);
  string io_dtype_str = Env::get<string>("IO_DTYPE", dtype_to_string(io_dtype));
  io_dtype = string_to_dtype(io_dtype_str);
  string param_dtype_str = Env::get<string>("PARAM_DTYPE", dtype_to_string(param_dtype));
  param_dtype = string_to_dtype(param_dtype_str);
  string compute_dtype_str = Env::get<string>("COMPUTE_DTYPE", dtype_to_string(compute_dtype));
  compute_dtype = string_to_dtype(compute_dtype_str);
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
}

static Result train_epoch(Graph &graph, unique_ptr<BaseDataLoader> &train_loader,
                          unique_ptr<Optimizer> &optimizer, const unique_ptr<Loss> &criterion,
                          unique_ptr<Scheduler> &scheduler, const TrainingConfig &config) {
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
  double total_corrects = 0.0;
  size_t cur_samples = 0;
  int num_batches = 0;
  int grad_accum_counter = 0;

  cout << "Training batches: " << train_loader->size() << endl;
  while (train_loader->get_batch(config.batch_size, batch_data, batch_labels) &&
         (config.max_steps == -1 || num_batches < config.max_steps)) {
    auto batch_start = chrono::high_resolution_clock::now();
    ++num_batches;
    cur_samples += batch_data->dimension(0);
    auto device_labels = batch_labels->to_device(model_device);

    Tensor predictions = make_tensor(mem_pool, batch_data->data_type());

    const InputPack inputs{
        {"input", &batch_data},
    };
    OutputPack outputs{
        {"output", &predictions},
    };

    executor.forward(inputs, outputs);

    float loss;
    criterion->compute_loss(predictions, device_labels, loss);
    total_loss += loss;

    total_corrects += compute_class_corrects(predictions, device_labels);

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

    if (num_batches % config.progress_print_interval == 0) {
      cout << "Batch ID: " << num_batches << ", Batch's Loss: " << fixed << setprecision(4) << loss
           << ", Cumulative Accuracy: " << setprecision(2) << (total_corrects * 100.0 / cur_samples)
           << "%" << ", Batch Time: " << batch_duration.count() << "ms" << endl;
    }
  }
  cout << endl;

  const double avg_train_loss = total_loss / num_batches;
  const double avg_train_accuracy = total_corrects / cur_samples;

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

  thread_wrapper.execute([&]() -> void {
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
      cout << "Epoch " << epoch + 1 << "/" << config.epochs << endl;

      // train phrase
      auto [avg_train_loss, avg_train_accuracy] =
          train_epoch(graph, train_loader, optimizer, criterion, scheduler, config);

      // validation phrase
      auto [avg_val_loss, avg_val_accuracy] = validate_model(graph, val_loader, criterion, config);

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

      cout << get_memory_usage_kb() / 1024 << " MB of memory used." << endl;
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

      if (steps % config.progress_print_interval == 0) {
        cout << "Batch ID: " << steps << ", Batch's Loss: " << fixed << setprecision(4) << loss
             << ", Batch's Accuracy: " << setprecision(2) << (corrects * 100.0 / config.batch_size)
             << "%" << ", Batch Time: " << batch_duration.count() << "ms" << endl;
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
                      const unique_ptr<Loss> &criterion, const TrainingConfig &config) {
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
    val_corrects += compute_class_corrects(predictions, device_batch_labels);
    ++val_batches;
  }

  double avg_val_loss = val_loss / val_batches;
  double avg_val_accuracy = val_corrects / val_loader->size();

  return {avg_val_loss, avg_val_accuracy};
}

}  // namespace tnn
