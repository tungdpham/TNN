/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/train.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>

#include "device/pool_allocator.hpp"
#include "nn/accuracy.hpp"
#include "nn/sequential.hpp"
#include "tensor/tensor.hpp"
#include "threading/thread_wrapper.hpp"
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
}

static Result train_epoch(unique_ptr<Sequential> &model, unique_ptr<BaseDataLoader> &train_loader,
                          unique_ptr<Optimizer> &optimizer, const unique_ptr<Loss> &criterion,
                          unique_ptr<Scheduler> &scheduler, const TrainingConfig &config) {
  auto train_start = chrono::high_resolution_clock::now();
  Tensor batch_data = Tensor::create(config.dtype), batch_labels = Tensor::create(config.dtype);
  cout << "Starting training epoch..." << endl;
  model->set_training(true);
  train_loader->shuffle();
  train_loader->reset();

  float total_loss = 0.0;
  double total_corrects = 0.0;
  size_t cur_samples = 0;
  int num_batches = 0;
  csref<Device> model_device = model->get_device();

  PoolAllocator &mem_pool = PoolAllocator::instance(*model_device);

  Tensor device_labels = Tensor::create_pooled(mem_pool, config.dtype);
  Tensor loss_gradient = Tensor::create_pooled(mem_pool, model->get_io_dtype());
  Tensor predictions = Tensor::create_pooled(mem_pool, model->get_io_dtype()),
         backward_output = Tensor::create_pooled(mem_pool, model->get_io_dtype());

  int grad_accum_counter = 0;

  cout << "Training batches: " << train_loader->size() << endl;
  while (train_loader->get_batch(config.batch_size, batch_data, batch_labels) &&
         (config.max_steps == -1 || num_batches < config.max_steps)) {
    ++num_batches;
    cur_samples += batch_data->dimension(0);
    device_labels = batch_labels->to_device(model_device);

    model->forward(batch_data, predictions);

    float loss;
    criterion->compute_loss(predictions, device_labels, loss);

    total_loss += loss;
    total_corrects += compute_class_corrects(predictions, device_labels);

    criterion->compute_gradient(predictions, device_labels, loss_gradient);

    model->backward(loss_gradient, backward_output);

    if (num_batches % config.progress_print_interval == 0) {
      if (model->is_profiling_enabled()) {
        model->print_profiling_info();
      }
      cout << "Batch ID: " << num_batches << ", Batch's Loss: " << fixed << setprecision(4) << loss
           << ", Cumulative Accuracy: " << setprecision(2) << (total_corrects * 100.0 / cur_samples)
           << "%" << endl;
    }
    if (model->is_profiling_enabled() && config.profiler_type == ProfilerType::NORMAL) {
      model->reset_profiling_info();
    }
    if (++grad_accum_counter == config.gradient_accumulation_steps) {
      grad_accum_counter = 0;
      optimizer->update();
      optimizer->clear_gradients();
      if (scheduler) {
        scheduler->step();
      }
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

static void train_val(unique_ptr<Sequential> &model, unique_ptr<BaseDataLoader> &train_loader,
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
          train_epoch(model, train_loader, optimizer, criterion, scheduler, config);

      // validation phrase
      auto [avg_val_loss, avg_val_accuracy] = validate_model(model, val_loader, criterion, config);

      if (avg_val_accuracy > best_val_accuracy) {
        best_val_accuracy = avg_val_accuracy;
        cout << "New best validation accuracy: " << fixed << setprecision(2)
             << best_val_accuracy * 100.0 << "%" << endl;
        try {
          filesystem::create_directories("model_snapshots");
          string filepath = "model_snapshots/" + model->name();
          ofstream file(filepath, ios::binary);
          if (!file.is_open()) {
            throw runtime_error("Failed to open file: " + filepath);
          }
          model->save_state(file);
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

      if (model->is_profiling_enabled()) {
        model->reset_profiling_info();
      }

      if ((epoch + 1) % 5 == 0) {
        thread_wrapper.clean_buffers();
      }

      cout << get_memory_usage_kb() / 1024 << " MB of memory used." << endl;
    }
  });
}

static void train_step(unique_ptr<Sequential> &model, unique_ptr<BaseDataLoader> &train_loader,
                       const unique_ptr<Optimizer> &optimizer, const unique_ptr<Loss> &criterion,
                       const unique_ptr<Scheduler> &scheduler, const TrainingConfig &config) {
  ThreadWrapper thread_wrapper({config.num_threads});

  Tensor batch_data, batch_labels;
  cout << "Starting training epoch..." << endl;
  model->set_training(true);
  train_loader->shuffle();
  train_loader->reset();

  csref<Device> model_device = model->get_device();
  Tensor loss_gradient = Tensor::create(model->get_io_dtype(), {1}, model_device);
  Tensor device_labels = Tensor::create(model->get_io_dtype(), {1}, model_device);
  Tensor predictions = Tensor::create(model->get_io_dtype(), {1}, model_device);
  Tensor backward_output = Tensor::create(model->get_io_dtype(), {1}, model_device);

  int grad_accum_counter = 0;

  thread_wrapper.execute([&]() -> void {
    for (int steps = 0; steps < config.max_steps; ++steps) {
      if (!train_loader->get_batch(config.batch_size, batch_data, batch_labels)) {
        break;
      }
      device_labels = batch_labels->to_device(model_device);
      model->forward(batch_data, predictions);
      float loss;
      criterion->compute_loss(predictions, device_labels, loss);

      criterion->compute_gradient(predictions, device_labels, loss_gradient);

      model->backward(loss_gradient, backward_output);

      if (++grad_accum_counter == config.gradient_accumulation_steps) {
        grad_accum_counter = 0;
        optimizer->update();
        optimizer->clear_gradients();
        if (scheduler) {
          scheduler->step();
        }
      }

      if (steps % config.progress_print_interval == 0) {
        if (model->is_profiling_enabled()) {
          model->print_profiling_info();
        }
        cout << "Batch ID: " << steps << ", Batch's Loss: " << fixed << setprecision(4) << loss
             << endl;
      }
      if (model->is_profiling_enabled() && config.profiler_type == ProfilerType::NORMAL) {
        model->reset_profiling_info();
      }
    }

    // save model
    try {
      filesystem::create_directories("model_snapshots");
      string filepath = "model_snapshots/" + model->name();
      ofstream file(filepath, ios::binary);
      if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filepath);
      }
      model->save_state(file);
      file.close();
      cout << "Model saved to " << filepath << endl;
    } catch (const exception &e) {
      cerr << "Error saving model: " << e.what() << endl;
    }
  });
}

void train_model(unique_ptr<Sequential> &model, unique_ptr<BaseDataLoader> &train_loader,
                 unique_ptr<BaseDataLoader> &val_loader, unique_ptr<Optimizer> &optimizer,
                 const unique_ptr<Loss> &criterion, unique_ptr<Scheduler> &scheduler,
                 const TrainingConfig &config) {
  optimizer->attach(model->parameters(), model->gradients());
  Tensor batch_data, batch_labels;

  if (config.profiler_type == ProfilerType::NONE) {
    model->enable_profiling(false);
  } else if (config.profiler_type == ProfilerType::NORMAL ||
             config.profiler_type == ProfilerType::CUMULATIVE) {
    model->enable_profiling(true);
  }

  cout << "Training batches: " << train_loader->size() << endl;
  cout << "Validation batches: " << val_loader->size() << endl;

  vector<size_t> data_shape = train_loader->get_data_shape();
  data_shape.insert(data_shape.begin(), config.batch_size);  // add batch dimension
  model->print_summary(data_shape);

  bool is_val = config.max_steps == -1;

  if (is_val) {
    train_val(model, train_loader, val_loader, optimizer, criterion, scheduler, config);
  } else {
    train_step(model, train_loader, optimizer, criterion, scheduler, config);
  }
}

Result validate_model(unique_ptr<Sequential> &model, unique_ptr<BaseDataLoader> &val_loader,
                      const unique_ptr<Loss> &criterion, const TrainingConfig &config) {
  PoolAllocator &mem_pool = PoolAllocator::instance(model->get_device());
  Tensor batch_data = Tensor::create_pooled(mem_pool, model->get_io_dtype()),
         batch_labels = Tensor::create_pooled(mem_pool, model->get_io_dtype());

  model->set_training(false);
  val_loader->reset();

  cout << "Starting validation..." << endl;
  double val_loss = 0.0;
  double val_corrects = 0.0;
  int val_batches = 0;
  csref<Device> model_device = model->get_device();

  Tensor device_batch_labels = Tensor::create(model->get_io_dtype(), {}, model_device);
  Tensor predictions = Tensor::create(model->get_io_dtype(), {}, model_device);

  while (val_loader->get_batch(config.batch_size, batch_data, batch_labels)) {
    model->forward(batch_data, predictions);

    device_batch_labels = batch_labels->to_device(model_device);
    float loss;
    criterion->compute_loss(predictions, device_batch_labels, loss);
    val_loss += loss;
    val_corrects += compute_class_corrects(predictions, device_batch_labels);
    ++val_batches;
  }

  const double avg_val_loss = val_loss / val_batches;
  const double avg_val_accuracy = val_corrects / val_loader->size();

  return {avg_val_loss, avg_val_accuracy};
}

}  // namespace tnn
