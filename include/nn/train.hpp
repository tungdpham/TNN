/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "data_loading/image_data_loader.hpp"
#include "data_loading/regression_data_loader.hpp"
#include "device/device_type.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "nn/sequential.hpp"
#include "utils/env.hpp"
#include "utils/memory.hpp"
#include "utils/utils_extended.hpp"

#ifdef USE_TBB
#include <tbb/info.h>
#include <tbb/scalable_allocator.h>
#include <tbb/task_arena.h>
#endif

#ifdef USE_MK
#include <mkl.h>
#endif

namespace tnn {
enum class ProfilerType { NONE = 0, NORMAL = 1, CUMULATIVE = 2 };

#ifdef USE_TBB
inline void tbb_cleanup();
#endif

constexpr int DEFAULT_EPOCH = 10;
constexpr size_t DEFAULT_BATCH_SIZE = 32;
constexpr float DEFAULT_LR_DECAY_FACTOR = 0.9f;
constexpr size_t DEFAULT_LR_DECAY_INTERVAL = 5; // in epochs
constexpr int DEFAULT_PRINT_INTERVAL = 100;
constexpr int64_t DEFAULT_NUM_THREADS = 8; // Typical number of P-Cores on laptop CPUs

struct TrainingConfig {
  // Trainer params
  int epochs = 10;
  size_t batch_size = 32;
  float lr_decay_factor = 0.9f;
  size_t lr_decay_interval = 5; // in epochs
  int progress_print_interval = 100;
  int64_t num_threads = 8; // Typical number of P-Cores on laptop CPUs
  ProfilerType profiler_type = ProfilerType::NONE;
  bool print_layer_profiling = false;
  bool print_layer_memory_usage = false;
  DeviceType device_type = DeviceType::CPU;

  // Distributed params
  size_t num_microbatches = 2;

  void print_config() const;
  void load_from_env();
};

struct ClassResult {
  float avg_loss = 0.0f;
  float avg_accuracy = 0.0f;
};

struct RegResult {
  float avg_loss = 0.0f;
  float avg_error = 0.0f;
};

// Classification training functions
template <typename T>
ClassResult train_class_epoch(Sequential<T> &model, ImageDataLoader<T> &train_loader,
                              Optimizer<T> &optimizer, Loss<T> &loss_function,
                              const TrainingConfig &config = TrainingConfig());

template <typename T>
ClassResult validate_class_model(Sequential<T> &model, ImageDataLoader<T> &test_loader,
                                 Loss<T> &loss_function);

template <typename T>
void train_classification_model(Sequential<T> &model, ImageDataLoader<T> &train_loader,
                                ImageDataLoader<T> &test_loader,
                                std::unique_ptr<Optimizer<T>> optimizer,
                                std::unique_ptr<Loss<T>> loss_function,
                                std::unique_ptr<Scheduler<T>> scheduler,
                                const TrainingConfig &config = TrainingConfig());

// Regression training functions
template <typename T>
RegResult train_reg_epoch(Sequential<T> &model, RegressionDataLoader<T> &train_loader,
                          Optimizer<T> &optimizer, Loss<T> &loss_function,
                          const TrainingConfig &config = TrainingConfig());

template <typename T>
RegResult validate_reg_model(Sequential<T> &model, RegressionDataLoader<T> &test_loader,
                             Loss<T> &loss_function);

template <typename T>
void train_regression_model(Sequential<T> &model, RegressionDataLoader<T> &train_loader,
                            RegressionDataLoader<T> &test_loader,
                            std::unique_ptr<Optimizer<T>> optimizer,
                            std::unique_ptr<Loss<T>> loss_function,
                            std::unique_ptr<Scheduler<T>> scheduler,
                            const TrainingConfig &config = TrainingConfig());

} // namespace tnn