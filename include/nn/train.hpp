/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "data_loading/data_loader.hpp"
#include "data_loading/regression_data_loader.hpp"
#include "device/device_type.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "type/type.hpp"

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
enum class TrainingMode { CLASSIFICATION = 0, REGRESSION = 1, CUSTOM = 2 };

#ifdef USE_TBB
inline void tbb_cleanup();
#endif

constexpr int DEFAULT_EPOCH = 10;
constexpr size_t DEFAULT_BATCH_SIZE = 32;
constexpr float DEFAULT_LR_DECAY_FACTOR = 0.9f;
constexpr size_t DEFAULT_LR_DECAY_INTERVAL = 5;  // in epochs
constexpr int DEFAULT_PRINT_INTERVAL = 100;
constexpr int64_t DEFAULT_NUM_THREADS = 8;  // Typical number of P-Cores on laptop CPUs

struct TrainingConfig {
  // Trainer params
  DType_t dtype = DType_t::FP32;
  int epochs = 10;
  size_t batch_size = 32;
  int64_t max_steps = -1;  // -1 for no limit, otherwise max number of batches per epoch
  float lr_initial = 0.001f;
  int gradient_accumulation_steps = 1;
  int progress_print_interval = 100;
  int64_t num_threads = 8;  // Typical number of P-Cores on laptop CPUs
  ProfilerType profiler_type = ProfilerType::NONE;
  bool print_layer_profiling = false;
  bool print_layer_memory_usage = false;
  DeviceType device_type = DeviceType::CPU;

  // Distributed params
  size_t num_microbatches = 2;

  void print_config() const;
  void load_from_env();
};

struct Result {
  double avg_loss = 0.0f;
  double avg_accuracy = -1.0f;
};

Result validate_model(std::unique_ptr<Sequential> &model,
                      std::unique_ptr<BaseDataLoader> &val_loader,
                      const std::unique_ptr<Loss> &criterion, const TrainingConfig &config);

void train_model(std::unique_ptr<Sequential> &model, std::unique_ptr<BaseDataLoader> &train_loader,
                 std::unique_ptr<BaseDataLoader> &val_loader, std::unique_ptr<Optimizer> &optimizer,
                 const std::unique_ptr<Loss> &criterion, std::unique_ptr<Scheduler> &scheduler,
                 const TrainingConfig &config = TrainingConfig());

}  // namespace tnn