/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <nlohmann/json.hpp>

#include "data_loading/data_loader.hpp"
#include "data_loading/regression_data_loader.hpp"
#include "device/device_type.hpp"
#include "nn/csv_logger.hpp"
#include "nn/graph_api.hpp"
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

struct TrainingConfig {
  // Trainer params
  int epochs = 10;
  size_t batch_size = 32;
  int64_t max_steps = -1;  // -1 for no limit, otherwise max number of batches per epoch
  //   "epoch" -> full dataloader epoch, then validation + epoch CSV summary.
  //   "batch" -> fixed-step training loop for GPT/OpenWebText style runs.
  //   "auto"  -> old behavior: epoch if max_steps == -1, batch otherwise.
  std::string train_mode = "auto";
  float lr_initial = 0.001f;
  int gradient_accumulation_steps = 1;
  int progress_print_interval = 100;
  int64_t num_threads = 8;  // Typical number of P-Cores on laptop CPUs
  ProfilerType profiler_type = ProfilerType::NONE;
  bool print_layer_profiling = false;
  bool print_layer_memory_usage = false;
  std::string model_name =
      "cifar10_resnet9";          // If set, will try to load this model from example models
  std::string model_path = "";    // If set, will try to load model from this path before training
  std::string dataset_name = "";  // e.g., "cifar10", "mnist", "open-web-text", etc.
  std::string dataset_path = "data";
  DeviceType device_type = DeviceType::CPU;
  DType_t io_dtype = DType_t::FP32;
  DType_t param_dtype = DType_t::FP32;
  DType_t compute_dtype = DType_t::FP32;
  std::string log_dir = "logs";  // directory for CSV metric logs

  bool prefetch_data = false;
  size_t prefetch_depth = 2;
  bool async_pipeline = true;
  bool augmentation = true;

  LogMode log_mode;  // what metrics to log

  size_t num_microbatches = 2;

  OptimizerConfig optimizer_config;
  SchedulerConfig scheduler_config;
  LossConfig loss_config;

  void print_config() const;
  void load_from_env();
  void load_from_json(const std::string &config_path);
};

struct Result {
  double avg_loss = 0.0f;
  double avg_accuracy = -1.0f;
};

Result validate_model(Graph &graph, std::unique_ptr<BaseDataLoader> &val_loader,
                      const std::unique_ptr<Loss> &criterion, const TrainingConfig &config,
                      CsvLogger *logger = nullptr, int epoch = 0);

void train_model(Graph &graph, std::unique_ptr<BaseDataLoader> &train_loader,
                 std::unique_ptr<BaseDataLoader> &val_loader, std::unique_ptr<Optimizer> &optimizer,
                 const std::unique_ptr<Loss> &criterion, std::unique_ptr<Scheduler> &scheduler,
                 const TrainingConfig &config = TrainingConfig());

}  // namespace tnn