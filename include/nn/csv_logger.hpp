/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "logging/logger.hpp"

namespace tunx {

struct LogMode {
  bool log_loss = true;
  bool log_accuracy = true;
  bool log_precision = false;
  bool log_recall = false;
  bool log_f1_score = false;
  bool log_perplexity = false;
  bool log_top_k_accuracy = false;
  bool log_mae = false;
  bool log_mse = false;
  bool log_rmse = false;
};

inline std::string csv_timestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  struct tm tm_buf {};
  localtime_r(&t, &tm_buf);
  char ts[20];
  std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);
  return ts;
}

inline long long get_timestamp_ms() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

class CsvLogger {
private:
  Logger train_step_logger_;
  Logger val_step_logger_;
  Logger epoch_logger_;
  std::vector<std::string> train_step_metrics_;
  std::vector<std::string> val_step_metrics_;
  std::vector<std::string> epoch_metrics_;

public:
  CsvLogger(const std::string &model_name, const std::string &log_dir,
            const LogMode *log_mode = nullptr);

  void log_train_step(int epoch, int step, const std::unordered_map<std::string, double> &metrics);

  void log_val_step(int epoch, int step, const std::unordered_map<std::string, double> &metrics);

  void log_epoch(int epoch, const std::unordered_map<std::string, double> &metrics);

  void flush() {
    train_step_logger_.flush();
    val_step_logger_.flush();
    epoch_logger_.flush();
  }
};

class WorkerCsvLogger {
private:
  Logger compute_logger_;

public:
  WorkerCsvLogger(const std::string &worker_name, const std::string &log_dir)
      : compute_logger_(worker_name + "_compute", "") {
    std::filesystem::create_directories(log_dir);
    std::string ts = csv_timestamp();

    std::string path = log_dir + "/" + worker_name + "_compute_" + ts + ".csv";
    compute_logger_.set_log_file(path);
    compute_logger_.set_pattern("%v");
    compute_logger_.info("step,event_type,time_ms");
  }

  void log(int step, const std::string &event_type, long time_ms, size_t device_used_memory_mb) {
    std::ostringstream row;
    row << step << "," << event_type << "," << time_ms << "," << device_used_memory_mb;
    compute_logger_.info(row.str());
  }

  void flush() { compute_logger_.flush(); }
};

}  // namespace tunx
