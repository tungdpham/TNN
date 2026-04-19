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

namespace tnn {

struct LogMode;  // Forward declaration

inline std::string csv_timestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  struct tm tm_buf {};
  localtime_r(&t, &tm_buf);
  char ts[20];
  std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);
  return ts;
}

struct CsvLogger {
  Logger batch_logger;
  Logger val_logger;
  Logger epoch_logger;
  std::vector<std::string> batch_metrics;
  std::vector<std::string> val_metrics;
  std::vector<std::string> epoch_metrics;

  CsvLogger(const std::string &model_name, const std::string &log_dir,
            const LogMode *log_mode = nullptr);

  void log_batch(int epoch, int step, const std::unordered_map<std::string, double> &metrics);

  void log_val_batch(int epoch, int step, const std::unordered_map<std::string, double> &metrics);

  void log_epoch(int epoch, const std::unordered_map<std::string, double> &metrics);

  void flush() {
    batch_logger.flush();
    val_logger.flush();
    epoch_logger.flush();
  }
};

struct WorkerCsvLogger {
  Logger compute_logger;

  WorkerCsvLogger(const std::string &worker_name, const std::string &log_dir)
      : compute_logger(worker_name + "_compute", "") {
    std::filesystem::create_directories(log_dir);
    std::string ts = csv_timestamp();

    std::string path = log_dir + "/" + worker_name + "_compute_" + ts + ".csv";
    compute_logger.set_log_file(path);
    compute_logger.set_pattern("%v");
    compute_logger.info("step,event_type,time_ms");
  }

  void log(int step, const std::string &event_type, long time_ms, size_t device_used_memory_mb) {
    std::ostringstream row;
    row << step << "," << event_type << "," << time_ms << "," << device_used_memory_mb;
    compute_logger.info(row.str());
  }

  void flush() { compute_logger.flush(); }
};

}  // namespace tnn
