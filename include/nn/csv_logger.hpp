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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "logging/logger.hpp"

namespace tnn {

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

  CsvLogger(const std::string &model_name, const std::string &log_dir)
      : batch_logger(model_name + "_batch", ""),
        val_logger(model_name + "_val", ""),
        epoch_logger(model_name + "_epoch", "") {
    std::filesystem::create_directories(log_dir);
    std::string ts = csv_timestamp();

    std::string batch_path = log_dir + "/" + model_name + "_batch_" + ts + ".csv";
    std::string val_path = log_dir + "/" + model_name + "_val_" + ts + ".csv";
    std::string epoch_path = log_dir + "/" + model_name + "_epoch_" + ts + ".csv";

    batch_logger.set_log_file(batch_path);
    batch_logger.set_pattern("%v");
    batch_logger.info("epoch,step,loss,accuracy_pct,time_ms");
    batch_logger.flush();

    val_logger.set_log_file(val_path);
    val_logger.set_pattern("%v");
    val_logger.info("epoch,step,loss,accuracy_pct");
    val_logger.flush();

    epoch_logger.set_log_file(epoch_path);
    epoch_logger.set_pattern("%v");
    epoch_logger.info("epoch,train_loss,train_accuracy_pct,val_loss,val_accuracy_pct");
    epoch_logger.flush();
  }

  void log_batch(int epoch, int step, float loss, double accuracy_pct, long time_ms) {
    std::ostringstream row;
    row << epoch << "," << step << "," << std::fixed << std::setprecision(6) << loss << ","
        << std::setprecision(4) << accuracy_pct << "," << time_ms;
    batch_logger.info(row.str());
    batch_logger.flush();
  }

  void log_val_batch(int epoch, int step, float loss, double accuracy_pct) {
    std::ostringstream row;
    row << epoch << "," << step << "," << std::fixed << std::setprecision(6) << loss << ","
        << std::setprecision(4) << accuracy_pct;
    val_logger.info(row.str());
    val_logger.flush();
  }

  void log_epoch(int epoch, double train_loss, double train_acc_pct, double val_loss,
                 double val_acc_pct) {
    std::ostringstream row;
    row << epoch << "," << std::fixed << std::setprecision(6) << train_loss << ","
        << std::setprecision(4) << train_acc_pct << "," << std::setprecision(6) << val_loss << ","
        << std::setprecision(4) << val_acc_pct;
    epoch_logger.info(row.str());
    epoch_logger.flush();
  }

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
