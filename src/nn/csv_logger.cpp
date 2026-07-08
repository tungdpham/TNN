/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/csv_logger.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>

namespace tunx {

CsvLogger::CsvLogger(const std::string &model_name, const std::string &log_dir,
                     const LogMode *log_mode)
    : train_step_logger_(model_name + "_batch", ""),
      val_step_logger_(model_name + "_val", ""),
      epoch_logger_(model_name + "_epoch", "") {
  std::filesystem::create_directories(log_dir);
  std::string ts = csv_timestamp();

  std::string batch_path = log_dir + "/" + model_name + "_batch_" + ts + ".csv";
  std::string val_path = log_dir + "/" + model_name + "_val_" + ts + ".csv";
  std::string epoch_path = log_dir + "/" + model_name + "_epoch_" + ts + ".csv";

  train_step_logger_.set_log_file(batch_path);
  train_step_logger_.set_pattern("%v");

  val_step_logger_.set_log_file(val_path);
  val_step_logger_.set_pattern("%v");

  epoch_logger_.set_log_file(epoch_path);
  epoch_logger_.set_pattern("%v");

  train_step_metrics_ = {"epoch",    "step",           "timestamp_ms", "loss",       "batch_loss",
                         "avg_loss", "avg_perplexity", "accuracy_pct", "perplexity", "time_ms"};

  val_step_metrics_ = {"epoch", "step", "timestamp_ms", "loss", "accuracy_pct", "perplexity"};

  epoch_metrics_ = {"epoch",
                    "timestamp_ms",
                    "train_loss",
                    "val_loss",
                    "train_accuracy_pct",
                    "val_accuracy_pct",
                    "train_perplexity",
                    "val_perplexity",
                    "train_time_ms",
                    "val_time_ms",
                    "epoch_total_time_ms"};

  // Write headers
  std::ostringstream train_step_header, val_step_header, epoch_header;
  for (size_t i = 0; i < train_step_metrics_.size(); ++i) {
    if (i > 0) train_step_header << ",";
    train_step_header << train_step_metrics_[i];
  }
  for (size_t i = 0; i < val_step_metrics_.size(); ++i) {
    if (i > 0) val_step_header << ",";
    val_step_header << val_step_metrics_[i];
  }
  for (size_t i = 0; i < epoch_metrics_.size(); ++i) {
    if (i > 0) epoch_header << ",";
    epoch_header << epoch_metrics_[i];
  }

  train_step_logger_.info(train_step_header.str());
  val_step_logger_.info(val_step_header.str());
  epoch_logger_.info(epoch_header.str());

  train_step_logger_.flush();
  val_step_logger_.flush();
  epoch_logger_.flush();
}

void CsvLogger::log_train_step(int epoch, int step,
                               const std::unordered_map<std::string, double> &metrics) {
  std::ostringstream row;
  row << epoch << "," << step << "," << get_timestamp_ms();

  for (size_t i = 3; i < train_step_metrics_.size(); ++i) {
    row << ",";
    auto it = metrics.find(train_step_metrics_[i]);
    if (it != metrics.end()) {
      row << std::fixed << std::setprecision(6) << it->second;
    } else {
      row << "";
    }
  }

  train_step_logger_.info(row.str());
  train_step_logger_.flush();
}

void CsvLogger::log_val_step(int epoch, int step,
                             const std::unordered_map<std::string, double> &metrics) {
  std::ostringstream row;
  row << epoch << "," << step << "," << get_timestamp_ms();

  for (size_t i = 3; i < val_step_metrics_.size(); ++i) {
    row << ",";
    auto it = metrics.find(val_step_metrics_[i]);
    if (it != metrics.end()) {
      row << std::fixed << std::setprecision(6) << it->second;
    } else {
      row << "";
    }
  }

  val_step_logger_.info(row.str());
  val_step_logger_.flush();
}

void CsvLogger::log_epoch(int epoch, const std::unordered_map<std::string, double> &metrics) {
  std::ostringstream row;
  row << epoch << "," << get_timestamp_ms();

  for (size_t i = 2; i < epoch_metrics_.size(); ++i) {
    row << ",";
    auto it = metrics.find(epoch_metrics_[i]);
    if (it != metrics.end()) {
      row << std::fixed << std::setprecision(6) << it->second;
    } else {
      row << "";
    }
  }

  epoch_logger_.info(row.str());
  epoch_logger_.flush();
}

}  // namespace tunx
