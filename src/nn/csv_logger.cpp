/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/csv_logger.hpp"

#include <sstream>
#include <iostream>
#include <cstdlib>

#include "nn/train.hpp"

namespace tnn {

CsvLogger::CsvLogger(const std::string &model_name, const std::string &log_dir,
                     const LogMode *log_mode)
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

  val_logger.set_log_file(val_path);
  val_logger.set_pattern("%v");

  epoch_logger.set_log_file(epoch_path);
  epoch_logger.set_pattern("%v");

  // Build deterministic metric order for stable CSV logs.
  // Dynamic ordering caused column/value mismatches during incremental builds.

  batch_metrics = {
      "epoch",
      "step",
      "loss",
      "batch_loss",
      "avg_loss",
      "avg_perplexity",
      "accuracy_pct",
      "perplexity",
      "time_ms"
  };

  val_metrics = {
      "epoch",
      "step",
      "loss",
      "accuracy_pct",
      "perplexity"
  };

  epoch_metrics = {
      "epoch",
      "train_loss",
      "val_loss",
      "train_accuracy_pct",
      "val_accuracy_pct",
      "train_perplexity",
      "val_perplexity",
      "train_time_ms",
      "val_time_ms",
      "epoch_total_time_ms"
  };

  // Epoch-level wall-clock timing. These are always present because they are
  // essential for paper/benchmark tables even when only loss/accuracy logging
  // is enabled.
  epoch_metrics.push_back("train_time_ms");
  epoch_metrics.push_back("val_time_ms");
  epoch_metrics.push_back("epoch_total_time_ms");

  // Write headers
  std::ostringstream batch_header, val_header, epoch_header;
  for (size_t i = 0; i < batch_metrics.size(); ++i) {
    if (i > 0) batch_header << ",";
    batch_header << batch_metrics[i];
  }
  for (size_t i = 0; i < val_metrics.size(); ++i) {
    if (i > 0) val_header << ",";
    val_header << val_metrics[i];
  }
  for (size_t i = 0; i < epoch_metrics.size(); ++i) {
    if (i > 0) epoch_header << ",";
    epoch_header << epoch_metrics[i];
  }

  if (const char *dbg = std::getenv("TNN_DEBUG_CSV")) {
    if (std::string(dbg) != "0") {
      std::cout << "[CSVDBG][constructor_batch_header] " << batch_header.str() << std::endl;
    }
  }

  batch_logger.info(batch_header.str());
  val_logger.info(val_header.str());
  epoch_logger.info(epoch_header.str());

  batch_logger.flush();
  val_logger.flush();
  epoch_logger.flush();
}

void CsvLogger::log_batch(int epoch, int step,
                          const std::unordered_map<std::string, double> &metrics) {
  std::ostringstream row;
  row << epoch << "," << step;

  if (const char *dbg = std::getenv("TNN_DEBUG_CSV")) {
    if (std::string(dbg) != "0" && step <= 5) {
      std::cout << "[CSVDBG][batch_header] step=" << step;
      for (const auto &h : batch_metrics) {
        std::cout << " [" << h << "]";
      }
      std::cout << std::endl;

      std::cout << "[CSVDBG][metric_keys] step=" << step;
      for (const auto &kv : metrics) {
        std::cout << " [" << kv.first << "=" << kv.second << "]";
      }
      std::cout << std::endl;
    }
  }

  for (size_t i = 2; i < batch_metrics.size(); ++i) {
    row << ",";
    auto it = metrics.find(batch_metrics[i]);
    if (it != metrics.end()) {
      row << std::fixed << std::setprecision(6) << it->second;
    } else {
      row << "";
    }
  }

  batch_logger.info(row.str());
  batch_logger.flush();
}

void CsvLogger::log_val_batch(int epoch, int step,
                              const std::unordered_map<std::string, double> &metrics) {
  std::ostringstream row;
  row << epoch << "," << step;

  for (size_t i = 2; i < val_metrics.size(); ++i) {
    row << ",";
    auto it = metrics.find(val_metrics[i]);
    if (it != metrics.end()) {
      row << std::fixed << std::setprecision(6) << it->second;
    } else {
      row << "";
    }
  }

  val_logger.info(row.str());
  val_logger.flush();
}

void CsvLogger::log_epoch(int epoch, const std::unordered_map<std::string, double> &metrics) {
  std::ostringstream row;
  row << epoch;

  for (size_t i = 1; i < epoch_metrics.size(); ++i) {
    row << ",";
    auto it = metrics.find(epoch_metrics[i]);
    if (it != metrics.end()) {
      row << std::fixed << std::setprecision(6) << it->second;
    } else {
      row << "";
    }
  }

  epoch_logger.info(row.str());
  epoch_logger.flush();
}

}  // namespace tnn
