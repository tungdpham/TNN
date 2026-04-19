/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#include "nn/csv_logger.hpp"

#include <sstream>

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

  // Build headers based on log_mode
  batch_metrics = {"epoch", "step"};
  val_metrics = {"epoch", "step"};
  epoch_metrics = {"epoch"};

  if (log_mode) {
    if (log_mode->log_loss) {
      batch_metrics.push_back("loss");
      val_metrics.push_back("loss");
      epoch_metrics.push_back("train_loss");
      epoch_metrics.push_back("val_loss");
    }
    if (log_mode->log_accuracy) {
      batch_metrics.push_back("accuracy_pct");
      val_metrics.push_back("accuracy_pct");
      epoch_metrics.push_back("train_accuracy_pct");
      epoch_metrics.push_back("val_accuracy_pct");
    }
    if (log_mode->log_precision) {
      batch_metrics.push_back("precision");
      val_metrics.push_back("precision");
      epoch_metrics.push_back("train_precision");
      epoch_metrics.push_back("val_precision");
    }
    if (log_mode->log_recall) {
      batch_metrics.push_back("recall");
      val_metrics.push_back("recall");
      epoch_metrics.push_back("train_recall");
      epoch_metrics.push_back("val_recall");
    }
    if (log_mode->log_f1_score) {
      batch_metrics.push_back("f1_score");
      val_metrics.push_back("f1_score");
      epoch_metrics.push_back("train_f1_score");
      epoch_metrics.push_back("val_f1_score");
    }
    if (log_mode->log_perplexity) {
      batch_metrics.push_back("perplexity");
      val_metrics.push_back("perplexity");
      epoch_metrics.push_back("train_perplexity");
      epoch_metrics.push_back("val_perplexity");
    }
    if (log_mode->log_top_k_accuracy) {
      batch_metrics.push_back("top_k_accuracy");
      val_metrics.push_back("top_k_accuracy");
      epoch_metrics.push_back("train_top_k_accuracy");
      epoch_metrics.push_back("val_top_k_accuracy");
    }
    if (log_mode->log_mae) {
      batch_metrics.push_back("mae");
      val_metrics.push_back("mae");
      epoch_metrics.push_back("train_mae");
      epoch_metrics.push_back("val_mae");
    }
    if (log_mode->log_mse) {
      batch_metrics.push_back("mse");
      val_metrics.push_back("mse");
      epoch_metrics.push_back("train_mse");
      epoch_metrics.push_back("val_mse");
    }
    if (log_mode->log_rmse) {
      batch_metrics.push_back("rmse");
      val_metrics.push_back("rmse");
      epoch_metrics.push_back("train_rmse");
      epoch_metrics.push_back("val_rmse");
    }
  } else {
    // Default: log loss and accuracy
    batch_metrics.push_back("loss");
    batch_metrics.push_back("accuracy_pct");
    val_metrics.push_back("loss");
    val_metrics.push_back("accuracy_pct");
    epoch_metrics.push_back("train_loss");
    epoch_metrics.push_back("train_accuracy_pct");
    epoch_metrics.push_back("val_loss");
    epoch_metrics.push_back("val_accuracy_pct");
  }

  batch_metrics.push_back("time_ms");

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
