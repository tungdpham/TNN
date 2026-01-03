#include "logging/logger.hpp"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace tnn {

Logger::Logger(const std::string &name, const std::string &log_file, LogLevel level)
    : logger_name_(name) {
  if (log_file.empty()) {
    logger_ = spdlog::stdout_color_mt(name);
  } else {
    logger_ = spdlog::basic_logger_mt(name, log_file);
  }

  logger_->set_level(level);
  logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
}

void Logger::set_level(spdlog::level::level_enum level) {
  if (logger_) {
    logger_->set_level(level);
  }
}

void Logger::set_log_file(const std::string &log_file) {
  if (logger_) {
    spdlog::drop(logger_name_);
  }

  logger_ = spdlog::basic_logger_mt(logger_name_, log_file);
  logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
}

void Logger::enable_console_logging(bool enable) {
  if (enable && logger_) {
    spdlog::drop(logger_name_);

    logger_ = spdlog::stdout_color_mt(logger_name_);
    logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
  }
}

} // namespace tnn
