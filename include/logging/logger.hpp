#pragma once

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace tnn {

typedef spdlog::level::level_enum LogLevel;

class Logger {
public:
  Logger(const std::string &name = "default_logger", const std::string &log_file = "",
         LogLevel level = LogLevel::info);

  ~Logger() = default;

  void set_level(LogLevel level);

  void set_log_file(const std::string &log_file);

  void enable_console_logging(bool enable = true);

  template <typename... Args>
  void trace(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    if (logger_) logger_->trace(fmt, std::forward<Args>(args)...);
  }

  void trace(std::string_view msg) {
    if (logger_) logger_->trace(msg);
  }

  template <typename... Args>
  void debug(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    if (logger_) logger_->debug(fmt, std::forward<Args>(args)...);
  }

  void debug(std::string_view msg) {
    if (logger_) logger_->debug(msg);
  }

  template <typename... Args>
  void info(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    if (logger_) logger_->info(fmt, std::forward<Args>(args)...);
  }

  void info(std::string_view msg) {
    if (logger_) logger_->info(msg);
  }

  template <typename... Args>
  void warn(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    if (logger_) logger_->warn(fmt, std::forward<Args>(args)...);
  }

  void warn(std::string_view msg) {
    if (logger_) logger_->warn(msg);
  }

  template <typename... Args>
  void error(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    if (logger_) logger_->error(fmt, std::forward<Args>(args)...);
  }

  void error(std::string_view msg) {
    if (logger_) logger_->error(msg);
  }

  template <typename... Args>
  void critical(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    if (logger_) logger_->critical(fmt, std::forward<Args>(args)...);
  }

  void critical(std::string_view msg) {
    if (logger_) logger_->critical(msg);
  }

  void log_runtime(LogLevel level, std::string_view msg) {
    if (logger_) logger_->log(level, msg);
  }

private:
  std::shared_ptr<spdlog::logger> logger_;
  std::string logger_name_;
};

class GlobalLogger {
private:
  inline static Logger logger_{"global_logger", "", spdlog::level::info};

public:
  static void set_level(LogLevel level) { logger_.set_level(level); }

  static void set_log_file(const std::string &log_file) { logger_.set_log_file(log_file); }

  static void enable_console_logging(bool enable = true) { logger_.enable_console_logging(enable); }

  template <typename... Args>
  static void trace(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    logger_.trace(fmt, std::forward<Args>(args)...);
  }

  static void trace(std::string_view msg) { logger_.trace(msg); }

  template <typename... Args>
  static void debug(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    logger_.debug(fmt, std::forward<Args>(args)...);
  }

  static void debug(std::string_view msg) { logger_.debug(msg); }

  template <typename... Args>
  static void info(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    logger_.info(fmt, std::forward<Args>(args)...);
  }

  static void info(std::string_view msg) { logger_.info(msg); }

  template <typename... Args>
  static void warn(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    logger_.warn(fmt, std::forward<Args>(args)...);
  }

  static void warn(std::string_view msg) { logger_.warn(msg); }

  template <typename... Args>
  static void error(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    logger_.error(fmt, std::forward<Args>(args)...);
  }

  static void error(std::string_view msg) { logger_.error(msg); }

  template <typename... Args>
  static void critical(spdlog::format_string_t<Args...> fmt, Args &&...args) {
    logger_.critical(fmt, std::forward<Args>(args)...);
  }

  static void critical(std::string_view msg) { logger_.critical(msg); }

  static void log_runtime(LogLevel level, std::string_view msg) { logger_.log_runtime(level, msg); }
};

}  // namespace tnn