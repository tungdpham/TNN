#pragma once

#include "command_type.hpp"
#include "task.hpp"
#include <chrono>
#include <variant>
#include <string>

namespace tpipeline {

template <typename T = float> struct Message {
  CommandType command_type;

  using PayloadType = std::variant<std::monostate, Task<T>, std::string, bool>;
  PayloadType payload;

  std::string sender_id;
  std::string recipient_id;

  std::chrono::steady_clock::time_point timestamp;
  int sequence_number = 0;

  Message() : payload(std::monostate{}) {}

  Message(CommandType cmd_type)
      : command_type(cmd_type), payload(std::monostate{}), 
        timestamp(std::chrono::steady_clock::now()) {}

  Message(CommandType cmd_type, const Task<T> &task_data)
      : command_type(cmd_type), payload(task_data),
        timestamp(std::chrono::steady_clock::now()) {}

  Message(CommandType cmd_type, const std::string &text)
      : command_type(cmd_type), payload(text),
        timestamp(std::chrono::steady_clock::now()) {}

  Message(CommandType cmd_type, bool signal_value)
      : command_type(cmd_type), payload(signal_value),
        timestamp(std::chrono::steady_clock::now()) {}

  // Helper methods using std::holds_alternative and std::get
  bool has_task() const { return std::holds_alternative<Task<T>>(payload); }
  bool has_text() const { return std::holds_alternative<std::string>(payload); }
  bool has_signal() const { return std::holds_alternative<bool>(payload); }
  
  const Task<T>& get_task() const { return std::get<Task<T>>(payload); }
  const std::string& get_text() const { return std::get<std::string>(payload); }
  bool get_signal() const { return std::get<bool>(payload); }

  bool is_task_message() const {
    return command_type == CommandType::FORWARD_TASK ||
           command_type == CommandType::BACKWARD_TASK;
  }

  bool is_control_message() const {
    return command_type == CommandType::START_TRAINING ||
           command_type == CommandType::STOP_TRAINING ||
           command_type == CommandType::PAUSE_TRAINING ||
           command_type == CommandType::RESUME_TRAINING;
  }

  static Message<T> forward_task(const Task<T> &task,
                                 const std::string &sender = "",
                                 const std::string &recipient = "") {
    Message<T> msg(CommandType::FORWARD_TASK, task);
    msg.sender_id = sender;
    msg.recipient_id = recipient;
    return msg;
  }

  static Message<T> backward_task(const Task<T> &task,
                                  const std::string &sender = "",
                                  const std::string &recipient = "") {
    Message<T> msg(CommandType::BACKWARD_TASK, task);
    msg.sender_id = sender;
    msg.recipient_id = recipient;
    return msg;
  }

  static Message<T> create_control_message(CommandType cmd_type,
                                           const std::string &sender = "",
                                           const std::string &recipient = "") {
    Message<T> msg(cmd_type);
    msg.sender_id = sender;
    msg.recipient_id = recipient;
    return msg;
  }

  static Message<T> status_message(const std::string &status_text,
                                   const std::string &sender = "",
                                   const std::string &recipient = "") {
    Message<T> msg(CommandType::STATUS_RESPONSE, status_text);
    msg.sender_id = sender;
    msg.recipient_id = recipient;
    return msg;
  }

  static Message<T> parameters_updated(const std::string &sender = "",
                                       const std::string &recipient = "") {
    Message<T> msg(CommandType::PARAMETERS_UPDATED);
    msg.sender_id = sender;
    msg.recipient_id = recipient;
    return msg;
  }

  static Message<T> ready_signal(const std::string &sender = "",
                                 const std::string &recipient = "") {
    Message<T> msg(CommandType::READY_SIGNAL, true);
    msg.sender_id = sender;
    msg.recipient_id = recipient;
    return msg;
  }

  static Message<T> error_message(const std::string &error_text,
                                  const std::string &sender = "",
                                  const std::string &recipient = "") {
    Message<T> msg(CommandType::ERROR_REPORT, error_text);
    msg.sender_id = sender;
    msg.recipient_id = recipient;
    return msg;
  }

  static Message<T> create_text_message(CommandType cmd_type,
                                        const std::string &text,
                                        const std::string &sender = "",
                                        const std::string &recipient = "") {
    Message<T> msg(cmd_type, text);
    msg.sender_id = sender;
    msg.recipient_id = recipient;
    return msg;
  }

  static Message<T> create_signal_message(CommandType cmd_type,
                                          bool signal_value,
                                          const std::string &sender = "",
                                          const std::string &recipient = "") {
    Message<T> msg(cmd_type, signal_value);
    msg.sender_id = sender;
    msg.recipient_id = recipient;
    return msg;
  }

  std::string to_string() const {
    std::string result =
        "Message(" + std::to_string(static_cast<int>(command_type)) +
        ", sender: " + sender_id + ", recipient: " + recipient_id;
    if (has_task()) {
      result += ", task: " + get_task().to_string();
    }
    if (has_text()) {
      result += ", text: " + get_text();
    }
    if (has_signal()) {
      result += ", signal: " + std::to_string(get_signal());
    }
    result += ")";
    return result;
  }
};

} // namespace tpipeline