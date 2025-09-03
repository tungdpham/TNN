/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "message.hpp"
#include "pipeline_endpoint.hpp"
#include "task.hpp"
#include "concurrent_message_map.hpp"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace tpipeline {

template <typename T = float> class PipelineCommunicator {
public:
  PipelineCommunicator() = default;

  virtual ~PipelineCommunicator() {
    std::cout << "Communicator getting destroyed" << std::endl;
    std::lock_guard<std::mutex> out_lock(out_message_mutex_);
    std::lock_guard<std::mutex> rec_lock(recipients_mutex_);

    // Clear all messages using the concurrent map
    message_queues_.clear();

    std::queue<Message<T>> empty_out;
    out_message_queue_.swap(empty_out);

    recipients_.clear();

    message_notification_callback_ = nullptr;
  }

  virtual void send_message(const Message<T> &message) = 0;

  virtual void flush_output_messages() = 0;

  virtual void register_recipient(const std::string &recipient_id,
                                  const StageEndpoint &endpoint) {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    recipients_[recipient_id] = endpoint;
  }

  virtual void unregister_recipient(const std::string &recipient_id) {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    recipients_.erase(recipient_id);
  }

  virtual StageEndpoint
  get_recipient(const std::string &recipient_id) const {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    auto it = recipients_.find(recipient_id);
    if (it == recipients_.end()) {
      throw std::runtime_error("Recipient not found: " + recipient_id);
    }
    return it->second;
  }

  inline void enqueue_input_message(const Message<T> &message) {
    if(message.sender_id.empty()){
      std::cerr << "WARNING: Unknown Sender ID" << std::endl;
      return;
    }
    
    message_queues_.push(message.command_type, message);

    if (message_notification_callback_) {
      message_notification_callback_();
    }
  }

  inline void enqueue_output_message(const Message<T> &message) {
    if(message.recipient_id.empty()) {
      std::cout << message.to_string() << std::endl;
      throw std::runtime_error("Message recipient_id is empty");
    }
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    this->out_message_queue_.push(message);
  }

  inline Message<T> dequeue_input_message() {
    Message<T> message;

    // Priority order: FORWARD_TASK, BACKWARD_TASK, UPDATE_PARAMETERS first
    if (message_queues_.pop(CommandType::FORWARD_TASK, message)) {
      return message;
    }

    if (message_queues_.pop(CommandType::BACKWARD_TASK, message)) {
      return message;
    }

    if (message_queues_.pop(CommandType::UPDATE_PARAMETERS, message)) {
      return message;
    }

    // Then try control messages (excluding status messages)
    std::vector<CommandType> control_types = {
      CommandType::START,
      CommandType::TRAIN_MODE,
      CommandType::EVAL_MODE,
      CommandType::SHUTDOWN,
      CommandType::HANDSHAKE_REQUEST,
      CommandType::HANDSHAKE_RESPONSE,
      CommandType::READY_SIGNAL,
      CommandType::CONFIG_RECEIVED,
      CommandType::PARAMETERS_UPDATED,
      CommandType::ERROR_REPORT,
      CommandType::TASK_FAILURE,
      CommandType::BARRIER_SYNC,
      CommandType::CHECKPOINT_REQUEST,
      CommandType::CHECKPOINT_COMPLETE,
      CommandType::CLEAR_PROFILING
    };

    for (CommandType cmd_type : control_types) {
      if (message_queues_.pop(cmd_type, message)) {
        return message;
      }
    }

    // Finally status messages
    std::vector<CommandType> status_types = {
      CommandType::STATUS_REQUEST,
      CommandType::STATUS_RESPONSE,
      CommandType::HEALTH_CHECK,
      CommandType::MEMORY_REPORT,
      CommandType::RESOURCE_REQUEST,
      CommandType::QUERY_STAGE_INFO,
      CommandType::STAGE_INFO_RESPONSE,
      CommandType::PRINT_PROFILING
    };

    for (CommandType cmd_type : status_types) {
      if (message_queues_.pop(cmd_type, message)) {
        return message;
      }
    }

    throw std::runtime_error("No input messages available");
  }

  inline Message<T>
  dequeue_message_by_type(CommandType target_type) {
    Message<T> message;
    if (!message_queues_.pop(target_type, message)) {
      throw std::runtime_error("No message of specified type available");
    }
    return message;
  }

  inline std::vector<Message<T>> dequeue_all_messages_by_type(CommandType target_type) {
    std::vector<Message<T>> messages = message_queues_.pop_all(target_type);
    if (messages.empty()) {
      throw std::runtime_error("No messages of specified type available");
    }
    return messages;
  }

  inline size_t forward_message_count() const {
    return message_count_by_type(CommandType::FORWARD_TASK);
  }

  inline size_t backward_message_count() const {
    return message_count_by_type(CommandType::BACKWARD_TASK);
  }

  inline size_t params_updated_count() const {
    return message_count_by_type(CommandType::PARAMETERS_UPDATED);
  }

  inline size_t message_count_by_type(CommandType target_type) const {
    return message_queues_.size(target_type);
  }

  inline size_t status_message_count() const {
    return message_count_by_type(CommandType::STATUS_RESPONSE);
  }

  inline size_t parameter_update_count() const {
    return message_count_by_type(CommandType::PARAMETERS_UPDATED);
  }

  inline bool has_input_message() const {
    return message_queues_.has_any_message();
  }

  inline bool has_task_message() const {
    return !message_queues_.empty(CommandType::FORWARD_TASK) ||
           !message_queues_.empty(CommandType::BACKWARD_TASK);
  }

  inline bool has_message_of_type(CommandType target_type) const {
    return !message_queues_.empty(target_type);
  }

  inline bool has_status_message() const {
    return has_message_of_type(CommandType::STATUS_RESPONSE);
  }

  inline bool has_parameter_update_message() const {
    bool has_update = has_message_of_type(CommandType::PARAMETERS_UPDATED);
    return has_update;
  }

  inline bool has_output_message() const {
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    return !this->out_message_queue_.empty();
  }

  inline void
  set_message_notification_callback(std::function<void()> callback) {
    message_notification_callback_ = callback;
  }

protected:
  ConcurrentMessageMap<T> message_queues_;

  std::queue<Message<T>> out_message_queue_;

  mutable std::mutex out_message_mutex_;
  mutable std::mutex recipients_mutex_;

  std::function<void()> message_notification_callback_;

  std::unordered_map<std::string, StageEndpoint> recipients_;
};
} // namespace tpipeline
