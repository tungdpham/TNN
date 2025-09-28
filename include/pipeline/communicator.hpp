/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "concurrent_message_map.hpp"
#include "message.hpp"
#include "pipeline_endpoint.hpp"
#include "task.hpp"
#include "utils/misc.hpp"
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
#include <vector>

namespace tpipeline {
/**
 * @brief Abstract base class for pipeline communication
 * Defines the interface for sending and receiving messages
 * between different stages in a distributed pipeline.
 */
template <typename T = float> class PipelineCommunicator {
private:
  std::vector<CommandType> all_command_types_ = utils::get_enum_vector<CommandType>();

public:
  PipelineCommunicator() = default;

  virtual ~PipelineCommunicator() {
    std::lock_guard<std::mutex> out_lock(out_message_mutex_);
    std::lock_guard<std::mutex> rec_lock(recipients_mutex_);

    message_queues_.clear();

    std::queue<Message<T>> empty_out;
    out_message_queue_.swap(empty_out);

    recipients_.clear();

    message_notification_callback_ = nullptr;
  }

  virtual void send_message(const Message<T> &message) = 0;

  virtual void flush_output_messages() = 0;

  virtual void register_recipient(const std::string &recipient_id, const StageEndpoint &endpoint) {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    recipients_[recipient_id] = endpoint;
  }

  virtual void unregister_recipient(const std::string &recipient_id) {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    recipients_.erase(recipient_id);
  }

  virtual StageEndpoint get_recipient(const std::string &recipient_id) const {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    auto it = recipients_.find(recipient_id);
    if (it == recipients_.end()) {
      throw std::runtime_error("Recipient not found: " + recipient_id);
    }
    return it->second;
  }

  inline void enqueue_input_message(const Message<T> &message) {
    if (message.sender_id.empty()) {
      std::cerr << "WARNING: Unknown Sender ID" << std::endl;
      return;
    }

    message_queues_.push(message.command_type, message);

    if (message_notification_callback_) {
      message_notification_callback_();
    }
  }

  inline void enqueue_output_message(const Message<T> &message) {
    if (message.recipient_id.empty()) {
      std::cout << message.to_string() << std::endl;
      throw std::runtime_error("Message recipient_id is empty");
    }
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    this->out_message_queue_.push(message);
  }

  inline Message<T> dequeue_input_message() {
    Message<T> message;

    for (const auto &cmd_type : all_command_types_) {
      if (message_queues_.pop(cmd_type, message)) {
        return message;
      }
    }

    return Message<T>();
  }

  inline Message<T> dequeue_input_message(CommandType target_type) {
    Message<T> message;
    if (!message_queues_.pop(target_type, message)) {
      std::cerr << "No messages of type " << static_cast<int>(target_type) << " available"
                << std::endl;
    }
    return message;
  }

  inline std::vector<Message<T>> dequeue_all_messages_by_type(CommandType target_type) {
    std::vector<Message<T>> messages = message_queues_.pop_all(target_type);
    return messages;
  }

  inline size_t empty() const { return message_queues_.empty(); }

  inline size_t message_count(CommandType target_type) const {
    return message_queues_.size(target_type);
  }

  inline bool has_input_message() const { return !message_queues_.empty(); }

  inline bool has_output_message() const {
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    return !this->out_message_queue_.empty();
  }

  inline void set_message_notification_callback(std::function<void()> callback) {
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
