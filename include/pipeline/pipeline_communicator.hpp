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
    std::lock_guard<std::mutex> in_lock(in_message_mutex_);
    std::lock_guard<std::mutex> out_lock(out_message_mutex_);
    std::lock_guard<std::mutex> rec_lock(recipients_mutex_);

    // Clear all command type queues
    for (auto& pair : message_queues_) {
      std::queue<Message<T>> empty_queue;
      pair.second.swap(empty_queue);
    }
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
    
    {
      std::lock_guard<std::mutex> lock(this->in_message_mutex_);
      message_queues_[message.command_type].push(message);
    }

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
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    if (!message_queues_[CommandType::FORWARD_TASK].empty()) {
      Message<T> message = message_queues_[CommandType::FORWARD_TASK].front();
      message_queues_[CommandType::FORWARD_TASK].pop();
      return message;
    }

    if (!message_queues_[CommandType::BACKWARD_TASK].empty()) {
      Message<T> message = message_queues_[CommandType::BACKWARD_TASK].front();
      message_queues_[CommandType::BACKWARD_TASK].pop();
      return message;
    }

    if (!message_queues_[CommandType::UPDATE_PARAMETERS].empty()) {
      Message<T> message = message_queues_[CommandType::UPDATE_PARAMETERS].front();
      message_queues_[CommandType::UPDATE_PARAMETERS].pop();
      return message;
    }

    // Then control messages
    for (auto& pair : message_queues_) {
      CommandType cmd_type = pair.first;
      if (cmd_type != CommandType::FORWARD_TASK && 
          cmd_type != CommandType::BACKWARD_TASK && 
          cmd_type != CommandType::UPDATE_PARAMETERS &&
          cmd_type != CommandType::STATUS_REQUEST &&
          cmd_type != CommandType::STATUS_RESPONSE &&
          cmd_type != CommandType::HEALTH_CHECK &&
          cmd_type != CommandType::MEMORY_REPORT &&
          cmd_type != CommandType::RESOURCE_REQUEST &&
          cmd_type != CommandType::QUERY_STAGE_INFO &&
          cmd_type != CommandType::STAGE_INFO_RESPONSE &&
          cmd_type != CommandType::PRINT_PROFILING) {
        if (!pair.second.empty()) {
          Message<T> message = pair.second.front();
          pair.second.pop();
          return message;
        }
      }
    }

    // Finally status messages
    for (auto& pair : message_queues_) {
      CommandType cmd_type = pair.first;
      if (cmd_type == CommandType::STATUS_REQUEST ||
          cmd_type == CommandType::STATUS_RESPONSE ||
          cmd_type == CommandType::HEALTH_CHECK ||
          cmd_type == CommandType::MEMORY_REPORT ||
          cmd_type == CommandType::RESOURCE_REQUEST ||
          cmd_type == CommandType::QUERY_STAGE_INFO ||
          cmd_type == CommandType::STAGE_INFO_RESPONSE ||
          cmd_type == CommandType::PRINT_PROFILING) {
        if (!pair.second.empty()) {
          Message<T> message = pair.second.front();
          pair.second.pop();
          return message;
        }
      }
    }

    throw std::runtime_error("No input messages available");
  }

  inline Message<T>
  dequeue_message_by_type(CommandType target_type) {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    if (message_queues_[target_type].empty()) {
      throw std::runtime_error("No message of specified type available");
    }

    Message<T> message = message_queues_[target_type].front();
    message_queues_[target_type].pop();
    return message;
  }

  inline std::vector<Message<T>> dequeue_all_messages_by_type(CommandType target_type) {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    std::vector<Message<T>> messages;
    auto& queue = message_queues_[target_type];
    
    while (!queue.empty()) {
      messages.push_back(queue.front());
      queue.pop();
    }

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
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    auto it = message_queues_.find(target_type);
    if (it != message_queues_.end()) {
      return it->second.size();
    }
    return 0;
  }

  inline size_t status_message_count() const {
    return message_count_by_type(CommandType::STATUS_RESPONSE);
  }

  inline size_t parameter_update_count() const {
    return message_count_by_type(CommandType::PARAMETERS_UPDATED);
  }

  inline bool has_input_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    for (const auto& pair : message_queues_) {
      if (!pair.second.empty()) {
        return true;
      }
    }
    return false;
  }

  inline bool has_task_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    auto forward_it = message_queues_.find(CommandType::FORWARD_TASK);
    auto backward_it = message_queues_.find(CommandType::BACKWARD_TASK);
    
    return (forward_it != message_queues_.end() && !forward_it->second.empty()) ||
           (backward_it != message_queues_.end() && !backward_it->second.empty());
  }

  inline bool has_message_of_type(CommandType target_type) const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    auto it = message_queues_.find(target_type);
    return it != message_queues_.end() && !it->second.empty();
  }

  inline bool has_status_message() const {
    return has_message_of_type(CommandType::STATUS_RESPONSE);
  }

  inline bool has_parameter_update_message() const {
    bool has_update = has_message_of_type(CommandType::PARAMETERS_UPDATED);
    return has_update;
  }

  inline std::vector<Message<T>> get_input_messages() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    std::vector<Message<T>> messages;

    for (const auto& pair : message_queues_) {
      auto temp_queue = pair.second;
      while (!temp_queue.empty()) {
        messages.push_back(temp_queue.front());
        temp_queue.pop();
      }
    }

    return messages;
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
  std::unordered_map<CommandType, std::queue<Message<T>>> message_queues_;

  std::queue<Message<T>> out_message_queue_;

  mutable std::mutex in_message_mutex_;
  mutable std::mutex out_message_mutex_;
  mutable std::mutex recipients_mutex_;

  std::function<void()> message_notification_callback_;

  std::unordered_map<std::string, StageEndpoint> recipients_;
};
} // namespace tpipeline
