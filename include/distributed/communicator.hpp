/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "device/iallocator.hpp"
#include "endpoint.hpp"
#include "message.hpp"
#include "message_map.hpp"
#include "utils/misc.hpp"

namespace tnn {
/**
 * @brief Abstract base class for pipeline communication
 * Defines the interface for sending and receiving messages
 * between different stages in a distributed pipeline.
 */
class Communicator {
private:
  std::vector<CommandType> all_command_types_ = get_enum_vector<CommandType>();

public:
  Communicator(Endpoint endpoint)
      : endpoint_(endpoint) {}

  virtual ~Communicator() {
    std::lock_guard<std::mutex> out_lock(out_message_mutex_);

    message_queues_.clear();

    std::queue<std::pair<Message, Endpoint>> empty_out;
    out_message_queue_.swap(empty_out);

    message_notification_callback_ = nullptr;
  }

  Endpoint endpoint() const { return endpoint_; }

  void send_message(Message &&message, const Endpoint &endpoint) {
    if (endpoint.type() == CommunicationType::IN_PROCESS) {
      auto other_communicator = endpoint.get_parameter<Communicator *>("communicator");
      other_communicator->enqueue_input_message(std::move(message));
    } else {
      this->send_impl(std::move(message), endpoint);
    }
  }

  virtual void flush_output_messages() = 0;

  bool connect(const Endpoint &endpoint) {
    try {
      if (endpoint.type() == CommunicationType::IN_PROCESS) {
        return true;
      }
      if (!connect_to_endpoint(endpoint)) {
        return false;
      }
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Failed to connect to endpoint: " << e.what() << std::endl;
      return false;
    }
  }

  bool disconnect(const Endpoint &endpoint) {
    try {
      if (endpoint.type() == CommunicationType::IN_PROCESS) {
        return true;
      }
      if (!disconnect_from_endpoint(endpoint)) {
        return false;
      }
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Failed to disconnect from endpoint: " << e.what() << std::endl;
      return false;
    }
  }

  inline void enqueue_input_message(Message &&message) {
    message_queues_.push(message.header().command_type, std::move(message));

    if (message_notification_callback_) {
      message_notification_callback_();
    }
  }

  inline void enqueue_output_message(Message &&message, const Endpoint &endpoint) {
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    this->out_message_queue_.push(std::make_pair(std::move(message), endpoint));
  }

  inline Message dequeue_input_message() {
    Message message;

    for (const auto &cmd_type : all_command_types_) {
      if (message_queues_.pop(cmd_type, message)) {
        return message;
      }
    }

    return Message();
  }

  inline Message dequeue_input_message(CommandType target_type) {
    Message message;
    if (!message_queues_.pop(target_type, message)) {
      std::cerr << "No messages of type " << static_cast<int>(target_type) << " available"
                << std::endl;
    }
    return message;
  }

  inline std::vector<Message> dequeue_all_messages_by_type(CommandType target_type) {
    std::vector<Message> messages = message_queues_.pop_all(target_type);
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

  inline void set_callback(std::function<void()> callback) {
    message_notification_callback_ = callback;
  }

  size_t num_input_messages() const { return message_queues_.total_size(); }

  size_t num_output_messages() const {
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    return this->out_message_queue_.size();
  }

  std::unordered_map<std::string, int64_t> get_profile_data() {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    return profile_data_;
  }

  void clear_profile_data() {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    profile_data_.clear();
  }

  virtual IAllocator &out_allocator() = 0;

protected:
  virtual void send_impl(Message &&message, const Endpoint &endpoint) = 0;

  virtual bool connect_to_endpoint(const Endpoint &endpoint) = 0;

  virtual bool disconnect_from_endpoint(const Endpoint &endpoint) = 0;

  void add_profile_data(const std::string &key, int64_t value) {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    auto it = profile_data_.find(key);
    if (it == profile_data_.end()) {
      profile_data_[key] = value;
    } else {
      it->second += value;
    }
  }

protected:
  Endpoint endpoint_;
  MessageMap message_queues_;
  std::queue<std::pair<Message, Endpoint>> out_message_queue_;
  mutable std::mutex out_message_mutex_;
  std::function<void()> message_notification_callback_;

private:
  std::mutex profile_mutex_;
  std::unordered_map<std::string, int64_t> profile_data_;  // profiling data in microseconds
};
}  // namespace tnn
