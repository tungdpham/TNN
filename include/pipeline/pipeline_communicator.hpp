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

enum class MessagePriority { TASK, CONTROL, STATUS };

template <typename T = float> class PipelineCommunicator {
public:
  PipelineCommunicator() = default;

  virtual ~PipelineCommunicator() {
    std::cout << "Communicator getting destroyed" << std::endl;
    std::lock_guard<std::mutex> in_lock(in_message_mutex_);
    std::lock_guard<std::mutex> out_lock(out_message_mutex_);
    std::lock_guard<std::mutex> rec_lock(recipients_mutex_);

    std::queue<Message<T>> empty_task;
    std::queue<Message<T>> empty_control;
    std::queue<Message<T>> empty_status;
    std::queue<Message<T>> empty_out;

    task_queue_.swap(empty_task);
    control_queue_.swap(empty_control);
    status_queue_.swap(empty_status);
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
    MessagePriority priority = get_message_priority(message.command_type);
    {
      std::lock_guard<std::mutex> lock(this->in_message_mutex_);
      switch (priority) {
      case MessagePriority::TASK:
        this->task_queue_.push(message);
        break;
      case MessagePriority::CONTROL:
        this->control_queue_.push(message);
        break;
      case MessagePriority::STATUS:
        this->status_queue_.push(message);
        break;
      }
    }

    if (message_notification_callback_) {
      message_notification_callback_();
    }
  }

  inline void enqueue_output_message(const Message<T> &message) {
    if(message.recipient_id.empty()) {
      throw std::runtime_error("Message recipient_id is empty");
    }
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    this->out_message_queue_.push(message);
  }

  inline Message<T> dequeue_input_message() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    if (!this->task_queue_.empty()) {
      Message<T> message = this->task_queue_.front();
      this->task_queue_.pop();
      return message;
    }

    if (!this->control_queue_.empty()) {
      Message<T> message = this->control_queue_.front();
      this->control_queue_.pop();
      return message;
    }

    if (!this->status_queue_.empty()) {
      Message<T> message = this->status_queue_.front();
      this->status_queue_.pop();
      return message;
    }

    throw std::runtime_error("No input messages available");
  }

  inline Message<T>
  dequeue_message_by_type(CommandType target_type) {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    MessagePriority target_priority = get_message_priority(target_type);

    auto search_and_dequeue =
        [target_type](std::queue<Message<T>> &queue)
        -> std::optional<Message<T>> {
      std::queue<Message<T>> temp_queue;
      Message<T> target_message;
      bool found = false;

      while (!queue.empty()) {
        auto message = queue.front();
        queue.pop();

        if (!found && message.command_type == target_type) {
          target_message = message;
          found = true;
        } else {
          temp_queue.push(message);
        }
      }

      while (!temp_queue.empty()) {
        queue.push(temp_queue.front());
        temp_queue.pop();
      }

      if (found) {
        return target_message;
      }
      return std::nullopt;
    };

    std::optional<Message<T>> result;
    switch (target_priority) {
    case MessagePriority::TASK:
      result = search_and_dequeue(this->task_queue_);
      break;
    case MessagePriority::CONTROL:
      result = search_and_dequeue(this->control_queue_);
      break;
    case MessagePriority::STATUS:
      result = search_and_dequeue(this->status_queue_);
      break;
    }

    if (!result) {
      throw std::runtime_error("No message of specified type available");
    }

    return result.value();
  }

  inline Message<T> dequeue_task_message() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    std::queue<Message<T>> temp_queue;
    Message<T> target_message;
    bool found = false;

    while (!this->task_queue_.empty()) {
      auto message = this->task_queue_.front();
      this->task_queue_.pop();

      if (!found && (message.command_type == CommandType::FORWARD_TASK ||
                     message.command_type == CommandType::BACKWARD_TASK)) {
        target_message = message;
        found = true;
      } else {
        temp_queue.push(message);
      }
    }

    while (!temp_queue.empty()) {
      this->task_queue_.push(temp_queue.front());
      temp_queue.pop();
    }

    if (!found) {
      throw std::runtime_error("No task message available");
    }

    return target_message;
  }

  inline Message<T> dequeue_status_message() {
    return dequeue_message_by_type(CommandType::STATUS_RESPONSE);
  }

  inline Message<T> dequeue_parameter_update_message() {
    return dequeue_message_by_type(CommandType::PARAMETERS_UPDATED);
  }

  inline size_t input_queue_size() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return this->task_queue_.size() + this->control_queue_.size() +
           this->status_queue_.size();
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
    size_t count = 0;

    MessagePriority target_priority = get_message_priority(target_type);

    auto check_queue =
        [&count, target_type](const std::queue<Message<T>> &queue) {
          auto temp_queue = queue;
          while (!temp_queue.empty()) {
            auto message = temp_queue.front();
            temp_queue.pop();
            if (message.command_type == target_type) {
              count++;
            }
          }
        };

    switch (target_priority) {
    case MessagePriority::TASK:
      check_queue(this->task_queue_);
      break;
    case MessagePriority::CONTROL:
      check_queue(this->control_queue_);
      break;
    case MessagePriority::STATUS:
      check_queue(this->status_queue_);
      break;
    }

    return count;
  }

  inline size_t status_message_count() const {
    return message_count_by_type(CommandType::STATUS_RESPONSE);
  }

  inline size_t parameter_update_count() const {
    return message_count_by_type(CommandType::PARAMETERS_UPDATED);
  }

  inline bool has_input_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return !this->task_queue_.empty() || !this->control_queue_.empty() ||
           !this->status_queue_.empty();
  }

private:
  static MessagePriority get_message_priority(CommandType command_type) {
    switch (command_type) {

    case CommandType::FORWARD_TASK:
    case CommandType::BACKWARD_TASK:
    case CommandType::UPDATE_PARAMETERS:
      return MessagePriority::TASK;

    case CommandType::START_TRAINING:
    case CommandType::STOP_TRAINING:
    case CommandType::PAUSE_TRAINING:
    case CommandType::RESUME_TRAINING:
    case CommandType::HANDSHAKE_REQUEST:
    case CommandType::HANDSHAKE_RESPONSE:
    case CommandType::READY_SIGNAL:
    case CommandType::CONFIG_RECEIVED:
    case CommandType::PARAMETERS_UPDATED:
    case CommandType::ERROR_REPORT:
    case CommandType::TASK_FAILURE:
    case CommandType::BARRIER_SYNC:
    case CommandType::CHECKPOINT_REQUEST:
    case CommandType::CHECKPOINT_COMPLETE:
      return MessagePriority::CONTROL;

    case CommandType::STATUS_REQUEST:
    case CommandType::STATUS_RESPONSE:
    case CommandType::HEALTH_CHECK:
    case CommandType::MEMORY_REPORT:
    case CommandType::RESOURCE_REQUEST:
    case CommandType::QUERY_STAGE_INFO:
    case CommandType::STAGE_INFO_RESPONSE:
    case CommandType::PRINT_PROFILING:
    default:
      return MessagePriority::STATUS;
    }
  }

public:
  inline bool has_task_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    auto temp_queue = this->task_queue_;
    while (!temp_queue.empty()) {
      auto message = temp_queue.front();
      temp_queue.pop();
      if (message.command_type == CommandType::FORWARD_TASK ||
          message.command_type == CommandType::BACKWARD_TASK) {
        return true;
      }
    }
    return false;
  }

  inline bool has_message_of_type(CommandType target_type) const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    MessagePriority target_priority = get_message_priority(target_type);

    auto check_queue =
        [target_type](const std::queue<Message<T>> &queue) {
          auto temp_queue = queue;
          while (!temp_queue.empty()) {
            auto message = temp_queue.front();
            temp_queue.pop();
            if (message.command_type == target_type) {
              return true;
            }
          }
          return false;
        };

    switch (target_priority) {
    case MessagePriority::TASK:
      return check_queue(this->task_queue_);
    case MessagePriority::CONTROL:
      return check_queue(this->control_queue_);
    case MessagePriority::STATUS:
      return check_queue(this->status_queue_);
    }

    return false;
  }

  inline bool has_status_message() const {
    return has_message_of_type(CommandType::STATUS_RESPONSE);
  }

  inline bool has_parameter_update_message() const {
    bool has_update = has_message_of_type(CommandType::PARAMETERS_UPDATED);
    return has_update;
  }

  inline bool has_control_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    auto temp_queue = this->control_queue_;
    while (!temp_queue.empty()) {
      auto message = temp_queue.front();
      temp_queue.pop();
      if (message.command_type == CommandType::START_TRAINING ||
          message.command_type == CommandType::STOP_TRAINING ||
          message.command_type == CommandType::PAUSE_TRAINING ||
          message.command_type == CommandType::RESUME_TRAINING) {
        return true;
      }
    }
    return false;
  }

  inline std::vector<Message<T>> get_input_messages() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);

    std::vector<Message<T>> messages;

    auto task_temp = this->task_queue_;
    while (!task_temp.empty()) {
      messages.push_back(task_temp.front());
      task_temp.pop();
    }

    auto control_temp = this->control_queue_;
    while (!control_temp.empty()) {
      messages.push_back(control_temp.front());
      control_temp.pop();
    }

    auto status_temp = this->status_queue_;
    while (!status_temp.empty()) {
      messages.push_back(status_temp.front());
      status_temp.pop();
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

  inline size_t task_queue_size() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return this->task_queue_.size();
  }

  inline size_t control_queue_size() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return this->control_queue_.size();
  }

  inline size_t status_queue_size() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return this->status_queue_.size();
  }

  inline bool has_task_queue_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return !this->task_queue_.empty();
  }

  inline bool has_control_queue_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return !this->control_queue_.empty();
  }

  inline bool has_status_queue_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return !this->status_queue_.empty();
  }

protected:
  std::queue<Message<T>> task_queue_;
  std::queue<Message<T>> control_queue_;
  std::queue<Message<T>> status_queue_;

  std::queue<Message<T>> out_message_queue_;

  mutable std::mutex in_message_mutex_;
  mutable std::mutex out_message_mutex_;
  mutable std::mutex recipients_mutex_;

  std::function<void()> message_notification_callback_;

  std::unordered_map<std::string, StageEndpoint> recipients_;
};

template <typename T>
class InProcessPipelineCommunicator : public PipelineCommunicator<T> {
public:
  InProcessPipelineCommunicator() { start_delivery_thread(); }

  ~InProcessPipelineCommunicator() {
    if (delivery_thread_.joinable()) {
      delivery_thread_.join();
    }
  }

  // This is the producer side.
  void send_message(const Message<T> &message) override {
    if(message.recipient_id.empty()) {
      throw std::runtime_error("Message recipient_id is empty");
    }
    {
      std::lock_guard<std::mutex> lock(outgoing_queue_mutex_);
      outgoing_queue_.push(message);
    }
    outgoing_cv_.notify_one(); // Wake up the consumer thread
  }

  void start_delivery_thread() {
    delivery_thread_ = std::thread([this]() {
      while (true) { // might as well add a flag
        std::unique_lock<std::mutex> lock(outgoing_queue_mutex_);
        outgoing_cv_.wait(lock, [this]() { return !outgoing_queue_.empty(); });

        Message<T> outgoing = outgoing_queue_.front();
        outgoing_queue_.pop();
        lock.unlock();

        std::lock_guard<std::mutex> comm_lock(communicators_mutex_);
        auto it = communicators_.find(outgoing.recipient_id);
        if (it != communicators_.end() && it->second) {
          it->second->enqueue_input_message(outgoing);
        }
      }
    });
  }

  void flush_output_messages() override {
    while (this->has_output_message()) {
      Message<T> message = this->out_message_queue_.front();
      this->out_message_queue_.pop();

      send_message(message);
    }
  }

  void
  register_communicator(const std::string &recipient_id,
                        std::shared_ptr<PipelineCommunicator<T>> communicator) {
    std::lock_guard<std::mutex> lock(communicators_mutex_);
    communicators_[recipient_id] = communicator;
  }

private:
  std::queue<Message<T>> outgoing_queue_;
  std::mutex outgoing_queue_mutex_;
  std::condition_variable outgoing_cv_;
  std::thread delivery_thread_;
  std::unordered_map<std::string, std::shared_ptr<PipelineCommunicator<T>>>
      communicators_;
  mutable std::mutex communicators_mutex_;
};

} // namespace tpipeline
