#pragma once

#include "pipeline_endpoint.hpp"
#include "message.hpp"
#include "task.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <string>

namespace tpipeline {

enum class MessagePriority {
  TASK,    // FORWARD_TASK, BACKWARD_TASK, UPDATE_PARAMETERS
  CONTROL, // Control commands like START_TRAINING, STOP_TRAINING, etc.
  STATUS   // Status requests, stage info, misc messages
};

template <typename T = float> class PipelineCommunicator {
public:
  PipelineCommunicator() = default;
  
  virtual ~PipelineCommunicator() {
    // Clear all message queues
    std::lock_guard<std::mutex> in_lock(in_message_mutex_);
    std::lock_guard<std::mutex> out_lock(out_message_mutex_);
    std::lock_guard<std::mutex> rec_lock(recipients_mutex_);
    
    // Clear queues
    std::queue<tpipeline::Message<T>> empty_task;
    std::queue<tpipeline::Message<T>> empty_control;
    std::queue<tpipeline::Message<T>> empty_status;
    std::queue<OutgoingMessage> empty_out;
    
    task_queue_.swap(empty_task);
    control_queue_.swap(empty_control);
    status_queue_.swap(empty_status);
    out_message_queue_.swap(empty_out);
    
    // Clear maps
    recipients_.clear();
    
    // Reset callback
    message_notification_callback_ = nullptr;
  }

  // Send a message to a specific recipient
  virtual void send_message(const std::string& recipient_id, const tpipeline::Message<T>& message) = 0;
  
  // Send all queued output messages
  virtual void flush_output_messages() = 0;

  // Receive messages from the communication layer (for distributed implementations)
  virtual void receive_messages() = 0;

  // Register a recipient endpoint
  virtual void register_recipient(const std::string& recipient_id, const tpipeline::StageEndpoint& endpoint) {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    recipients_[recipient_id] = endpoint;
  }

  // Remove a recipient
  virtual void unregister_recipient(const std::string& recipient_id) {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    recipients_.erase(recipient_id);
  }

  // Get recipient endpoint
  virtual tpipeline::StageEndpoint get_recipient(const std::string& recipient_id) const {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    auto it = recipients_.find(recipient_id);
    if (it == recipients_.end()) {
      throw std::runtime_error("Recipient not found: " + recipient_id);
    }
    return it->second;
  }

  // Enqueue a message into the input queue (called by other stages/coordinator)
  inline virtual void enqueue_input_message(const tpipeline::Message<T>& message) {
    MessagePriority priority = get_message_priority(message.command_type);
    
    {
      std::lock_guard<std::mutex> lock(this->in_message_mutex_);
      switch(priority) {
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
    // Notify stage that a message is available
    if (message_notification_callback_) {
      message_notification_callback_();
    }
  }

  // Enqueue a message for sending (to be sent by send_message or flush)
  inline void enqueue_output_message(const std::string& recipient_id, const tpipeline::Message<T>& message) {
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    this->out_message_queue_.push({recipient_id, message});
  }

  // Simplified version that uses message's recipient_id
  inline void enqueue_output_message(const tpipeline::Message<T>& message) {
    if (message.recipient_id.empty()) {
      throw std::runtime_error("Message recipient_id is empty");
    }
    enqueue_output_message(message.recipient_id, message);
  }

  // Dequeue a message from the input queue (priority order: TASK -> CONTROL -> STATUS)
  inline tpipeline::Message<T> dequeue_input_message() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    // Try task queue first
    if (!this->task_queue_.empty()) {
      tpipeline::Message<T> message = this->task_queue_.front();
      this->task_queue_.pop();
      return message;
    }
    
    // Then control queue
    if (!this->control_queue_.empty()) {
      tpipeline::Message<T> message = this->control_queue_.front();
      this->control_queue_.pop();
      return message;
    }
    
    // Finally status queue
    if (!this->status_queue_.empty()) {
      tpipeline::Message<T> message = this->status_queue_.front();
      this->status_queue_.pop();
      return message;
    }
    
    throw std::runtime_error("No input messages available");
  }

  // Dequeue a specific type of message (task, status, parameter update, etc.)
  inline tpipeline::Message<T> dequeue_message_by_type(CommandType target_type) {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    MessagePriority target_priority = get_message_priority(target_type);
    
    auto search_and_dequeue = [target_type](std::queue<tpipeline::Message<T>>& queue) -> std::optional<tpipeline::Message<T>> {
      std::queue<tpipeline::Message<T>> temp_queue;
      tpipeline::Message<T> target_message;
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
      
      // Put back all non-target messages
      while (!temp_queue.empty()) {
        queue.push(temp_queue.front());
        temp_queue.pop();
      }
      
      if (found) {
        return target_message;
      }
      return std::nullopt;
    };
    
    std::optional<tpipeline::Message<T>> result;
    switch(target_priority) {
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

  // Convenience method to dequeue any task message (FORWARD_TASK or BACKWARD_TASK)
  inline tpipeline::Message<T> dequeue_task_message() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    // Task messages should only be in task queue
    std::queue<tpipeline::Message<T>> temp_queue;
    tpipeline::Message<T> target_message;
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
    
    // Put back all non-task messages
    while (!temp_queue.empty()) {
      this->task_queue_.push(temp_queue.front());
      temp_queue.pop();
    }
    
    if (!found) {
      throw std::runtime_error("No task message available");
    }
    
    return target_message;
  }

  // Convenience method to dequeue status messages
  inline tpipeline::Message<T> dequeue_status_message() {
    return dequeue_message_by_type(CommandType::STATUS_RESPONSE);
  }

  // Convenience method to dequeue parameter update messages  
  inline tpipeline::Message<T> dequeue_parameter_update_message() {
    return dequeue_message_by_type(CommandType::PARAMETERS_UPDATED);
  }

  inline size_t input_queue_size() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return this->task_queue_.size() + 
           this->control_queue_.size() + 
           this->status_queue_.size();
  }

  // Count only task messages (FORWARD_TASK, BACKWARD_TASK) in input queue
  inline size_t actual_task_message_count() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    size_t count = 0;
    
    // Task messages should only be in task queue
    auto temp_queue = this->task_queue_;
    while (!temp_queue.empty()) {
      auto message = temp_queue.front();
      temp_queue.pop();
      if (message.command_type == CommandType::FORWARD_TASK || 
          message.command_type == CommandType::BACKWARD_TASK) {
        count++;
      }
    }
    return count;
  }

  // Count messages of a specific type in input queue
  inline size_t message_count_by_type(CommandType target_type) const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    size_t count = 0;
    
    // Check all three queues based on the target type's priority
    MessagePriority target_priority = get_message_priority(target_type);
    
    auto check_queue = [&count, target_type](const std::queue<tpipeline::Message<T>>& queue) {
      auto temp_queue = queue;
      while (!temp_queue.empty()) {
        auto message = temp_queue.front();
        temp_queue.pop();
        if (message.command_type == target_type) {
          count++;
        }
      }
    };
    
    switch(target_priority) {
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

  // Convenience methods for common message type counts
  inline size_t status_message_count() const {
    return message_count_by_type(CommandType::STATUS_RESPONSE);
  }

  inline size_t parameter_update_count() const {
    return message_count_by_type(CommandType::PARAMETERS_UPDATED);
  }

  // Check if the input queue is empty
  inline bool has_input_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return !this->task_queue_.empty() || 
           !this->control_queue_.empty() || 
           !this->status_queue_.empty();
  }

private:
  // Helper function to determine message priority
  static MessagePriority get_message_priority(CommandType command_type) {
    switch(command_type) {
      // Task priority: Core pipeline tasks
      case CommandType::FORWARD_TASK:
      case CommandType::BACKWARD_TASK:
      case CommandType::UPDATE_PARAMETERS:
        return MessagePriority::TASK;
        
      // Control priority: Control commands
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
        
      // Status priority: Status, info, misc messages
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

  // Check if there are any task messages in the input queue
  inline bool has_task_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    // Check task queue (where all task messages should be)
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

  // Check if there are messages of a specific type in the input queue
  inline bool has_message_of_type(CommandType target_type) const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    MessagePriority target_priority = get_message_priority(target_type);

    auto check_queue = [target_type](const std::queue<tpipeline::Message<T>>& queue) {
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
    
    switch(target_priority) {
      case MessagePriority::TASK:
        return check_queue(this->task_queue_);
      case MessagePriority::CONTROL:
        return check_queue(this->control_queue_);
      case MessagePriority::STATUS:
        return check_queue(this->status_queue_);
    }
    
    return false;
  }

  // Convenience methods for common message type checks
  inline bool has_status_message() const {
    return has_message_of_type(CommandType::STATUS_RESPONSE);
  }

  inline bool has_parameter_update_message() const {
    bool has_update = has_message_of_type(CommandType::PARAMETERS_UPDATED);
    return has_update;
  }

  inline bool has_control_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    // Control messages should be in control queue
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

  inline std::vector<tpipeline::Message<T>> get_input_messages() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    std::vector<tpipeline::Message<T>> messages;
    
    // Add task messages first
    auto task_temp = this->task_queue_;
    while (!task_temp.empty()) {
      messages.push_back(task_temp.front());
      task_temp.pop();
    }
    
    // Add control messages
    auto control_temp = this->control_queue_;
    while (!control_temp.empty()) {
      messages.push_back(control_temp.front());
      control_temp.pop();
    }
    
    // Add status messages
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

  // Convenience methods for common communication patterns
  inline void send_to_next_stage(const tpipeline::Message<T>& message) {
    enqueue_output_message("next_stage", message);
  }

  inline void send_to_prev_stage(const tpipeline::Message<T>& message) {
    enqueue_output_message("prev_stage", message);
  }

  inline void send_to_coordinator(const tpipeline::Message<T>& message) {
    enqueue_output_message("coordinator", message);
  }

  // Set callback for message notification (event-based)
  inline void set_message_notification_callback(std::function<void()> callback) {
    message_notification_callback_ = callback;
  }

  // Additional convenience methods for priority-based queuing
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
  struct OutgoingMessage {
    std::string recipient_id;
    tpipeline::Message<T> message;
  };

  // Three priority queues for input messages
  std::queue<tpipeline::Message<T>> task_queue_;     // FORWARD_TASK, BACKWARD_TASK, UPDATE_PARAMETERS
  std::queue<tpipeline::Message<T>> control_queue_;  // Control commands
  std::queue<tpipeline::Message<T>> status_queue_;   // Status, info, misc messages
  
  std::queue<OutgoingMessage> out_message_queue_;
  
  // Mutex locks
  mutable std::mutex in_message_mutex_;
  mutable std::mutex out_message_mutex_;
  mutable std::mutex recipients_mutex_;

  // Event-based notification callback
  std::function<void()> message_notification_callback_;

  // Map of recipient IDs to their endpoints
  std::unordered_map<std::string, tpipeline::StageEndpoint> recipients_;
};

template <typename T = float>
class InProcessPipelineCommunicator : public PipelineCommunicator<T> {
public:
  InProcessPipelineCommunicator() = default;
  ~InProcessPipelineCommunicator() override = default;

  void send_message(const std::string& recipient_id, const tpipeline::Message<T>& message) override {
    std::lock_guard<std::mutex> lock(communicators_mutex_);
    auto it = communicators_.find(recipient_id);
    if (it != communicators_.end() && it->second) {
      it->second->enqueue_input_message(message);
    } else {
      throw std::runtime_error("Recipient communicator not found: " + recipient_id);
    }
  }
  
  void flush_output_messages() override {
    while (this->has_output_message()) {
      std::lock_guard<std::mutex> lock(this->out_message_mutex_);
      if (!this->out_message_queue_.empty()) {
        auto outgoing = this->out_message_queue_.front();
        this->out_message_queue_.pop();
        
        // Release lock before sending to avoid deadlock
        lock.~lock_guard();
        send_message(outgoing.recipient_id, outgoing.message);
      }
    }
  }

  void receive_messages() override {
    // Not applicable for in-process communication
  }

  // Set reference to other communicators for in-process communication
  void register_communicator(const std::string& recipient_id, 
                            std::shared_ptr<PipelineCommunicator<T>> communicator) {
    std::lock_guard<std::mutex> lock(communicators_mutex_);
    communicators_[recipient_id] = communicator;
  }

private:
  std::unordered_map<std::string, std::shared_ptr<PipelineCommunicator<T>>> communicators_;
  mutable std::mutex communicators_mutex_;
};

} // namespace tpipeline
