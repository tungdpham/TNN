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

template <typename T = float> class PipelineCommunicator {
public:
  PipelineCommunicator() = default;
  virtual ~PipelineCommunicator() = default;

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
    {
      std::lock_guard<std::mutex> lock(this->in_message_mutex_);
      this->in_message_queue_.push(message);
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

  // Dequeue a message from the input queue
  inline tpipeline::Message<T> dequeue_input_message() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    if (this->in_message_queue_.empty()) {
      throw std::runtime_error("No input messages available");
    }
    tpipeline::Message<T> message = this->in_message_queue_.front();
    this->in_message_queue_.pop();
    return message;
  }

  // Dequeue a specific type of message (task, status, parameter update, etc.)
  inline tpipeline::Message<T> dequeue_message_by_type(CommandType target_type) {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    // Search for the target message type
    std::queue<tpipeline::Message<T>> temp_queue;
    tpipeline::Message<T> target_message;
    bool found = false;
    
    while (!this->in_message_queue_.empty()) {
      auto message = this->in_message_queue_.front();
      this->in_message_queue_.pop();
      
      if (!found && message.command_type == target_type) {
        target_message = message;
        found = true;
      } else {
        temp_queue.push(message);
      }
    }
    
    // Put back all non-target messages
    while (!temp_queue.empty()) {
      this->in_message_queue_.push(temp_queue.front());
      temp_queue.pop();
    }
    
    if (!found) {
      throw std::runtime_error("No message of specified type available");
    }
    
    return target_message;
  }

  // Convenience method to dequeue any task message (FORWARD_TASK or BACKWARD_TASK)
  inline tpipeline::Message<T> dequeue_task_message() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    std::queue<tpipeline::Message<T>> temp_queue;
    tpipeline::Message<T> target_message;
    bool found = false;
    
    while (!this->in_message_queue_.empty()) {
      auto message = this->in_message_queue_.front();
      this->in_message_queue_.pop();
      
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
      this->in_message_queue_.push(temp_queue.front());
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
    return this->in_message_queue_.size();
  }

  // Count only task messages (FORWARD_TASK, BACKWARD_TASK) in input queue
  inline size_t task_queue_size() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    size_t count = 0;
    auto temp_queue = this->in_message_queue_;
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
    auto temp_queue = this->in_message_queue_;
    while (!temp_queue.empty()) {
      auto message = temp_queue.front();
      temp_queue.pop();
      if (message.command_type == target_type) {
        count++;
      }
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
    return !this->in_message_queue_.empty();
  }

  // Check if there are any task messages in the input queue
  inline bool has_task_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    auto temp_queue = this->in_message_queue_;
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
    auto temp_queue = this->in_message_queue_;
    while (!temp_queue.empty()) {
      auto message = temp_queue.front();
      temp_queue.pop();
      if (message.command_type == target_type) {
        return true;
      }
    }
    return false;
  }

  // Convenience methods for common message type checks
  inline bool has_status_message() const {
    return has_message_of_type(CommandType::STATUS_RESPONSE);
  }

  inline bool has_parameter_update_message() const {
    return has_message_of_type(CommandType::PARAMETERS_UPDATED);
  }

  inline bool has_control_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    auto temp_queue = this->in_message_queue_;
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
    // get a vector copy of all messages
    std::queue<tpipeline::Message<T>> temp_queue = this->in_message_queue_;
    // Ensure in message queue is not modified while copying
    int num_messages = this->in_message_queue_.size();
    std::vector<tpipeline::Message<T>> messages;
    while (!temp_queue.empty()) {
      messages.push_back(temp_queue.front());
      temp_queue.pop();
    }
    if(this->in_message_queue_.size() != num_messages) {
      throw std::runtime_error("Inconsistent message queue size during copy");
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

protected:
  struct OutgoingMessage {
    std::string recipient_id;
    tpipeline::Message<T> message;
  };

  std::queue<tpipeline::Message<T>> in_message_queue_;
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
