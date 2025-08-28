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
  TASK,    
  CONTROL, 
  STATUS   
};

template <typename T = float> class PipelineCommunicator {
public:
  PipelineCommunicator() = default;
  
  virtual ~PipelineCommunicator() {
    
    std::lock_guard<std::mutex> in_lock(in_message_mutex_);
    std::lock_guard<std::mutex> out_lock(out_message_mutex_);
    std::lock_guard<std::mutex> rec_lock(recipients_mutex_);

    std::queue<tpipeline::Message<T>> empty_task;
    std::queue<tpipeline::Message<T>> empty_control;
    std::queue<tpipeline::Message<T>> empty_status;
    std::queue<OutgoingMessage> empty_out;
    
    task_queue_.swap(empty_task);
    control_queue_.swap(empty_control);
    status_queue_.swap(empty_status);
    out_message_queue_.swap(empty_out);
    
    
    recipients_.clear();
    
    
    message_notification_callback_ = nullptr;
  }

  virtual void send_message(const std::string& recipient_id, const tpipeline::Message<T>& message) = 0;
  
  virtual void flush_output_messages() = 0;

  virtual void register_recipient(const std::string& recipient_id, const tpipeline::StageEndpoint& endpoint) {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    recipients_[recipient_id] = endpoint;
  }

  
  virtual void unregister_recipient(const std::string& recipient_id) {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    recipients_.erase(recipient_id);
  }

  
  virtual tpipeline::StageEndpoint get_recipient(const std::string& recipient_id) const {
    std::lock_guard<std::mutex> lock(recipients_mutex_);
    auto it = recipients_.find(recipient_id);
    if (it == recipients_.end()) {
      throw std::runtime_error("Recipient not found: " + recipient_id);
    }
    return it->second;
  }

  
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
    
    if (message_notification_callback_) {
      message_notification_callback_();
    }
  }

  
  inline void enqueue_output_message(const std::string& recipient_id, const tpipeline::Message<T>& message) {
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    this->out_message_queue_.push({recipient_id, message});
  }

  
  inline void enqueue_output_message(const tpipeline::Message<T>& message) {
    if (message.recipient_id.empty()) {
      throw std::runtime_error("Message recipient_id is empty");
    }
    enqueue_output_message(message.recipient_id, message);
  }

  
  inline tpipeline::Message<T> dequeue_input_message() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    
    if (!this->task_queue_.empty()) {
      tpipeline::Message<T> message = this->task_queue_.front();
      this->task_queue_.pop();
      return message;
    }
    
    if (!this->control_queue_.empty()) {
      tpipeline::Message<T> message = this->control_queue_.front();
      this->control_queue_.pop();
      return message;
    }
    
    if (!this->status_queue_.empty()) {
      tpipeline::Message<T> message = this->status_queue_.front();
      this->status_queue_.pop();
      return message;
    }
    
    throw std::runtime_error("No input messages available");
  }

  
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

  
  inline tpipeline::Message<T> dequeue_task_message() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    
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
    
    
    while (!temp_queue.empty()) {
      this->task_queue_.push(temp_queue.front());
      temp_queue.pop();
    }
    
    if (!found) {
      throw std::runtime_error("No task message available");
    }
    
    return target_message;
  }

  
  inline tpipeline::Message<T> dequeue_status_message() {
    return dequeue_message_by_type(CommandType::STATUS_RESPONSE);
  }

  
  inline tpipeline::Message<T> dequeue_parameter_update_message() {
    return dequeue_message_by_type(CommandType::PARAMETERS_UPDATED);
  }

  inline size_t input_queue_size() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return this->task_queue_.size() + 
           this->control_queue_.size() + 
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

  
  inline size_t status_message_count() const {
    return message_count_by_type(CommandType::STATUS_RESPONSE);
  }

  inline size_t parameter_update_count() const {
    return message_count_by_type(CommandType::PARAMETERS_UPDATED);
  }

  
  inline bool has_input_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return !this->task_queue_.empty() || 
           !this->control_queue_.empty() || 
           !this->status_queue_.empty();
  }

private:
  
  static MessagePriority get_message_priority(CommandType command_type) {
    switch(command_type) {
      
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

  inline std::vector<tpipeline::Message<T>> get_input_messages() {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    
    std::vector<tpipeline::Message<T>> messages;
    
    
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

  
  inline void send_to_next_stage(const tpipeline::Message<T>& message) {
    enqueue_output_message("next_stage", message);
  }

  inline void send_to_prev_stage(const tpipeline::Message<T>& message) {
    enqueue_output_message("prev_stage", message);
  }

  inline void send_to_coordinator(const tpipeline::Message<T>& message) {
    enqueue_output_message("coordinator", message);
  }

  
  inline void set_message_notification_callback(std::function<void()> callback) {
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
  struct OutgoingMessage {
    std::string recipient_id;
    tpipeline::Message<T> message;
  };

  
  std::queue<tpipeline::Message<T>> task_queue_;     
  std::queue<tpipeline::Message<T>> control_queue_;  
  std::queue<tpipeline::Message<T>> status_queue_;   
  
  std::queue<OutgoingMessage> out_message_queue_;
  
  
  mutable std::mutex in_message_mutex_;
  mutable std::mutex out_message_mutex_;
  mutable std::mutex recipients_mutex_;

  
  std::function<void()> message_notification_callback_;

  
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
        
        
        lock.~lock_guard();
        send_message(outgoing.recipient_id, outgoing.message);
      }
    }
  }
  
  void register_communicator(const std::string& recipient_id, 
                            std::shared_ptr<PipelineCommunicator<T>> communicator) {
    std::lock_guard<std::mutex> lock(communicators_mutex_);
    communicators_[recipient_id] = communicator;
  }

private:
  std::unordered_map<std::string, std::shared_ptr<PipelineCommunicator<T>>> communicators_;
  mutable std::mutex communicators_mutex_;
};

} 
