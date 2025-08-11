#pragma once

#include "pipeline_endpoint.hpp"
#include "message.hpp"
#include "task.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>

namespace tpipeline {

template <typename T = float> class PipelineCommunicator {
public:
  PipelineCommunicator() = default;
  virtual ~PipelineCommunicator() = default;

  // Send an output message to the appropriate stage
  virtual void send_output_message() = 0;

  // Receive an input message from corresponding method of communication 
  // (in memory for in-process, websocket for distributed)
  virtual void receive_input_message() = 0;

  // Enqueue a message into the input queue (called by other stages)
  inline virtual void enqueue_input_message(const tpipeline::Message<T> &message) {
    {
      std::lock_guard<std::mutex> lock(this->in_message_mutex_);
      this->in_message_queue_.push(message);
    }
    // Notify stage that a message is available
    if (message_notification_callback_) {
      message_notification_callback_();
    }
  }

  inline void enqueue_output_message(const tpipeline::Message<T> &message) {
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    this->out_message_queue_.push(message);
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

  inline size_t input_queue_size() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return this->in_message_queue_.size();
  }

  // Check if the input queue is empty
  inline bool has_input_message() const {
    std::lock_guard<std::mutex> lock(this->in_message_mutex_);
    return !this->in_message_queue_.empty();
  }

  inline bool has_output_message() const {
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);
    return !this->out_message_queue_.empty();
  }

  inline virtual void
  set_next_stage_endpoint(const tpipeline::StageEndpoint &endpoint) {
    next_stage_endpoint_ = endpoint;
  }

  inline virtual void
  set_prev_stage_endpoint(const tpipeline::StageEndpoint &endpoint) {
    prev_stage_endpoint_ = endpoint;
  }

  // Set callback for message notification (event-based)
  inline void set_message_notification_callback(std::function<void()> callback) {
    message_notification_callback_ = callback;
  }

protected:
  std::queue<tpipeline::Message<T>> in_message_queue_;
  std::queue<tpipeline::Message<T>> out_message_queue_;
  // mutex lock for input
  mutable std::mutex in_message_mutex_;
  // mutex lock for output
  mutable std::mutex out_message_mutex_;

  // Event-based notification callback
  std::function<void()> message_notification_callback_;

  tpipeline::StageEndpoint next_stage_endpoint_;
  tpipeline::StageEndpoint prev_stage_endpoint_;
};

template <typename T = float>
class InProcessPipelineCommunicator : public PipelineCommunicator<T> {
public:
  InProcessPipelineCommunicator() = default;
  ~InProcessPipelineCommunicator() override = default;

  void send_output_message() override;

  void receive_input_message() override {
    // Not applicable for in-process communication
  }
};

} // namespace tpipeline

#include "in_process_pipeline_communicator.tpp"