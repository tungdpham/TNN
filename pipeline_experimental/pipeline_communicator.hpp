#pragma once

#include "pipeline_endpoint.hpp"
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

  // Send an output task to the stage in the correct direction
  virtual void send_output_task() = 0;

  // Receive an input task from corresponding method of communication (in memory
  // for in-process, websocket for distributed)
  virtual void receive_input_task() = 0;

  // Enqueue a task into the input queue (called by other stages)
  inline virtual void enqueue_input_task(const tpipeline::Task<T> &task) {
    {
      std::lock_guard<std::mutex> lock(this->in_task_mutex_);
      this->in_task_queue_.push(task);
    }
    // Notify stage that a task is available
    if (task_notification_callback_) {
      task_notification_callback_();
    }
  }

  inline void enqueue_output_task(const tpipeline::Task<T> &task) {
    std::lock_guard<std::mutex> lock(this->out_task_mutex_);
    this->out_task_queue_.push(task);
  }

  // Dequeue a task from the input queue
  inline tpipeline::Task<T> dequeue_input_task() {
    std::lock_guard<std::mutex> lock(this->in_task_mutex_);
    if (this->in_task_queue_.empty()) {
      throw std::runtime_error("No input tasks available");
    }
    tpipeline::Task<T> task = this->in_task_queue_.front();
    this->in_task_queue_.pop();
    return task;
  }

  inline size_t input_queue_size() const {
    std::lock_guard<std::mutex> lock(this->in_task_mutex_);
    return this->in_task_queue_.size();
  }

  // Check if the input queue is empty
  inline bool has_input_task() const {
    std::lock_guard<std::mutex> lock(this->in_task_mutex_);
    return !this->in_task_queue_.empty();
  }

  inline bool has_output_task() const {
    std::lock_guard<std::mutex> lock(this->out_task_mutex_);
    return !this->out_task_queue_.empty();
  }

  inline virtual void
  set_next_stage_endpoint(const tpipeline::StageEndpoint &endpoint) {
    next_stage_endpoint_ = endpoint;
  }

  inline virtual void
  set_prev_stage_endpoint(const tpipeline::StageEndpoint &endpoint) {
    prev_stage_endpoint_ = endpoint;
  }

  // Set callback for task notification (event-based)
  inline void set_task_notification_callback(std::function<void()> callback) {
    task_notification_callback_ = callback;
  }

protected:
  std::queue<tpipeline::Task<T>> in_task_queue_;
  std::queue<tpipeline::Task<T>> out_task_queue_;
  // mutex lock for input
  mutable std::mutex in_task_mutex_;
  // mutex lock for output
  mutable std::mutex out_task_mutex_;

  // Event-based notification callback
  std::function<void()> task_notification_callback_;

  tpipeline::StageEndpoint next_stage_endpoint_;
  tpipeline::StageEndpoint prev_stage_endpoint_;
};

template <typename T = float>
class InProcessPipelineCommunicator : public PipelineCommunicator<T> {
public:
  InProcessPipelineCommunicator() = default;
  ~InProcessPipelineCommunicator() override = default;

  void send_output_task() override;

  void receive_input_task() override {
    // Not applicable for in-process communication
  }
};

} // namespace tpipeline

#include "in_process_pipeline_communicator.tpp"