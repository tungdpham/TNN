#pragma once

#include "task.hpp"
#include <queue>
#include <mutex>
#include <stdexcept>
#include <memory>

template <typename T = float> class PipelineCommunicator {
public:
  PipelineCommunicator() = default;
  virtual ~PipelineCommunicator() = default;

  // Send an output task to the stage in the correct direction
  virtual void send_output_task() = 0;

  // Receive an input task from corresponding method of communication (in memory for in-process, websocket for distributed)
  virtual void receive_input_task() = 0;

  // Enqueue a task into the input queue (called by other stages)
  virtual void enqueue_task(const tpipeline::Task<T> &task) {
    std::lock_guard<std::mutex> lock(this->in_task_mutex_);
    this->in_task_queue_.push(task);
  }

  void enqueue_output_task(const tpipeline::Task<T> &task) {
    std::lock_guard<std::mutex> lock(this->out_task_mutex_);
    this->out_task_queue_.push(task);
  }

  // Dequeue a task from the input queue
  tpipeline::Task<T> dequeue_input_task() {
    std::lock_guard<std::mutex> lock(this->in_task_mutex_);
    if (this->in_task_queue_.empty()) {
      throw std::runtime_error("No input tasks available");
    }
    tpipeline::Task<T> task = this->in_task_queue_.front();
    this->in_task_queue_.pop();
    return task;
  }

  // Check if the input queue is empty
  bool has_input_task() const {
    std::lock_guard<std::mutex> lock(this->in_task_mutex_);
    return !this->in_task_queue_.empty();
  }

  virtual void set_next_stage(PipelineCommunicator<T> *next_stage) = 0;

  virtual void set_prev_stage(PipelineCommunicator<T> *prev_stage) = 0;

protected:
  std::queue<tpipeline::Task<T>> in_task_queue_;
  std::queue<tpipeline::Task<T>> out_task_queue_;
  //mutex lock for input
  mutable std::mutex in_task_mutex_;
  //mutex lock for output
  mutable std::mutex out_task_mutex_;
};


template <typename T = float> class InProcessPipelineCommunicator : public PipelineCommunicator<T> {
public:
  InProcessPipelineCommunicator(InProcessPipelineCommunicator<T> *prev_stage_comm,
                                InProcessPipelineCommunicator<T> *next_stage_comm)
      : prev_stage_comm_(prev_stage_comm), next_stage_comm_(next_stage_comm) {}

  ~InProcessPipelineCommunicator() override = default;  

  void send_output_task() override {
    std::lock_guard<std::mutex> lock(this->out_task_mutex_);
    if (!this->out_task_queue_.empty()) {
      tpipeline::Task<T> task = this->out_task_queue_.front();
      this->out_task_queue_.pop();
      if(task.type == tpipeline::TaskType::Forward) {
        next_stage_comm_->enqueue_task(task);
      } else if(task.type == tpipeline::TaskType::Backward) {
        prev_stage_comm_->enqueue_task(task);
      }
    }
  }

  void receive_input_task() override {
    // Not applicable for in-process communication - tasks are directly enqueued
  } 

  void set_next_stage(PipelineCommunicator<T> *next_stage) override {
    next_stage_comm_ = static_cast<InProcessPipelineCommunicator<T> *>(next_stage);
  }

  void set_prev_stage(PipelineCommunicator<T> *prev_stage) override {
    prev_stage_comm_ = static_cast<InProcessPipelineCommunicator<T> *>(prev_stage);
  }

private:
  InProcessPipelineCommunicator<T> *prev_stage_comm_ = nullptr; // Pointer to the previous stage communicator
  InProcessPipelineCommunicator<T> *next_stage_comm_ = nullptr; // Pointer to the next stage communicator
};