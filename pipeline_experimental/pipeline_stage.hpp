#pragma once

#include "../nn/sequential.hpp"
#include "pipeline_communicator.hpp" // Full definition needed here
#include "task.hpp"
#include "thread_pool.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace tpipeline {

template <typename T = float> class PipelineStage {
public:
  explicit PipelineStage(std::unique_ptr<tnn::Sequential<T>> model,
                         std::unique_ptr<PipelineCommunicator<T>> communicator,
                         const std::string &name = "")
      : model_(std::move(model)), communicator_(std::move(communicator)),
        name_(name), should_stop_(false), is_processing_(false),
        thread_pool_(2) {} // 2 threads: main event listener and task processor

  virtual ~PipelineStage() { stop(); }

  // Initialize and start the stage
  virtual void start() {
    should_stop_ = false;

    // Start the main event listener
    event_listener_future_ =
        thread_pool_.enqueue([this]() { event_listener(); });

    // Set up the communicator to notify this stage when tasks arrive
    communicator_->set_task_notification_callback(
        [this]() { notify_task_available(); });
  }

  // Stop the stage
  virtual void stop() {
    should_stop_ = true;

    // Notify waiting threads to wake up and exit
    task_available_cv_.notify_all();

    // Wait for event listener to complete
    if (event_listener_future_.valid())
      event_listener_future_.wait();
  }

  bool is_processing() const { return is_processing_; }

  // Get the model associated with this stage
  tnn::Sequential<T> *get_model() { return model_.get(); }

  // Get the name of the stage
  std::string name() const { return name_; }

  // Get the communicator (useful for coordinator to send tasks)
  PipelineCommunicator<T> *get_communicator() { return communicator_.get(); }

protected:
  // Event-driven listener that waits for task notifications
  void event_listener() {
    while (!should_stop_) {
      std::unique_lock<std::mutex> lock(task_available_mutex_);

      // Wait for a task to be available or stop signal
      task_available_cv_.wait(lock, [this]() {
        return should_stop_ ||
               (communicator_->has_input_task() && !is_processing_);
      });

      if (should_stop_)
        break;

      // If we have a task and not currently processing, spawn a thread to
      // handle it
      if (communicator_->has_input_task() && !is_processing_) {
        is_processing_ = true;

        // Spawn a task processing thread
        thread_pool_.enqueue([this]() {
          tpipeline::Task<T> task = communicator_->dequeue_input_task();
          process_task(task);
          is_processing_ = false;
          notify_task_available();
        });
      }
    }
  }

  // Called by communicator when a new task arrives
  void notify_task_available() {
    std::lock_guard<std::mutex> lock(task_available_mutex_);
    task_available_cv_.notify_one();
  }

  void process_task(const tpipeline::Task<T> &task) {
    // Forward or backward pass based on task type
    tpipeline::Task<T> output_task = task;

    if (task.type == tpipeline::TaskType::Forward) {
      // Forward pass
      output_task.data = this->model_->forward(task.data);
    } else if (task.type == tpipeline::TaskType::Backward) {
      // Backward pass
      output_task.data = this->model_->backward(task.data);
    }
    // Send the result immediately after processing
    this->communicator_->enqueue_output_task(output_task);
    this->communicator_->send_output_task();
  }

protected:
  std::unique_ptr<tnn::Sequential<T>> model_;
  std::unique_ptr<PipelineCommunicator<T>> communicator_;
  std::string name_;

  std::atomic<bool> should_stop_;
  std::atomic<bool> is_processing_;

  ThreadPool thread_pool_;
  std::future<void> event_listener_future_;

  // Event-based synchronization
  std::mutex task_available_mutex_;
  std::condition_variable task_available_cv_;
};

} // namespace tpipeline