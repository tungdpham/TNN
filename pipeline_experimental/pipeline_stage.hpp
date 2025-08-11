#pragma once

#include "../nn/sequential.hpp"
#include "pipeline_communicator.hpp" // Full definition needed here
#include "task.hpp"
#include "thread_pool.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
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

    // Set up the communicator to notify this stage when messages arrive
    communicator_->set_message_notification_callback(
        [this]() { notify_message_available(); });
  }

  // Stop the stage
  virtual void stop() {
    should_stop_ = true;

    // Notify waiting threads to wake up and exit
    message_available_cv_.notify_all();

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
  // Event-driven listener that waits for message notifications
  void event_listener() {
    while (!should_stop_) {
      std::unique_lock<std::mutex> lock(message_available_mutex_);

      // Wait for a message to be available or stop signal
      message_available_cv_.wait(lock, [this]() {
        return should_stop_ ||
               (communicator_->has_input_message() && !is_processing_);
      });

      if (should_stop_)
        break;

      // If we have a message and not currently processing, spawn a thread to handle it
      if (communicator_->has_input_message() && !is_processing_) {
        is_processing_ = true;

        // Spawn a message processing thread
        thread_pool_.enqueue([this]() {
          tpipeline::Message<T> message = communicator_->dequeue_input_message();
          process_message(message);
          is_processing_ = false;
          notify_message_available();
        });
      }
    }
  }

  // Called by communicator when a new message arrives
  void notify_message_available() {
    std::lock_guard<std::mutex> lock(message_available_mutex_);
    message_available_cv_.notify_one();
  }

  void process_message(const tpipeline::Message<T> &message) {
    switch (message.command_type) {
      case CommandType::FORWARD_TASK:
      case CommandType::BACKWARD_TASK: {
        process_task_message(message);
        break;
      }
      case CommandType::START_TRAINING: {
        // Handle start training command
        // This could set internal state, initialize resources, etc.
        break;
      }
      case CommandType::STOP_TRAINING: {
        // Handle stop training command
        should_stop_ = true;
        break;
      }
      case CommandType::STATUS_REQUEST: {
        // Handle status request - send back current status
        auto status = tpipeline::StatusInfo(is_processing_, 
                                communicator_->input_queue_size(), 
                                name_ + " operational");
        Message<T> response(CommandType::STATUS_RESPONSE, status, name_, message.sender_id);
        communicator_->enqueue_output_message(response);
        communicator_->send_output_message();
        break;
      }
      case CommandType::ERROR_REPORT: {
        // Handle error report from other stages
        if (message.template has_payload<tpipeline::ErrorInfo>()) {
          auto error_info = message.template get_payload<tpipeline::ErrorInfo>();
          // Log error, potentially stop processing, etc.
          printf("Stage %s received error: %s from %s\n", 
                 name_.c_str(), error_info.error_message.c_str(), error_info.stage_name.c_str());
        }
        break;
      }
      default: {
        // Handle unknown command types
        printf("Stage %s received unknown command type\n", name_.c_str());
        break;
      }
    }
  }

  void process_task_message(const tpipeline::Message<T> &message) {
    if (!message.template has_payload<Task<T>>()) {
      // Send error message
      auto error_info = tpipeline::ErrorInfo("Expected task payload but got different type", name_);
      Message<T> error_msg(CommandType::ERROR_REPORT, error_info, name_, "coordinator");
      communicator_->enqueue_output_message(error_msg);
      communicator_->send_output_message();
      return;
    }

    auto task = message.template get_payload<Task<T>>();
    
    // Create output message with same metadata
    Message<T> output_message = message; // Copy metadata
    
    if (message.command_type == CommandType::FORWARD_TASK) {
      // Forward pass
      Task<T> output_task = task;
      output_task.data = this->model_->forward(task.data);
      output_message.payload = output_task;
    } else if (message.command_type == CommandType::BACKWARD_TASK) {
      // Backward pass
      Task<T> output_task = task;
      output_task.data = this->model_->backward(task.data);
      output_message.payload = output_task;
    }
    
    // Send the result immediately after processing
    this->communicator_->enqueue_output_message(output_message);
    this->communicator_->send_output_message();
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
  std::mutex message_available_mutex_;
  std::condition_variable message_available_cv_;
};

} // namespace tpipeline