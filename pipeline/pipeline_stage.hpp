#pragma once

#include "../nn/sequential.hpp"
#include "pipeline_communicator.hpp"
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
                         std::unique_ptr<PipelineCommunicator<T>, std::function<void(PipelineCommunicator<T>*)>> communicator,
                         const std::string &name = "")
      : model_(std::move(model)), communicator_(std::move(communicator)),
        name_(name), should_stop_(false), is_processing_(false),
        thread_pool_(2) {}

  virtual ~PipelineStage() { stop(); }

  virtual void start() {
    should_stop_ = false;
    event_listener_future_ = thread_pool_.enqueue([this]() { event_listener(); });
    communicator_->set_message_notification_callback([this]() { notify_message_available(); });
  }

  virtual void stop() {
    should_stop_ = true;
    message_available_cv_.notify_all();
    if (event_listener_future_.valid())
      event_listener_future_.wait();
  }

  bool is_processing() const { return is_processing_; }
  tnn::Sequential<T> *get_model() { return model_.get(); }
  std::string name() const { return name_; }
  PipelineCommunicator<T> *get_communicator() { return communicator_.get(); }

protected:
  void event_listener() {
    while (!should_stop_) {
      std::unique_lock<std::mutex> lock(message_available_mutex_);
      message_available_cv_.wait(lock, [this]() {
        return should_stop_ || (communicator_->has_input_message() && !is_processing_);
      });

      if (communicator_->has_input_message() && !is_processing_) {
        is_processing_ = true;
        thread_pool_.enqueue([this]() {
          try {
            tpipeline::Message<T> message = communicator_->dequeue_input_message();
            process_message(message);
          } catch (const std::exception& e) {
            // Send error message to coordinator
            auto error_msg = Message<T>::error_message(
              std::string("Stage processing error: ") + e.what(), name_, "coordinator");
            communicator_->enqueue_output_message(error_msg);
            communicator_->flush_output_messages();
          }
          printf("Stage %s finished processing message\n", name_.c_str());
          is_processing_ = false;
          notify_message_available();
        });
      }
    }
  }

  void notify_message_available() {
    std::lock_guard<std::mutex> lock(message_available_mutex_);
    message_available_cv_.notify_one();
  }

  void process_message(const tpipeline::Message<T> &message) {
    switch (message.command_type) {
      case CommandType::FORWARD_TASK:
      case CommandType::BACKWARD_TASK:
        printf("Stage %s received task message\n", name_.c_str());
        process_task_message(message);
        printf("Stage %s processed task message\n", name_.c_str());
        break;
      case CommandType::UPDATE_PARAMETERS:
        printf("Stage %s received UPDATE_PARAMETERS command\n", name_.c_str());
        if (model_) {
          model_->update_parameters();
          //Send updated params confirmation back to coordinator
          auto response = Message<T>::parameters_updated(name_, "coordinator");
          communicator_->enqueue_output_message(response);
          communicator_->flush_output_messages();
        } else {
          printf("Warning: No model available to update parameters\n");
        }
        break;
      case CommandType::START_TRAINING:
        printf("Stage %s received START_TRAINING command\n", name_.c_str());
        this->start();
        break;
        
      case CommandType::STOP_TRAINING:
        printf("Stage %s received START_TRAINING command\n", name_.c_str());
        this->stop();
        break;
        
      case CommandType::STATUS_REQUEST: {
        printf("Stage %s received STATUS_REQUEST from %s\n", 
               name_.c_str(), message.sender_id.c_str());
        auto response = Message<T>::status_message(
            (communicator_->has_task_message() ? "busy" : "idle"),
            name_, message.sender_id);
        communicator_->enqueue_output_message(response);
        communicator_->flush_output_messages();
        break;
      }
      
      case CommandType::ERROR_REPORT:
        printf("Stage %s received ERROR_REPORT from %s\n", 
               name_.c_str(), message.sender_id.c_str());
        if (message.has_text()) {
          printf("Stage %s received error: %s from %s\n", 
                 name_.c_str(), message.text_data->c_str(), message.sender_id.c_str());
        }
        break;
        
      case CommandType::PRINT_PROFILING:
        printf("Stage %s received PRINT_PROFILING command\n", name_.c_str());
        if (model_) {
          model_->print_profiling_summary();
        } else {
          printf("Warning: No model available to print profiling data\n");
        }
        break;
      default:
        printf("Stage %s received unknown command type: %d\n", 
               name_.c_str(), static_cast<int>(message.command_type));
        break;
    }
  }

  void process_task_message(const tpipeline::Message<T> &message) {
    if (!message.is_task_message()) {
      auto error_msg = Message<T>::error_message(
        "Expected task payload but got different type", name_, "coordinator");
      communicator_->enqueue_output_message(error_msg);
      communicator_->flush_output_messages();
      return;
    }

    const auto& task = message.task.value();
    
    if (message.command_type == CommandType::FORWARD_TASK) {
      // Forward pass
      auto output_data = this->model_->forward(task.data);
      Task<T> output_task(TaskType::FORWARD, output_data, task.micro_batch_id); 
      
      auto output_message = Message<T>::forward_task(output_task, name_, "next_stage");
      output_message.sequence_number = message.sequence_number;
      
      communicator_->send_to_next_stage(output_message);
      
    } else if (message.command_type == CommandType::BACKWARD_TASK) {
      // Backward pass  
      auto output_data = this->model_->backward(task.data);
      Task<T> output_task(TaskType::BACKWARD, output_data, task.micro_batch_id); 
      
      auto output_message = Message<T>::backward_task(output_task, name_, "prev_stage");
      output_message.sequence_number = message.sequence_number;
      
      communicator_->send_to_prev_stage(output_message);
    }
    
    // Send all queued messages
    communicator_->flush_output_messages();
  }

protected:
  std::unique_ptr<tnn::Sequential<T>> model_;
  std::unique_ptr<PipelineCommunicator<T>, std::function<void(PipelineCommunicator<T>*)>> communicator_;
  std::string name_;

  std::atomic<bool> should_stop_;
  std::atomic<bool> is_processing_;
  std::atomic<bool> is_processing_task_;

  ThreadPool thread_pool_;
  std::future<void> event_listener_future_;

  std::mutex message_available_mutex_;
  std::condition_variable message_available_cv_;
};

} // namespace tpipeline