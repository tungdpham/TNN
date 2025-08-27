#pragma once

#include "nn/sequential.hpp"

#include "pipeline_communicator.hpp"
#include "task.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <omp.h>
#include <queue>
#include <string>

namespace tpipeline {

template <typename T = float> class PipelineStage {
public:
  explicit PipelineStage(
      std::unique_ptr<tnn::Sequential<T>> model,
      std::unique_ptr<PipelineCommunicator<T>,
                      std::function<void(PipelineCommunicator<T> *)>>
          communicator,
      const std::string &name = "")
      : model_(std::move(model)), communicator_(std::move(communicator)),
        name_(name), should_stop_(true), is_processing_(false) {}

  virtual ~PipelineStage() {}

protected:
  PipelineStage()
      : model_(nullptr), communicator_(nullptr), name_(""), should_stop_(true),
        is_processing_(false) {}

public:
  virtual void start() {
    if (!should_stop_) {
      std::cerr << "Stage " << name_ << " is already running" << std::endl;
      return;
    }

    should_stop_ = false;

    communicator_->set_message_notification_callback([this]() {
      std::lock_guard<std::mutex> lock(message_available_mutex_);
      message_available_cv_.notify_all();
    });
  }

  virtual void stop() {
    should_stop_ = true;
    message_available_cv_.notify_all();
  }

  void message_loop() {
    while (!should_stop_) {
      std::unique_lock<std::mutex> lock(message_available_mutex_);
      message_available_cv_.wait(lock, [this]() {
        return communicator_->has_input_message() || should_stop_;
      });

      if (should_stop_) {
        break;
      }

      while (communicator_->has_input_message()) {
        auto message = communicator_->dequeue_input_message();
        process_message(message);
      }
    }
  }

  virtual void process_message(const tpipeline::Message<T> &message) {
    switch (message.command_type) {
    case CommandType::FORWARD_TASK:
    case CommandType::BACKWARD_TASK: {

      process_task_message(message);

    } break;
    case CommandType::UPDATE_PARAMETERS:
      if (model_) {
        model_->update_parameters();
        auto response = Message<T>::parameters_updated(name_, "coordinator");
        communicator_->enqueue_output_message(response);
        communicator_->flush_output_messages();
      } else {
        std::cout << "Warning: No model available to update parameters"
                  << std::endl;
      }
      break;
    case CommandType::START_TRAINING:
      this->start();
      break;

    case CommandType::STOP_TRAINING:
      this->stop();
      break;

    case CommandType::STATUS_REQUEST: {
      auto response = Message<T>::status_message(
          (communicator_->has_task_message() ? "busy" : "idle"), name_,
          message.sender_id);
      communicator_->enqueue_output_message(response);
      communicator_->flush_output_messages();
      break;
    }

    case CommandType::ERROR_REPORT:
      if (message.has_text()) {
        std::cout << "Stage " << name_
                  << " received error: " << *message.text_data << " from "
                  << message.sender_id << std::endl;
      }
      break;

    case CommandType::PRINT_PROFILING:
      if (model_) {
        std::cout << "Received profiling request for stage " << name_
                  << std::endl;
        model_->print_profiling_summary();
      } else {
        std::cout << "Warning: No model available to print profiling data"
                  << std::endl;
      }
      break;
    case CommandType::CLEAR_PROFILING:
      if (model_) {
        model_->clear_profiling_data();
      } else {
        std::cout << "Warning: No model available to clear profiling data"
                  << std::endl;
      }
      break;
    default:
      std::cout << "Stage " << name_ << " received unknown command type: "
                << static_cast<int>(message.command_type) << std::endl;
      break;
    }
  }

  tnn::Sequential<T> *get_model() { return model_.get(); }
  std::string name() const { return name_; }
  PipelineCommunicator<T> *get_communicator() { return communicator_.get(); }

protected:
  void process_task_message(const tpipeline::Message<T> &message) {
    if (!message.is_task_message()) {
      auto error_msg = Message<T>::error_message(
          "Expected task payload but got different type", name_, "coordinator");
      communicator_->enqueue_output_message(error_msg);
      communicator_->flush_output_messages();
      return;
    }

    const auto &task = message.task.value();

    if (message.command_type == CommandType::FORWARD_TASK) {

      auto output_data = this->model_->forward(task.data, task.micro_batch_id);
      Task<T> output_task(TaskType::FORWARD, output_data, task.micro_batch_id);

      auto output_message =
          Message<T>::forward_task(output_task, name_, "next_stage");
      output_message.sequence_number = message.sequence_number;

      communicator_->send_to_next_stage(output_message);

    } else if (message.command_type == CommandType::BACKWARD_TASK) {

      auto output_data = this->model_->backward(task.data, task.micro_batch_id);
      Task<T> output_task(TaskType::BACKWARD, output_data, task.micro_batch_id);

      auto output_message =
          Message<T>::backward_task(output_task, name_, "prev_stage");
      output_message.sequence_number = message.sequence_number;

      communicator_->send_to_prev_stage(output_message);
    }

    communicator_->flush_output_messages();
  }

protected:
  std::unique_ptr<tnn::Sequential<T>> model_;
  std::unique_ptr<PipelineCommunicator<T>,
                  std::function<void(PipelineCommunicator<T> *)>>
      communicator_;
  std::string name_;

  std::atomic<bool> should_stop_;
  std::atomic<bool> is_processing_;

  std::mutex message_available_mutex_;
  std::condition_variable message_available_cv_;
};

} // namespace tpipeline