#pragma once

#include "../nn/sequential.hpp"
#include "pipeline_communicator.hpp"
#include "task.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <omp.h> // Include OpenMP header
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
        name_(name), should_stop_(true), is_processing_(false) {
  }

  virtual ~PipelineStage() { }

  void process_message(const tpipeline::Message<T> &message) {
    switch (message.command_type) {
    case CommandType::FORWARD_TASK:
    case CommandType::BACKWARD_TASK: {
      auto task_start = std::chrono::high_resolution_clock::now();
      process_task_message(message);
      auto task_end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                          task_end - task_start)
                          .count();
    } break;
    case CommandType::UPDATE_PARAMETERS:
      if (model_) {
        model_->update_parameters();
        auto response = Message<T>::parameters_updated(name_, "coordinator");
        communicator_->enqueue_output_message(response);
        communicator_->flush_output_messages();
      } else {
        printf("Warning: No model available to update parameters\n");
      }
      break;
    case CommandType::START_TRAINING:
      // this->start();
      break;

    case CommandType::STOP_TRAINING:
      // this->stop();
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
        printf("Stage %s received error: %s from %s\n", name_.c_str(),
               message.text_data->c_str(), message.sender_id.c_str());
      }
      break;

    case CommandType::PRINT_PROFILING:
      if (model_) {
        model_->print_profiling_summary();
      } else {
        printf("Warning: No model available to print profiling data\n");
      }
      break;
    default:
      printf("Stage %s received unknown command type: %d\n", name_.c_str(),
             static_cast<int>(message.command_type));
      break;
    }
  }

  bool is_processing() const { return is_processing_; }
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
      // Forward pass
      // NOTE: `this->model_->forward` can now safely contain its own
      // `#pragma omp parallel for` directives.
      auto output_data = this->model_->forward(task.data);
      Task<T> output_task(TaskType::FORWARD, output_data, task.micro_batch_id);

      auto output_message =
          Message<T>::forward_task(output_task, name_, "next_stage");
      output_message.sequence_number = message.sequence_number;

      communicator_->send_to_next_stage(output_message);

    } else if (message.command_type == CommandType::BACKWARD_TASK) {
      // Backward pass
      // NOTE: `this->model_->backward` can also contain `#pragma omp parallel
      // for`.
      auto output_data = this->model_->backward(task.data);
      Task<T> output_task(TaskType::BACKWARD, output_data, task.micro_batch_id);

      auto output_message =
          Message<T>::backward_task(output_task, name_, "prev_stage");
      output_message.sequence_number = message.sequence_number;

      communicator_->send_to_prev_stage(output_message);
    }

    // Send all queued messages
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