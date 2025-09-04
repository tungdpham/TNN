/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"

#include "pipeline_communicator.hpp"
#include "pipeline_stage.hpp"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

namespace tpipeline {

template <typename T = float> class PipelineCoordinator {
public:
  PipelineCoordinator(int num_stages = 4, int num_microbatches = 4)
      : num_stages_(num_stages), num_microbatches_(num_microbatches) {
    if (num_stages < 1 || num_microbatches < 1) {
      throw std::invalid_argument(
          "Number of stages and microbatches must be at least 1");
    }

    std::cout << "PipelineCoordinator initialized with " << num_stages_
              << " stages and " << num_microbatches_ << " microbatches."
              << std::endl;
  }

  virtual ~PipelineCoordinator() = default;

  void add_message_callback() {
    this->coordinator_comm_->set_message_notification_callback([this]() {
      std::lock_guard<std::mutex> lock(this->message_notification_mutex_);
      this->message_notification_cv_.notify_all();
    });
  }

  void start() {
    for (const auto &stage_name : this->stage_names_) {
      // currently defaults to train mode
      auto start_msg = Message<T>::create_control_message(
          CommandType::TRAIN_MODE, "coordinator", stage_name);
      this->coordinator_comm_->send_message(start_msg);
    }

    std::cout << "Started all " << this->num_stages_ << " pipeline stages"
              << std::endl;
  }

  void stop() {
    for (const auto &stage_name : this->stage_names_) {
      auto stop_msg = Message<T>::create_control_message(
          CommandType::SHUTDOWN, "coordinator", stage_name);
      this->coordinator_comm_->send_message(stop_msg);
    }

    std::cout << "Stopped all pipeline stages" << std::endl;
  }

  void forward(const Tensor<T> &input, int microbatch_id) {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &first_stage = this->stage_names_[0];

    Task<T> task{TaskType::FORWARD, input, microbatch_id};
    auto forward_msg =
        Message<T>::forward_task(task, "coordinator", first_stage);
    forward_msg.sequence_number = microbatch_id;

    this->coordinator_comm_->send_message(forward_msg);
  }

  void backward(const Tensor<T> &gradient, int microbatch_id) {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &last_stage = this->stage_names_.back();

    Task<T> task{TaskType::BACKWARD, gradient, microbatch_id};
    auto backward_msg =
        Message<T>::backward_task(task, "coordinator", last_stage);
    backward_msg.sequence_number = microbatch_id;

    this->coordinator_comm_->send_message(backward_msg);
  }

  void join(const bool direction) {
    const size_t expected_task_count_ = this->num_microbatches_;

    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);

    bool success = message_notification_cv_.wait_until(
        lock, timeout, [this, direction, expected_task_count_]() {
          if (direction) {
            return this->coordinator_comm_->forward_message_count() >=
                   expected_task_count_;
          } else {
            return this->coordinator_comm_->backward_message_count() >=
                   expected_task_count_;
          }
        });

    if (!success) {
      std::cout << "Warning: join() timed out waiting for task messages. "
                << "Expected: " << expected_task_count_ << ", Got: "
                << (direction
                        ? this->coordinator_comm_->forward_message_count()
                        : this->coordinator_comm_->backward_message_count())
                << '\n';
    }
    return;
  }

  void async_process_batch(std::vector<Tensor<T>> &microbatch_inputs,
                           std::vector<Tensor<T>> &microbatch_labels) {
    if (microbatch_inputs.size() !=
            static_cast<size_t>(this->num_microbatches_) ||
        microbatch_labels.size() !=
            static_cast<size_t>(this->num_microbatches_)) {
      throw std::invalid_argument(
          "Microbatch size mismatch with coordinator configuration");
    }

    if (loss_function_ == nullptr) {
      throw std::runtime_error(
          "Loss function not set for distributed coordinator");
    }

    for (int i = 0; i < this->num_microbatches_; ++i) {
      this->forward(microbatch_inputs[i], i);
    }

    // Backward on completion of any microbatch
    int processed_microbatches_ = 0;
    while (processed_microbatches_ < this->num_microbatches_) {
      std::unique_lock<std::mutex> lock(message_notification_mutex_);
      message_notification_cv_.wait(lock, [this]() {
        return this->coordinator_comm_->forward_message_count() > 0;
      });
      std::vector<Message<T>> forward_messages =
          this->coordinator_comm_->dequeue_all_messages_by_type(
              CommandType::FORWARD_TASK);

      for (const auto &forward_msg : forward_messages) {
        if (forward_msg.has_task()) {
          ++processed_microbatches_;
          // Process the forward task
          const auto &task = forward_msg.get_task();

          Tensor<T> predictions = task.data;
          Tensor<T> targets = microbatch_labels[task.micro_batch_id];
          Tensor<T> gradient =
              loss_function_->compute_gradient(predictions, targets);

          // Send backward task
          this->backward(gradient, task.micro_batch_id);
        }
      }
    }

    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    // Wait for all backward tasks to complete
    message_notification_cv_.wait(lock, [this]() {
      return this->coordinator_comm_->backward_message_count() >=
             static_cast<size_t>(this->num_microbatches_);
    });

    this->coordinator_comm_->dequeue_all_messages_by_type(
        CommandType::BACKWARD_TASK);
  }

  void print_profiling_on_all_stages() {
    for (const auto &stage_name : this->stage_names_) {
      auto profiling_msg = Message<T>::create_control_message(
          CommandType::PRINT_PROFILING, "coordinator", stage_name);
      this->coordinator_comm_->send_message(profiling_msg);
    }
  }

  void clear_profiling_data() {
    for (const auto &stage_name : this->stage_names_) {
      auto clear_msg = Message<T>::create_control_message(
          CommandType::CLEAR_PROFILING, "coordinator", stage_name);
      this->coordinator_comm_->send_message(clear_msg);
    }
  }

  void request_status_from_all_stages() {
    for (const auto &stage_name : this->stage_names_) {
      auto status_msg = Message<T>(CommandType::STATUS_REQUEST, true,
                                   "coordinator", stage_name);
      this->coordinator_comm_->send_message(status_msg);
    }
  }

  void update_parameters() {
    for (const auto &stage_name : this->stage_names_) {
      auto update_msg = Message<T>::create_signal_message(
          CommandType::UPDATE_PARAMETERS, true, "coordinator", stage_name);
      this->coordinator_comm_->send_message(update_msg);
    }

    wait_for_parameter_updates();
  }

  std::vector<Message<T>> dequeue_all_messages(CommandType target_type) {
    return this->coordinator_comm_->dequeue_all_messages_by_type(target_type);
  }

  bool wait_for_config_received() {
    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(60);

    bool success = message_notification_cv_.wait_until(lock, timeout, [this]() {
      return this->coordinator_comm_->message_count_by_type(
                 CommandType::CONFIG_RECEIVED) >=
             static_cast<size_t>(this->num_stages_);
    });

    if (!success) {
      std::cout << "Timeout waiting for config received confirmations\n";
      return false;
    }

    std::cout << "All stages confirmed configuration received!\n";

    return true;
  }

  bool wait_for_weights_received() {
    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(60);

    bool success = message_notification_cv_.wait_until(lock, timeout, [this]() {
      return this->coordinator_comm_->message_count_by_type(
                 CommandType::WEIGHTS_RECEIVED) >=
             static_cast<size_t>(this->num_stages_);
    });

    if (!success) {
      std::cout << "Timeout waiting for weights received confirmations\n";
      return false;
    }

    std::cout << "All stages confirmed weights received!\n";

    return true;
  }

  void set_loss_function(std::unique_ptr<tnn::Loss<T>> loss) {
    loss_function_ = std::move(loss);
  }

  const std::unique_ptr<tnn::Loss<T>> &get_loss_function() const {
    return loss_function_;
  }

protected:
  std::unique_ptr<tnn::Loss<T>> loss_function_;
  int num_stages_;
  int num_microbatches_;
  std::shared_ptr<PipelineCommunicator<T>> coordinator_comm_;
  std::vector<std::string> stage_names_;

  mutable std::mutex message_notification_mutex_;
  mutable std::condition_variable message_notification_cv_;

private:
  void wait_for_parameter_updates() {
    int confirmations = 0;
    (void)confirmations; // Suppress unused variable warning

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);

    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    bool success = message_notification_cv_.wait_until(lock, timeout, [this]() {
      return this->coordinator_comm_->params_updated_count() >=
             static_cast<size_t>(this->num_stages_);
    });

    if (!success) {
      std::cout << "Warning: wait_for_parameter_updates() timed out. "
                << "Expected: " << this->num_stages_
                << ", Got: " << this->coordinator_comm_->params_updated_count()
                << '\n';
      return;
    }
    return;
  }
};

} // namespace tpipeline