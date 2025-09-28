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

#include "communicator.hpp"
#include "old_binary_serializer.hpp"
#include "pipeline_stage.hpp"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

namespace tpipeline {

template <typename T = float> class PipelineCoordinator {
public:
  PipelineCoordinator(int num_stages, int num_microbatches, tnn::Sequential<T> model)
      : num_stages_(num_stages), num_microbatches_(num_microbatches), model_(std::move(model)) {
    if (num_stages < 1 || num_microbatches < 1) {
      throw std::invalid_argument("Number of stages and microbatches must be at least 1");
    }
    if (this->model_.get_layers().size() < static_cast<size_t>(num_stages)) {
      throw std::invalid_argument("Model must have at least as many layers as stages");
    }
  }

  ~PipelineCoordinator() {
    if (message_thread_.joinable()) {
      message_thread_.join();
    }
    coordinator_comm_.reset();
  }

  void add_message_callback() {
    this->coordinator_comm_->set_message_notification_callback([this]() {
      std::lock_guard<std::mutex> lock(this->message_notification_mutex_);
      this->message_notification_cv_.notify_all();
    });
  }

  void start() {
    for (const auto &stage_name : this->stage_names_) {
      auto start_msg =
          Message<T>::create_control_message(CommandType::TRAIN_MODE, "coordinator", stage_name);
      this->coordinator_comm_->send_message(start_msg);
    }

    message_thread_ = std::thread(&PipelineCoordinator::message_loop, this);

    std::cout << "Started all " << this->num_stages_ << " pipeline stages" << std::endl;
  }

  void stop() {
    for (const auto &stage_name : this->stage_names_) {
      auto stop_msg =
          Message<T>::create_control_message(CommandType::SHUTDOWN, "coordinator", stage_name);
      this->coordinator_comm_->send_message(stop_msg);
    }
    should_stop_ = true;

    message_notification_cv_.notify_all();

    if (message_thread_.joinable()) {
      message_thread_.join();
    }

    std::cout << "Stopped all pipeline stages" << std::endl;
  }

  void message_loop() {
    should_stop_ = false;
    while (!should_stop_) {
      std::unique_lock<std::mutex> lock(this->message_notification_mutex_);
      this->message_notification_cv_.wait(
          lock, [this]() { return this->coordinator_comm_->has_input_message() || should_stop_; });

      if (should_stop_) {
        std::cout << "Coordinator stopping message loop" << std::endl;
        break;
      }

      if (this->coordinator_comm_->message_count(CommandType::LOAD_REPORT) > 0) {
        std::vector<Message<T>> load_messages =
            this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::LOAD_REPORT);

        for (const auto &load_msg : load_messages) {
          if (load_msg.has_binary()) {
            std::vector<uint8_t> serialized_data = load_msg.get_binary();
            LoadTracker tracker = LoadTracker::deserialize(serialized_data);
            std::cout << "Received load report from " << load_msg.sender_id
                      << ": avg_forward_time=" << tracker.avg_forward_time_
                      << "ms, avg_backward_time=" << tracker.avg_backward_time_ << "ms\n";
          }
        }
      }
    }
  }

  void process_message(const Message<T> &message) {}

  /**
   * @brief Forwards input batch but does not wait for the result.
   * @param input The input tensor to be processed.
   * @param microbatch_id The ID of the microbatch (0 to num_microbatches - 1).
   */
  void forward(const Tensor<T> &input, size_t microbatch_id) {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &first_stage = this->stage_names_[0];

    Task<T> task{TaskType::FORWARD, input, microbatch_id};
    auto forward_msg = Message<T>::forward_task(task, "coordinator", first_stage);
    forward_msg.sequence_number = microbatch_id;

    this->coordinator_comm_->send_message(forward_msg);
  }

  /**
   * @brief Sends the backward gradient to the last stage.
   * @param gradient The gradient tensor to be backpropagated.
   * @param microbatch_id The ID of the microbatch (0 to num_microbatches - 1).
   */
  void backward(const Tensor<T> &gradient, size_t microbatch_id) {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &last_stage = this->stage_names_.back();

    Task<T> task{TaskType::BACKWARD, gradient, microbatch_id};
    auto backward_msg = Message<T>::backward_task(task, "coordinator", last_stage);
    backward_msg.sequence_number = microbatch_id;

    this->coordinator_comm_->send_message(backward_msg);
  }

  /**
   * @brief Computes the loss given predictions and targets using the model's loss function.
   * @param predictions The predicted output tensor.
   * @param targets The target output tensor.
   * @return The computed loss value.
   */
  float compute_loss(const Tensor<T> &predictions, const Tensor<T> &targets) {
    if (!this->model_.loss_function()) {
      throw std::runtime_error("No loss function defined in the model");
    }
    return this->model_.loss_function()->compute_loss(predictions, targets);
  }

  void update_parameters() {
    for (const auto &stage_name : this->stage_names_) {
      auto update_msg = Message<T>::create_signal_message(CommandType::UPDATE_PARAMETERS, true,
                                                          "coordinator", stage_name);
      this->coordinator_comm_->send_message(update_msg);
    }

    wait_for_parameter_updates();
  }

  bool send_params(const std::string &stage_id, const tnn::Partition &partition) {
    try {
      std::vector<Tensor<T> *> params = model_.parameters(partition);
      std::vector<Tensor<T>> params_copy;
      for (const auto &param_ptr : params) {
        if (param_ptr) {
          params_copy.push_back(*param_ptr);
        }
      }
      auto serialized_params = BinarySerializer::serialize_parameters(params_copy);

      auto params_msg = Message<T>::load_params_message(serialized_params, "coordinator", stage_id);

      this->coordinator_comm_->send_message(params_msg);
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Failed to send parameters to stage " << stage_id << ": " << e.what() << '\n';
      return false;
    }
  }

  /**
   * @brief Waits for a specified number of confirmations for a given command type.
   * @param type The command type to wait for (e.g., CommandType::UPDATE_PARAMETERS).
   * @param expected_count The number of confirmations to wait for.
   * @param timeout The maximum time to wait in seconds (default is 60 seconds).
   */
  bool join(const CommandType type, const size_t expected_count,
            const size_t timeout_duration = 60) {
    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_duration);

    bool success =
        message_notification_cv_.wait_until(lock, timeout, [this, type, expected_count]() {
          return this->coordinator_comm_->message_count(type) >= expected_count;
        });

    return success;
  }

  /**
   * @brief Forwards all microbatches and immediately compute loss and backward pass as results
   * arrive.
   * @param microbatch_inputs A vector of input tensors for each microbatch.
   * @param microbatch_labels A vector of target tensors for each microbatch.
   */
  void async_process_batch(std::vector<Tensor<T>> &microbatch_inputs,
                           std::vector<Tensor<T>> &microbatch_labels) {
    if (microbatch_inputs.size() != static_cast<size_t>(this->num_microbatches_) ||
        microbatch_labels.size() != static_cast<size_t>(this->num_microbatches_)) {
      throw std::invalid_argument("Microbatch size mismatch with coordinator configuration");
    }

    for (int i = 0; i < this->num_microbatches_; ++i) {
      this->forward(microbatch_inputs[i], i);
    }

    // Assuming no microbatch are lost during transmission/processing. May need additional handling
    // for production use.
    int processed_microbatches_ = 0;
    while (processed_microbatches_ < this->num_microbatches_) {
      std::unique_lock<std::mutex> lock(message_notification_mutex_);
      message_notification_cv_.wait(lock, [this]() {
        return this->coordinator_comm_->message_count(CommandType::FORWARD_TASK) > 0;
      });
      std::vector<Message<T>> FORWARD_TASKs =
          this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::FORWARD_TASK);

      for (const auto &forward_msg : FORWARD_TASKs) {
        if (forward_msg.has_task()) {
          ++processed_microbatches_;

          const Task<T> &task = forward_msg.get_task();

          Tensor<T> predictions = task.data;
          Tensor<T> targets = microbatch_labels[task.micro_batch_id];
          Tensor<T> gradient = this->model_.loss_function()->compute_gradient(predictions, targets);

          this->backward(gradient, task.micro_batch_id);
        }
      }
    }

    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    message_notification_cv_.wait(lock, [this]() {
      return this->coordinator_comm_->message_count(CommandType::BACKWARD_TASK) >=
             static_cast<size_t>(this->num_microbatches_);
    });

    this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::BACKWARD_TASK);
  }

  /**
   * @brief Sends a request to all stages for load report
   */
  void request_load_report_from_all_stages() {
    for (const auto &stage_name : this->stage_names_) {
      auto load_msg =
          Message<T>::create_control_message(CommandType::REPORT_LOAD, "coordinator", stage_name);
      this->coordinator_comm_->send_message(load_msg);
    }
  }

  /**
   * @brief Requests all stages to print their profiling data.
   */
  void print_profiling_on_all_stages() {
    for (const auto &stage_name : this->stage_names_) {
      auto profiling_msg = Message<T>::create_control_message(CommandType::PRINT_PROFILING,
                                                              "coordinator", stage_name);
      this->coordinator_comm_->send_message(profiling_msg);
    }
  }

  /**
   * @brief Requests all stages to clear their profiling data.
   */
  void clear_profiling_data() {
    for (const auto &stage_name : this->stage_names_) {
      auto clear_msg = Message<T>::create_control_message(CommandType::CLEAR_PROFILING,
                                                          "coordinator", stage_name);
      this->coordinator_comm_->send_message(clear_msg);
    }
  }

  void request_status_from_all_stages() {
    for (const auto &stage_name : this->stage_names_) {
      auto status_msg = Message<T>(CommandType::STATUS_REQUEST, true, "coordinator", stage_name);
      this->coordinator_comm_->send_message(status_msg);
    }
  }

  std::vector<Message<T>> dequeue_all_messages(CommandType target_type) {
    return this->coordinator_comm_->dequeue_all_messages_by_type(target_type);
  }

  bool wait_for_config_received() {
    return join(CommandType::CONFIG_RECEIVED, this->num_stages_, 30);
  }

  bool wait_for_params_loaded() { return join(CommandType::PARAMS_LOADED, this->num_stages_, 30); }

  bool wait_for_parameter_updates() {
    return join(CommandType::PARAMETERS_UPDATED, this->num_stages_, 60);
  }

protected:
  int num_stages_;
  int num_microbatches_;
  bool should_stop_ = true;
  tnn::Sequential<T> model_;
  std::shared_ptr<PipelineCommunicator<T>> coordinator_comm_;
  std::vector<std::string> stage_names_;
  std::vector<tnn::Partition> partitions_;
  std::thread message_thread_;

  mutable std::mutex message_notification_mutex_;
  mutable std::condition_variable message_notification_cv_;
};

} // namespace tpipeline