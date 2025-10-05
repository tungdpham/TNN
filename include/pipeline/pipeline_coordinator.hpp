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
#include "nn/partitioner.hpp"
#include "old_binary_serializer.hpp"
#include "pipeline_stage.hpp"
#include "utils/avx2.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

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

      // if (this->coordinator_comm_->message_count(CommandType::LOAD_REPORT) > 0) {
      //   std::vector<Message<T>> load_messages =
      //       this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::LOAD_REPORT);

      //   for (const auto &load_msg : load_messages) {
      //     if (load_msg.has_binary()) {
      //       std::vector<uint8_t> serialized_data = load_msg.get_binary();
      //       LoadTracker tracker = LoadTracker::deserialize(serialized_data);
      //       std::cout << "Received load report from " << load_msg.sender_id
      //                 << ": avg_forward_time=" << tracker.avg_forward_time_
      //                 << "ms, avg_backward_time=" << tracker.avg_backward_time_ << "ms\n";
      //     }
      //   }
      // }
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

    bool success = join(CommandType::PARAMETERS_UPDATED, this->num_stages_, 60);
    if (!success) {
      std::cerr << "Warning: Timeout waiting for parameter update confirmations from all stages\n";
    }
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
   * @brief Intelligently sends parameters only to stages that need them based on partition changes.
   * @param old_partitions The previous partition configuration
   * @param new_partitions The new partition configuration
   * @return true if all necessary parameters were sent successfully, false otherwise
   */
  bool send_updated_parameters(const std::vector<tnn::Partition> &old_partitions,
                               const std::vector<tnn::Partition> &new_partitions) {
    if (old_partitions.size() != new_partitions.size() ||
        new_partitions.size() != stage_names_.size()) {
      std::cerr << "Partition size mismatch in send_updated_parameters\n";
      return false;
    }

    std::vector<std::future<bool>> param_futures;

    for (size_t i = 0; i < stage_names_.size(); ++i) {
      const std::string &stage_name = stage_names_[i];
      const auto &old_partition = old_partitions[i];
      const auto &new_partition = new_partitions[i];

      // Check if this stage's partition actually changed
      bool partition_changed = (old_partition.start_layer != new_partition.start_layer ||
                                old_partition.end_layer != new_partition.end_layer);

      if (partition_changed) {
        std::cout << "Partition changed for stage " << stage_name << ": ["
                  << old_partition.start_layer << "," << old_partition.end_layer << ") -> ["
                  << new_partition.start_layer << "," << new_partition.end_layer << ")\n";

        auto future = std::async(std::launch::async, [this, stage_name, new_partition]() {
          return this->send_params(stage_name, new_partition);
        });
        param_futures.push_back(std::move(future));
      } else {
        std::cout << "No partition change for stage " << stage_name
                  << ", skipping parameter update\n";
      }
    }

    // Wait for all parameter transfers to complete
    bool all_params_sent = true;
    for (auto &future : param_futures) {
      if (!future.get()) {
        all_params_sent = false;
      }
    }

    return all_params_sent;
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
  float async_process_batch(std::vector<Tensor<T>> &microbatch_inputs,
                            std::vector<Tensor<T>> &microbatch_labels) {
    if (microbatch_inputs.size() != static_cast<size_t>(this->num_microbatches_) ||
        microbatch_labels.size() != static_cast<size_t>(this->num_microbatches_)) {
      throw std::invalid_argument("Microbatch size mismatch with coordinator configuration");
    }

    for (int i = 0; i < this->num_microbatches_; ++i) {
      this->forward(microbatch_inputs[i], i);
    }

    float total_loss = 0.0f;

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
          float loss = this->compute_loss(predictions, targets);
          total_loss += loss;
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

    return total_loss;
  }

  /**
   * @brief Sends a request to all stages for load report
   */
  void balance_load() {
    std::cout << "Starting load balancing procedure...\n";

    // Request load reports from all stages
    for (const auto &stage_name : this->stage_names_) {
      auto load_msg =
          Message<T>::create_control_message(CommandType::REPORT_LOAD, "coordinator", stage_name);
      this->coordinator_comm_->send_message(load_msg);
    }

    // Wait for all load reports to arrive
    bool received_all_reports = join(CommandType::LOAD_REPORT, this->num_stages_, 30);
    if (!received_all_reports) {
      std::cerr << "Warning: Not all stages reported load data within timeout. Using current "
                   "partitions.\n";
      return;
    }

    // Collect and process load reports
    std::vector<Message<T>> load_messages =
        this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::LOAD_REPORT);

    if (load_messages.size() != static_cast<size_t>(this->num_stages_)) {
      std::cerr << "Warning: Expected " << this->num_stages_ << " load reports, got "
                << load_messages.size() << ". Using current partitions.\n";
      return;
    }

    std::map<std::string, LoadTracker> load_trackers;

    // Collect load trackers by stage id
    for (const auto &load_msg : load_messages) {
      if (load_msg.has_binary()) {
        try {
          std::vector<uint8_t> serialized_data = load_msg.get_binary();
          LoadTracker tracker = LoadTracker::deserialize(serialized_data);
          load_trackers[load_msg.sender_id] = tracker;

          std::cout << "Received load report from " << load_msg.sender_id
                    << ": avg_forward_time=" << tracker.avg_forward_time_
                    << "ms, avg_backward_time=" << tracker.avg_backward_time_ << "ms\n";
          // NOTE: memory usage report is broken, needs some fixing. Just use top command for now.
          // std::cout << "  avg_cpu_utilization=" << tracker.avg_cpu_utilization_
          //           << "%, max_memory_usage=" << tracker.max_memory_usage_ << "MB\n";
        } catch (const std::exception &e) {
          std::cerr << "Warning: Failed to deserialize load data from " << load_msg.sender_id
                    << ": " << e.what() << "\n";
        }
      }
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
    bool all_printed = join(CommandType::PROFILING_PRINTED, this->num_stages_, 30);
    if (!all_printed) {
      std::cerr << "Warning: Not all stages confirmed profiling print within timeout.\n";
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
    bool all_cleared = join(CommandType::PROFILING_CLEARED, this->num_stages_, 30);
    if (!all_cleared) {
      std::cerr << "Warning: Not all stages confirmed profiling clear within timeout.\n";
    }
  }

  /**
   * @brief Collects current parameters from all stages to ensure coordinator has up-to-date
   * weights.
   * @return true if all parameters were collected successfully, false otherwise
   */
  bool collect_current_parameters() {
    std::cout << "Collecting current parameters from all stages...\n";

    // Request parameters from all stages
    for (const auto &stage_name : this->stage_names_) {
      auto params_request_msg =
          Message<T>::create_control_message(CommandType::SEND_PARAMS, "coordinator", stage_name);
      this->coordinator_comm_->send_message(params_request_msg);
    }

    // Wait for all parameter responses
    bool received_all_params = join(CommandType::PARAMS_TRANSFER, this->num_stages_, 30);
    if (!received_all_params) {
      std::cerr << "Warning: Not all stages sent their parameters within timeout.\n";
      return false;
    }

    // Collect parameter messages
    std::vector<Message<T>> params_messages =
        this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::PARAMS_TRANSFER);

    if (params_messages.size() != static_cast<size_t>(this->num_stages_)) {
      std::cerr << "Warning: Expected " << this->num_stages_ << " parameter messages, got "
                << params_messages.size() << ".\n";
      return false;
    }

    // Update model parameters with received data
    std::map<std::string, std::vector<Tensor<T>>> stage_parameters;

    for (const auto &params_msg : params_messages) {
      if (params_msg.has_binary()) {
        try {
          std::vector<uint8_t> serialized_data = params_msg.get_binary();
          auto params = BinarySerializer::deserialize_parameters<T>(serialized_data);
          stage_parameters[params_msg.sender_id] = std::move(params);

          std::cout << "Received " << params.size() << " parameters from " << params_msg.sender_id
                    << "\n";
        } catch (const std::exception &e) {
          std::cerr << "Warning: Failed to deserialize parameters from " << params_msg.sender_id
                    << ": " << e.what() << "\n";
          return false;
        }
      }
    }

    // Reconstruct the full model from stage parameters
    try {
      // Find the stage index for each stage name and update model accordingly
      for (size_t i = 0; i < stage_names_.size(); ++i) {
        const std::string &stage_name = stage_names_[i];
        auto it = stage_parameters.find(stage_name);
        if (it != stage_parameters.end()) {
          // Update model parameters for this partition
          auto model_params = model_.parameters(partitions_[i]);
          const auto &stage_params = it->second;

          if (model_params.size() == stage_params.size()) {
            for (size_t j = 0; j < model_params.size(); ++j) {
              if (model_params[j]) {
                // Check if tensors have the same size before copying
                if (model_params[j]->size() == stage_params[j].size()) {
                  utils::avx2_copy(stage_params[j].data(), model_params[j]->data(),
                                   model_params[j]->size());
                } else {
                  std::cerr << "Warning: Tensor size mismatch for parameter " << j << " in stage "
                            << stage_name << "\n";
                }
              }
            }
          } else {
            std::cerr << "Warning: Parameter count mismatch for stage " << stage_name
                      << ". Expected " << model_params.size() << ", got " << stage_params.size()
                      << "\n";
          }
        }
      }

      std::cout << "Successfully updated coordinator model with current parameters.\n";
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Error updating model parameters: " << e.what() << "\n";
      return false;
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

protected:
  /**
   * @brief Sends configuration to a specific stage
   * @param stage_id The stage identifier
   * @param partition The partition containing layer information for this stage
   * @return true if configuration was sent successfully, false otherwise
   */
  bool send_config(const std::string &stage_id, const tnn::Partition &partition) {
    try {
      // Create a model segment for this partition
      std::vector<tnn::Partition> partitions = {partition};
      auto stage_model = model_.split(partitions)[0];

      // Create stage configuration
      StageConfig config;
      config.stage_id = stage_id;
      config.model_config = stage_model.get_config();
      config.model_config["name"] = stage_id;

      // Set stage index based on position in stage_names_
      auto it = std::find(stage_names_.begin(), stage_names_.end(), stage_id);
      if (it != stage_names_.end()) {
        config.stage_index = static_cast<int>(std::distance(stage_names_.begin(), it));
      } else {
        config.stage_index = 0; // fallback
      }

      // Send configuration as JSON
      std::string config_json = config.to_json().dump();
      auto config_msg = Message<T>::create_text_message(CommandType::CONFIG_TRANSFER, config_json,
                                                        "coordinator", stage_id);

      this->coordinator_comm_->send_message(config_msg);

      std::cout << "Sent configuration to stage " << stage_id << " for partition ["
                << partition.start_layer << ", " << partition.end_layer << ")\n";

      return true;
    } catch (const std::exception &e) {
      std::cerr << "Failed to send configuration to stage " << stage_id << ": " << e.what() << '\n';
      return false;
    }
  }

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