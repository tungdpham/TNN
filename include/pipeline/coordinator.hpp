/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/loss.hpp"
#include "nn/sequential.hpp"

#include "communicator.hpp"
#include "partitioner/partitioner.hpp"
#include "stage_config.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace tnn {

class Coordinator {
public:
  Coordinator(Sequential<float> model) : model_(std::move(model)) {}

  virtual ~Coordinator() {
    if (message_thread_.joinable()) {
      message_thread_.join();
    }
    coordinator_comm_.reset();
  }

  void initialize() {
    for (int i = 0; i < num_stages_; ++i) {
      stage_names_.push_back("stage_" + std::to_string(i));
    }
    initialize_partitions();
    initialize_topology();
  }

  void set_partitioner(std::unique_ptr<Partitioner<float>> tnn) { partitioner_ = std::move(tnn); }

  void set_loss_function(std::unique_ptr<Loss<float>> loss) {
    if (!loss) {
      throw std::invalid_argument("Loss function cannot be null");
    }
    loss_function_ = std::move(loss);
  }

  int num_stages() const { return num_stages_; }

  bool set_num_microbatches(int num_microbatches) {
    if (num_microbatches <= 0) {
      std::cerr << "Number of microbatches must be positive\n";
      return false;
    }
    num_microbatches_ = num_microbatches;
    return true;
  }

  int num_microbatches() const { return num_microbatches_; }

  void add_message_callback() {
    this->coordinator_comm_->set_message_notification_callback([this]() {
      std::lock_guard<std::mutex> lock(this->message_notification_mutex_);
      this->message_notification_cv_.notify_all();
    });
  }

  void start() {
    for (const auto &stage_name : this->stage_names_) {
      Message start_msg(stage_name, CommandType::TRAIN_MODE, std::monostate{});
      this->coordinator_comm_->send_message(start_msg);
    }

    // message_thread_ = std::thread(&Coordinator::message_loop, this);

    std::cout << "Started all " << this->num_stages_ << " pipeline stages" << std::endl;
  }

  void stop() {
    for (const auto &stage_name : this->stage_names_) {
      Message stop_msg(stage_name, CommandType::SHUTDOWN, std::monostate{});
      this->coordinator_comm_->send_message(stop_msg);
    }
    should_stop_ = true;

    message_notification_cv_.notify_all();

    if (message_thread_.joinable()) {
      message_thread_.join();
    }

    this->coordinator_comm_.reset();

    std::cout << "Stopped all pipeline stages" << std::endl;
  }

  void process_message(const Message &message) {}

  /**
   * @brief Forwards input batch but does not wait for the result.
   * @param input The input tensor to be processed.
   * @param microbatch_id The ID of the microbatch (0 to num_microbatches - 1).
   */
  void forward(const Tensor<float> &input, size_t microbatch_id) {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &first_stage = this->stage_names_[0];

    Job<float> job{input, microbatch_id};
    Message forward_msg(first_stage, CommandType::FORWARD_JOB, job);

    this->coordinator_comm_->send_message(forward_msg);
  }

  /**
   * @brief Sends the backward gradient to the last stage.
   * @param gradient The gradient tensor to be backpropagated.
   * @param microbatch_id The ID of the microbatch (0 to num_microbatches - 1).
   */
  void backward(const Tensor<float> &gradient, size_t microbatch_id) {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &last_stage = this->stage_names_.back();

    Job<float> job{gradient, microbatch_id};
    Message backward_msg(last_stage, CommandType::BACKWARD_JOB, job);

    this->coordinator_comm_->send_message(backward_msg);
  }

  /**
   * @brief Computes the loss given predictions and targets using the model's loss function.
   * @param predictions The predicted output tensor.
   * @param targets The target output tensor.
   * @return The computed loss value.
   */
  float compute_loss(const Tensor<float> &predictions, const Tensor<float> &targets) {
    if (!loss_function_) {
      throw std::runtime_error("Loss function is not set in the coordinator");
    }
    float loss = 0.0f;
    loss_function_->compute_loss(predictions, targets, loss);
    return loss;
  }

  /**
   * @brief Computes gradient of the loss with respect to predictions using the model's loss
   * function.
   */
  Tensor<float> compute_gradient(const Tensor<float> &predictions, const Tensor<float> &targets) {
    if (!loss_function_) {
      throw std::runtime_error("Loss function is not set in the coordinator");
    }
    Tensor<float> gradient;
    loss_function_->compute_gradient(predictions, targets, gradient);
    return gradient;
  }

  void update_parameters() {
    for (const auto &stage_name : this->stage_names_) {
      Message update_msg(stage_name, CommandType::UPDATE_PARAMETERS, std::monostate{});
      this->coordinator_comm_->send_message(update_msg);
    }

    bool success = join(CommandType::PARAMETERS_UPDATED, this->num_stages_, 60);
    if (!success) {
      std::cerr << "Warning: Timeout waiting for parameter update confirmations from all stages\n";
    }
  }

  bool send_params(const std::string &stage_id, const Partition &partition) {
    try {
      throw new std::runtime_error("Not implemented yet");
    } catch (const std::exception &e) {
      std::cerr << "Failed to send parameters to stage " << stage_id << ": " << e.what() << '\n';
      return false;
    }
  }

  /**
   * @brief Intelligently sends parameters only to stages that need them based on partition
   * changes.
   * @param old_partitions The previous partition configuration
   * @param new_partitions The new partition configuration
   * @return true if all necessary parameters were sent successfully, false otherwise
   */
  bool send_updated_parameters(const std::vector<Partition> &old_partitions,
                               const std::vector<Partition> &new_partitions) {
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
  float async_process_batch(std::vector<Tensor<float>> &microbatch_inputs,
                            std::vector<Tensor<float>> &microbatch_labels) {
    if (microbatch_inputs.size() != static_cast<size_t>(this->num_microbatches_) ||
        microbatch_labels.size() != static_cast<size_t>(this->num_microbatches_)) {
      throw std::invalid_argument("Microbatch size mismatch with coordinator configuration");
    }

    for (int i = 0; i < this->num_microbatches_; ++i) {
      this->forward(microbatch_inputs[i], i);
    }

    float total_loss = 0.0f;

    // Assuming no microbatch are lost during transmission/processing. May need additional
    // handling for production use.
    int processed_microbatches_ = 0;
    while (processed_microbatches_ < this->num_microbatches_) {
      std::unique_lock<std::mutex> lock(message_notification_mutex_);
      message_notification_cv_.wait(lock, [this]() {
        return this->coordinator_comm_->message_count(CommandType::FORWARD_JOB) > 0;
      });
      std::vector<Message> FORWARD_JOBs =
          this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::FORWARD_JOB);

      for (auto &forward_msg : FORWARD_JOBs) {
        if (forward_msg.has_type<Job<float>>()) {
          ++processed_microbatches_;

          Job<float> &job = forward_msg.get<Job<float>>();
          Tensor<float> &predictions = job.data;
          Tensor<float> &targets = microbatch_labels[job.micro_batch_id];
          float loss = 0.0f;
          loss_function_->compute_loss(predictions, targets, loss);
          total_loss += loss;
          Tensor<float> gradient;
          loss_function_->compute_gradient(predictions, targets, gradient);
          this->backward(gradient, job.micro_batch_id);
        }
      }
    }

    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    message_notification_cv_.wait(lock, [this]() {
      return this->coordinator_comm_->message_count(CommandType::BACKWARD_JOB) >=
             static_cast<size_t>(this->num_microbatches_);
    });

    this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::BACKWARD_JOB);

    return total_loss;
  }

  /**
   * @brief Sends a request to all stages for load report
   */
  void balance_load() {
    std::cout << "Starting load balancing procedure...\n";

    // Request load reports from all stages
    for (const auto &stage_name : this->stage_names_) {
      Message load_msg(stage_name, CommandType::REPORT_LOAD, std::monostate{});
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
    std::vector<Message> load_messages =
        this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::LOAD_REPORT);

    if (load_messages.size() != static_cast<size_t>(this->num_stages_)) {
      std::cerr << "Warning: Expected " << this->num_stages_ << " load reports, got "
                << load_messages.size() << ". Using current partitions.\n";
      return;
    }

    std::map<std::string, LoadTracker> load_trackers;

    // Collect load trackers by stage id
    for (auto &load_msg : load_messages) {
      if (load_msg.has_type<LoadTracker>()) {
        try {
          LoadTracker tracker = load_msg.get<LoadTracker>();
          load_trackers[load_msg.header.sender_id] = tracker;

          std::cout << "Received load report from " << load_msg.header.sender_id
                    << ": avg_forward_time=" << tracker.avg_forward_time_
                    << "ms, avg_backward_time=" << tracker.avg_backward_time_ << "ms\n";
          // NOTE: memory usage report is broken, needs some fixing. Just use top command for now.
          // std::cout << "  avg_cpu_utilization=" << tracker.avg_cpu_utilization_
          //           << "%, max_memory_usage=" << tracker.max_memory_usage_ << "MB\n";
        } catch (const std::exception &e) {
          std::cerr << "Warning: Failed to deserialize load data from " << load_msg.header.sender_id
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
      Message profiling_msg(stage_name, CommandType::PRINT_PROFILING, std::monostate{});
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
      Message clear_msg(stage_name, CommandType::CLEAR_PROFILING, std::monostate{});
      this->coordinator_comm_->send_message(clear_msg);
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
      Message params_request_msg(stage_name, CommandType::SEND_PARAMS, std::monostate{});
      this->coordinator_comm_->send_message(params_request_msg);
    }

    // Wait for all parameter responses
    bool received_all_params = join(CommandType::PARAMS_TRANSFER, this->num_stages_, 30);
    if (!received_all_params) {
      std::cerr << "Warning: Not all stages sent their parameters within timeout.\n";
      return false;
    }

    // Collect parameter messages
    std::vector<Message> params_messages =
        this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::PARAMS_TRANSFER);

    if (params_messages.size() != static_cast<size_t>(this->num_stages_)) {
      std::cerr << "Warning: Expected " << this->num_stages_ << " parameter messages, got "
                << params_messages.size() << ".\n";
      return false;
    }

    try {

      return true;
    } catch (const std::exception &e) {
      std::cerr << "Error updating model parameters: " << e.what() << "\n";
      return false;
    }
  }

  void request_status_from_all_stages() {
    for (const auto &stage_name : this->stage_names_) {
      Message status_msg(stage_name, CommandType::STATUS_REQUEST, std::monostate{});
      this->coordinator_comm_->send_message(status_msg);
    }
  }

  std::vector<Message> dequeue_all_messages(CommandType target_type) {
    return this->coordinator_comm_->dequeue_all_messages_by_type(target_type);
  }

  bool deploy_stages() {
    if (is_deployed_) {
      std::cout << "Stages already deployed\n";
      return true;
    }

    std::vector<std::future<bool>> connection_futures;

    for (size_t i = 0; i < remote_endpoints_.size(); ++i) {
      auto future = std::async(std::launch::async, [this, i]() {
        return this->coordinator_comm_->connect(this->stage_names_[i], remote_endpoints_[i]);
      });
      connection_futures.push_back(std::move(future));
    }

    bool all_connected = true;
    for (auto &future : connection_futures) {
      if (!future.get()) {
        all_connected = false;
      }
    }

    if (!all_connected) {
      std::cout << "Failed to connect to all endpoints\n";
      return false;
    }

    std::cout << "Connected to all endpoints, sending stage configurations...\n" << std::endl;

    std::vector<std::future<bool>> deployment_futures;

    for (size_t i = 0; i < remote_endpoints_.size(); ++i) {
      auto future = std::async(std::launch::async, [this, i]() {
        return deploy_stage_config(this->stage_names_[i], stage_configs_[i]);
      });
      deployment_futures.push_back(std::move(future));
    }

    bool all_deployed = true;
    for (auto &future : deployment_futures) {
      if (!future.get()) {
        all_deployed = false;
      }
    }

    if (!all_deployed) {
      std::cout << "Failed to deploy all stages\n";
      return false;
    }

    if (!join(CommandType::CONFIG_RECEIVED, this->num_stages_, 60)) {
      std::cerr << "Not all stages reported ready\n";
      return false;
    }

    is_deployed_ = true;
    std::cout << "All stages deployed and ready!\n";
    return true;
  }

private:
  void initialize_partitions() {
    if (!partitioner_) {
      throw std::runtime_error("Partitioner must be set before initialization");
    }
    this->partitions_ = partitioner_->get_partitions(this->model_.get_layers(), this->num_stages_);
  }

  void initialize_topology() {
    if (coordinator_endpoint_.communication_type() == "") {
      throw std::runtime_error("Coordinator endpoint is not set");
    }
    if (remote_endpoints_.size() != static_cast<size_t>(num_stages_)) {
      throw std::runtime_error("Remote endpoints size does not match number of stages");
    }
    auto splitted_model = this->model_.split(this->partitions_);

    for (size_t i = 0; i < remote_endpoints_.size(); ++i) {
      StageConfig config;
      config.stage_id = this->stage_names_[i];
      config.model_config = splitted_model[i].get_config();
      config.model_config["name"] = this->stage_names_[i];
      config.coordinator_endpoint = coordinator_endpoint_;

      if (i > 0) {
        config.prev_stage_endpoint = remote_endpoints_[i - 1];
      } else {
        config.prev_stage_endpoint = coordinator_endpoint_;
      }
      if (i < remote_endpoints_.size() - 1) {
        config.next_stage_endpoint = remote_endpoints_[i + 1];
      } else {
        config.next_stage_endpoint = coordinator_endpoint_;
      }

      stage_configs_.push_back(config);
      this->stage_names_.push_back(config.stage_id);
    }
  }

  bool deploy_stage_config(const std::string &stage_id, const StageConfig &config) {
    try {
      std::string config_json = config.to_json().dump();
      auto config_msg = Message(stage_id, CommandType::CONFIG_TRANSFER, config_json);

      this->coordinator_comm_->send_message(config_msg);

      std::cout << "Sent configuration to stage " << stage_id << '\n';

      return true;
    } catch (const std::exception &e) {
      std::cout << "Failed to deploy config to stage " << stage_id << ": " << e.what() << '\n';
      return false;
    }
  }

protected:
  // Training Parameters
  int num_stages_;
  int num_microbatches_ = 1;
  bool should_stop_ = true;

  // Components of the coordinator
  Sequential<float> model_;
  std::unique_ptr<Communicator> coordinator_comm_;
  std::unique_ptr<Loss<float>> loss_function_;
  std::unique_ptr<Partitioner<float>> partitioner_;
  Endpoint coordinator_endpoint_;

  std::atomic<bool> is_deployed_;

  // Topology information
  std::vector<std::string> stage_names_;
  std::vector<Partition> partitions_;
  std::vector<Endpoint> remote_endpoints_;
  std::vector<StageConfig> stage_configs_;
  std::thread message_thread_;

  // Message synchronization
  mutable std::mutex message_notification_mutex_;
  mutable std::condition_variable message_notification_cv_;
};

} // namespace tnn