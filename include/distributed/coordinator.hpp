/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/mem_pool.hpp"
#include "logging/logger.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"

#include "communicator.hpp"
#include "partitioner/partitioner.hpp"
#include "profiling/event.hpp"
#include "profiling/profiler.hpp"
#include "stage_config.hpp"
#include "tensor/tensor.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace tnn {

class Coordinator {
public:
  Coordinator(std::unique_ptr<Sequential> model, std::unique_ptr<Optimizer> optimizer)
      : model_(std::move(model)), optimizer_(std::move(optimizer)) {}

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

  void set_partitioner(std::unique_ptr<Partitioner> partitioner) {
    partitioner_ = std::move(partitioner);
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
      Message start_msg("coordinator", stage_name, CommandType::TRAIN_MODE, std::monostate{});
      this->coordinator_comm_->send_message(std::move(start_msg));
    }
    std::cout << "Started all " << this->num_stages_ << " pipeline stages" << std::endl;
  }

  void stop() {
    for (const auto &stage_name : this->stage_names_) {
      std::cout << "Stopping stage " << stage_name << std::endl;
      Message stop_msg("coordinator", stage_name, CommandType::SHUTDOWN, std::monostate{});
      this->coordinator_comm_->send_message(std::move(stop_msg));
    }
    should_stop_ = true;
    message_notification_cv_.notify_all();
    if (message_thread_.joinable()) {
      message_thread_.join();
    }
    std::cout << "Stopped all pipeline stages" << std::endl;
  }

  void process_message(const Message &message) {}

  const std::vector<std::string> &stage_names() const { return stage_names_; }

  void send_message(Message &&message) {
    this->coordinator_comm_->send_message(std::move(message));
  }

  void set_training(bool training) {
    for (const auto &stage_name : this->stage_names_) {
      Message mode_msg("coordinator", stage_name,
                       training ? CommandType::TRAIN_MODE : CommandType::EVAL_MODE,
                       std::monostate{});
      this->coordinator_comm_->send_message(std::move(mode_msg));
    }
  }

  /**
   * @brief Forwards input batch but does not wait for the result.
   * @param input The input tensor to be processed.
   * @param microbatch_id The ID of the microbatch (0 to num_microbatches - 1).
   */
  void forward(Tensor &&input, size_t microbatch_id) {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &first_stage = this->stage_names_.front();

    Job job;
    job.mb_id = microbatch_id;
    job.data = std::move(input);
    Message forward_msg("coordinator", first_stage, CommandType::FORWARD_JOB, std::move(job));

    this->coordinator_comm_->send_message(std::move(forward_msg));
  }

  /**
   * @brief Sends the backward gradient to the last stage.
   * @param gradient The gradient tensor to be backpropagated.
   * @param microbatch_id The ID of the microbatch (0 to num_microbatches - 1).
   */
  void backward(Tensor &&gradient, size_t microbatch_id) {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &last_stage = this->stage_names_.back();

    Job job;
    job.mb_id = microbatch_id;
    job.data = std::move(gradient);
    Message backward_msg("coordinator", last_stage, CommandType::BACKWARD_JOB, std::move(job));

    this->coordinator_comm_->send_message(std::move(backward_msg));
  }

  void update_parameters() {
    for (const auto &stage_name : this->stage_names_) {
      Message update_msg("coordinator", stage_name, CommandType::UPDATE_PARAMETERS,
                         std::monostate{});
      this->coordinator_comm_->send_message(std::move(update_msg));
    }

    bool success = join(CommandType::PARAMETERS_UPDATED, this->num_stages_, 60);
    if (!success) {
      std::cerr << "Warning: Timeout waiting for parameter update confirmations from all stages\n";
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

    auto timeout = Time::steady_clock::now() + Time::seconds(timeout_duration);

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
  float async_process_batch(std::vector<Tensor> &microbatch_inputs,
                            std::vector<Tensor> &microbatch_labels,
                            const std::unique_ptr<Loss> &criterion) {
    if (microbatch_inputs.size() != static_cast<size_t>(this->num_microbatches_) ||
        microbatch_labels.size() != static_cast<size_t>(this->num_microbatches_)) {
      throw std::invalid_argument("Microbatch size mismatch with coordinator configuration");
    }

    for (int i = 0; i < this->num_microbatches_; ++i) {
      this->forward(std::move(microbatch_inputs[i]), i);
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
        if (forward_msg.has_type<Job>()) {
          ++processed_microbatches_;

          Job &job = forward_msg.get<Job>();
          Tensor &predictions = job.data;
          Tensor &targets = microbatch_labels[job.mb_id];
          float loss = 0.0f;
          criterion->compute_loss(predictions, targets, loss);
          total_loss += loss;
          Tensor gradient = Tensor::create_pooled(MemPool::instance(getCPU()),
                                                  predictions->data_type(), predictions->shape());
          criterion->compute_gradient(predictions, targets, gradient);
          this->backward(std::move(gradient), job.mb_id);
        }
      }
    }

    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    message_notification_cv_.wait(lock, [this]() {
      return this->coordinator_comm_->message_count(CommandType::BACKWARD_JOB) >=
             static_cast<size_t>(this->num_microbatches_);
    });

    this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::BACKWARD_JOB);

    return (this->num_microbatches_ > 0)
               ? (total_loss / static_cast<float>(this->num_microbatches_))
               : total_loss;
  }

  /**
   * @brief Requests all stages to print their profiling data.
   */
  void print_profiling() {
    for (const auto &stage_name : this->stage_names_) {
      Message profiling_msg("coordinator", stage_name, CommandType::PRINT_PROFILING,
                            std::monostate{});
      this->coordinator_comm_->send_message(std::move(profiling_msg));
    }
    bool all_printed = join(CommandType::PROFILING_PRINTED, this->num_stages_, 30);
    if (!all_printed) {
      std::cerr << "Warning: Not all stages confirmed profiling print within timeout.\n";
    }
  }

  /**
   * @brief Requests all stages to start profiling.
   */
  void start_profiling() {
    GlobalProfiler::init_start_time(Clock::now());
    for (const auto &stage_name : this->stage_names_) {
      Message start_msg("coordinator", stage_name, CommandType::START_PROFILING, std::monostate{});
      this->coordinator_comm_->send_message(std::move(start_msg));
    }
    bool all_started = join(CommandType::PROFILING_STARTED, this->num_stages_, 30);
    if (!all_started) {
      std::cerr << "Warning: Not all stages confirmed profiling start within timeout.\n";
    }
  }

  /**
   * @brief Requests all stages to report their profiling data.
   */
  void fetch_profiling() {
    for (const auto &stage_name : this->stage_names_) {
      Message report_msg("coordinator", stage_name, CommandType::REPORT_PROFILING,
                         std::monostate{});
      this->coordinator_comm_->send_message(std::move(report_msg));
    }
    bool all_reported = join(CommandType::PROFILING_REPORTED, this->num_stages_, 30);
    if (!all_reported) {
      std::cerr << "Warning: Not all stages reported profiling data within timeout.\n";
    }

    std::vector<Message> profiling_messages =
        this->coordinator_comm_->dequeue_all_messages_by_type(CommandType::PROFILING_REPORTED);

    auto &aggregator = GlobalProfiler::get_profiler();
    for (const auto &msg : profiling_messages) {
      if (msg.has_type<Profiler>()) {
        const Profiler &profiler = msg.get<Profiler>();
        aggregator.merge(profiler);
      }
    }

    aggregator.sort();

    for (auto &event : aggregator.get_events()) {
      Clock::time_point start_time = event.start_time;
      Clock::time_point end_time = event.end_time;
      Clock::duration offset = aggregator.start_time().time_since_epoch();
      start_time -= offset;
      end_time -= offset;

      double start_ms =
          Time::duration_cast<Time::microseconds>(start_time.time_since_epoch()).count() / 1000.0;
      double end_ms =
          Time::duration_cast<Time::microseconds>(end_time.time_since_epoch()).count() / 1000.0;
      double duration_ms =
          Time::duration_cast<Time::microseconds>(end_time - start_time).count() / 1000.0;

      if (start_ms < 0 || end_ms < 0 || duration_ms < 0) {
        continue;
      }

      logger_.info(
          "Event: {}, Source: {}, Type: {}, Start: {:.2f} ms, End: {:.2f} ms, Duration: {:.2f} ms",
          event.name, event.source, event_type_to_string(event.type), start_ms, end_ms,
          duration_ms);
    }
  }

  /**
   * @brief Requests all stages to clear their profiling data.
   */
  void clear_profiling() {
    for (const auto &stage_name : this->stage_names_) {
      Message clear_msg("coordinator", stage_name, CommandType::CLEAR_PROFILING, std::monostate{});
      this->coordinator_comm_->send_message(std::move(clear_msg));
    }
  }

  /**
   * @brief Collects current parameters from all stages to ensure coordinator has up-to-date
   * weights.
   * @return true if all parameters were collected successfully, false otherwise
   */
  bool collect_params() {
    std::cout << "Collecting current parameters from all stages...\n";

    // Request parameters from all stages
    for (const auto &stage_name : this->stage_names_) {
      Message params_request_msg("coordinator", stage_name, CommandType::SEND_PARAMS,
                                 std::monostate{});
      this->coordinator_comm_->send_message(std::move(params_request_msg));
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
      throw new std::runtime_error("Stage deployment failed");
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
    this->partitions_ = partitioner_->get_partitions(this->model_->get_layers());
  }

  void initialize_topology() {
    if (coordinator_endpoint_.communication_type() == "") {
      throw std::runtime_error("Coordinator endpoint is not set");
    }
    if (remote_endpoints_.size() != static_cast<size_t>(num_stages_)) {
      throw std::runtime_error("Remote endpoints size does not match number of stages");
    }
    auto splitted_model = this->model_->split(this->partitions_);

    for (size_t i = 0; i < remote_endpoints_.size(); ++i) {
      StageConfig config;
      config.stage_id = this->stage_names_[i];
      config.model_config = splitted_model[i].get_config().to_json();
      config.optimizer_config = optimizer_->get_config().to_json();
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
    }
  }

  bool deploy_stage_config(const std::string &stage_id, const StageConfig &config) {
    try {
      std::string config_json = config.to_json().dump();
      auto config_msg = Message("coordinator", stage_id, CommandType::CONFIG_TRANSFER, config_json);
      this->coordinator_comm_->send_message(std::move(config_msg));
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
  std::unique_ptr<Sequential> model_;
  std::unique_ptr<Optimizer> optimizer_;
  std::unique_ptr<Communicator> coordinator_comm_;
  std::unique_ptr<Partitioner> partitioner_;
  Endpoint coordinator_endpoint_;

  std::atomic<bool> is_deployed_ = false;

  // Topology information
  std::vector<std::string> stage_names_;
  std::vector<Partition> partitions_;
  std::vector<Endpoint> remote_endpoints_;
  std::vector<StageConfig> stage_configs_;
  std::thread message_thread_;
  Logger logger_{"profiler", "logs/profiler.log", LogLevel::info};

  // Message synchronization
  mutable std::mutex message_notification_mutex_;
  mutable std::condition_variable message_notification_cv_;
};

} // namespace tnn