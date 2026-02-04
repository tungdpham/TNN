/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "communicator.hpp"
#include "device/pool_allocator.hpp"
#include "distributed/endpoint.hpp"
#include "distributed/worker.hpp"
#include "logging/logger.hpp"
#include "nn/accuracy.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "partitioner/partitioner.hpp"
#include "profiling/event.hpp"
#include "profiling/profiler.hpp"
#include "stage_config.hpp"
#include "tensor/tensor.hpp"

namespace tnn {

enum class ParallelMode_t { DATA, PIPELINE };

struct CoordinatorConfig {
  ParallelMode_t parallel_mode = ParallelMode_t::DATA;
  std::unique_ptr<Sequential> model;
  std::unique_ptr<Optimizer> optimizer;
  std::unique_ptr<Partitioner> partitioner;
  std::unique_ptr<Worker> local_worker = nullptr;
  Endpoint coordinator_endpoint;
  std::vector<Endpoint> worker_endpoints;
};

class Coordinator {
public:
  Coordinator(CoordinatorConfig config)
      : parallel_mode_(config.parallel_mode),
        model_(std::move(config.model)),
        optimizer_(std::move(config.optimizer)),
        partitioner_(std::move(config.partitioner)),
        local_worker_(std::move(config.local_worker)),
        coordinator_endpoint_(config.coordinator_endpoint),
        worker_endpoints_(config.worker_endpoints) {
    if (local_worker_) {
      std::thread worker_thread([this]() { local_worker_->start(); });
      worker_thread.detach();
    }
  }

  virtual ~Coordinator() { comm_.reset(); }

  void initialize() {
    initialize_partitions();
    initialize_topology();
  }

  void set_partitioner(std::unique_ptr<Partitioner> partitioner) {
    partitioner_ = std::move(partitioner);
  }

  void add_message_callback() {
    this->comm_->set_callback([this]() {
      std::lock_guard<std::mutex> lock(this->message_notification_mutex_);
      this->message_notification_cv_.notify_all();
    });
  }

  void start() {}

  void stop() {
    for (const auto &worker_endpoint : this->worker_endpoints_) {
      Message stop_msg(CommandType::SHUTDOWN, std::monostate{});
      this->comm_->send_message(std::move(stop_msg), worker_endpoint);
    }
    should_stop_ = true;
    message_notification_cv_.notify_all();
    std::cout << "Stopped all pipeline stages" << std::endl;
  }

  void process_message(const Message &message) {}

  void send_message(Message &&message, const Endpoint &endpoint) {
    this->comm_->send_message(std::move(message), endpoint);
  }

  void set_training(bool training) {
    for (const auto &worker_endpoint : this->worker_endpoints_) {
      Message mode_msg(training ? CommandType::TRAIN_MODE : CommandType::EVAL_MODE);
      this->comm_->send_message(std::move(mode_msg), worker_endpoint);
    }
  }

  /**
   * @brief Forwards input batch but does not wait for the result.
   * @param input The input tensor to be processed.
   * @param microbatch_id The ID of the microbatch (0 to num_microbatches - 1).
   */
  void forward(Tensor &&input, size_t microbatch_id) {
    Job job(std::move(input), microbatch_id);
    Message forward_msg(CommandType::FORWARD_JOB, std::move(job));
    this->comm_->send_message(std::move(forward_msg), worker_endpoints_.front());
  }

  /**
   * @brief Sends the backward gradient to the last stage.
   * @param gradient The gradient tensor to be backpropagated.
   * @param microbatch_id The ID of the microbatch (0 to num_microbatches - 1).
   */
  void backward(Tensor &&gradient, size_t microbatch_id) {
    Job job(std::move(gradient), microbatch_id);
    Message backward_msg(CommandType::BACKWARD_JOB, std::move(job));
    this->comm_->send_message(std::move(backward_msg), worker_endpoints_.back());
  }

  void update_parameters() {
    for (const auto &worker_endpoint : this->worker_endpoints_) {
      Message update_msg(CommandType::UPDATE_PARAMETERS, std::monostate{});
      this->comm_->send_message(std::move(update_msg), worker_endpoint);
    }
    bool success = join(CommandType::PARAMETERS_UPDATED, this->worker_endpoints_.size(), 60);
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
          return this->comm_->message_count(type) >= expected_count;
        });

    return success;
  }

  /**
   * @brief Forwards all microbatches and immediately compute loss and backward pass as results
   * arrive.
   * @param microbatch_inputs A vector of input tensors for each microbatch.
   * @param microbatch_labels A vector of target tensors for each microbatch.
   */
  Result async_train_batch(std::vector<Tensor> &microbatch_inputs,
                           std::vector<Tensor> &microbatch_labels,
                           const std::unique_ptr<Loss> &criterion) {
    if (microbatch_inputs.size() != microbatch_labels.size()) {
      throw std::runtime_error("Mismatched number of inputs and labels in async_train_batch");
    }
    size_t num_microbatches = microbatch_inputs.size();

    for (size_t i = 0; i < num_microbatches; ++i) {
      this->forward(std::move(microbatch_inputs[i]), i);
    }

    float total_loss = 0.0f;
    int total_corrects = 0;

    size_t processed_microbatches_ = 0;
    while (processed_microbatches_ < num_microbatches) {
      std::unique_lock<std::mutex> lock(message_notification_mutex_);
      message_notification_cv_.wait(
          lock, [this]() { return this->comm_->message_count(CommandType::FORWARD_JOB) > 0; });
      std::vector<Message> FORWARD_JOBs =
          this->comm_->dequeue_all_messages_by_type(CommandType::FORWARD_JOB);

      for (auto &forward_msg : FORWARD_JOBs) {
        if (forward_msg.has_type<Job>()) {
          ++processed_microbatches_;

          Job &job = forward_msg.get<Job>();
          Tensor &predictions = job.data;
          Tensor &targets = microbatch_labels[job.mb_id];
          Tensor device_targets = targets->to_device(predictions->device());
          float loss = 0.0f;
          criterion->compute_loss(predictions, device_targets, loss);
          total_corrects += compute_class_corrects(predictions, device_targets);

          total_loss += loss;
          Tensor gradient = make_tensor(PoolAllocator::instance(predictions->device()),
                                        predictions->data_type(), predictions->shape());
          criterion->compute_gradient(predictions, device_targets, gradient);
          this->backward(std::move(gradient), job.mb_id);
        } else {
          throw std::runtime_error("Unexpected message type in FORWARD_JOB");
        }
      }
    }

    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    message_notification_cv_.wait(lock, [this, num_microbatches]() {
      return this->comm_->message_count(CommandType::BACKWARD_COMPLETE) >= num_microbatches;
    });

    this->comm_->dequeue_all_messages_by_type(CommandType::BACKWARD_COMPLETE);

    return {total_loss, static_cast<double>(total_corrects)};
  }

  /**
   * @brief Forwards all microbatches and immediately compute loss and backward pass as results
   * arrive.
   * @param microbatch_inputs A vector of input tensors for each microbatch.
   * @param microbatch_labels A vector of target tensors for each microbatch.
   */
  Result async_val_batch(std::vector<Tensor> &microbatch_inputs,
                         std::vector<Tensor> &microbatch_labels,
                         const std::unique_ptr<Loss> &criterion) {
    if (microbatch_inputs.size() != microbatch_labels.size()) {
      throw std::runtime_error("Mismatched number of inputs and labels in async_train_batch");
    }
    size_t num_microbatches = microbatch_inputs.size();
    for (size_t i = 0; i < num_microbatches; ++i) {
      this->forward(std::move(microbatch_inputs[i]), i);
    }
    float total_loss = 0.0f;
    int total_corrects = 0;

    size_t processed_microbatches_ = 0;
    while (processed_microbatches_ < num_microbatches) {
      std::unique_lock<std::mutex> lock(message_notification_mutex_);
      message_notification_cv_.wait(
          lock, [this]() { return this->comm_->message_count(CommandType::FORWARD_JOB) > 0; });
      std::vector<Message> FORWARD_JOBs =
          this->comm_->dequeue_all_messages_by_type(CommandType::FORWARD_JOB);

      for (auto &forward_msg : FORWARD_JOBs) {
        if (forward_msg.has_type<Job>()) {
          ++processed_microbatches_;

          Job &job = forward_msg.get<Job>();
          Tensor &predictions = job.data;
          Tensor &targets = microbatch_labels[job.mb_id];
          Tensor device_targets = targets->to_device(predictions->device());
          float loss = 0.0f;
          criterion->compute_loss(predictions, device_targets, loss);
          total_corrects += compute_class_corrects(predictions, device_targets);

          total_loss += loss;
        } else {
          throw std::runtime_error("Unexpected message type in FORWARD_JOB");
        }
      }
    }

    return {total_loss, static_cast<double>(total_corrects)};
  }

  /**
   * @brief Requests all stages to print their profiling data.
   */
  void print_profiling() {
    for (const auto &worker_endpoint : this->worker_endpoints_) {
      Message profiling_msg(CommandType::PRINT_PROFILING, std::monostate{});
      this->comm_->send_message(std::move(profiling_msg), worker_endpoint);
    }
    bool all_printed = join(CommandType::PROFILING_PRINTED, this->worker_endpoints_.size(), 30);
    if (!all_printed) {
      std::cerr << "Warning: Not all stages confirmed profiling print within timeout.\n";
    }
  }

  /**
   * @brief Requests all stages to start profiling.
   */
  void start_profiling() {
    GlobalProfiler::init_start_time(Clock::now());
    for (const auto &worker_endpoint : this->worker_endpoints_) {
      Message start_msg(CommandType::START_PROFILING, std::monostate{});
      this->comm_->send_message(std::move(start_msg), worker_endpoint);
    }
    bool all_started = join(CommandType::PROFILING_STARTED, this->worker_endpoints_.size(), 30);
    if (!all_started) {
      std::cerr << "Warning: Not all stages confirmed profiling start within timeout.\n";
    }
  }

  /**
   * @brief Requests all stages to report their profiling data.
   */
  void fetch_profiling() {
    for (const auto &worker_endpoint : this->worker_endpoints_) {
      Message report_msg(CommandType::REPORT_PROFILING, std::monostate{});
      this->comm_->send_message(std::move(report_msg), worker_endpoint);
    }
    bool all_reported = join(CommandType::PROFILING_REPORTED, this->worker_endpoints_.size(), 30);
    if (!all_reported) {
      std::cerr << "Warning: Not all stages reported profiling data within timeout.\n";
    }

    std::vector<Message> profiling_messages =
        this->comm_->dequeue_all_messages_by_type(CommandType::PROFILING_REPORTED);

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
    for (const auto &worker_endpoint : this->worker_endpoints_) {
      Message clear_msg(CommandType::CLEAR_PROFILING, std::monostate{});
      this->comm_->send_message(std::move(clear_msg), worker_endpoint);
    }
  }

  std::vector<Message> dequeue_all_messages(CommandType target_type) {
    return this->comm_->dequeue_all_messages_by_type(target_type);
  }

  bool deploy_stages() {
    for (const auto &endpoint : worker_endpoints_) {
      std::cout << "Expecting worker at " << endpoint.to_json().dump(4) << std::endl;
    }

    bool all_connected = true;
    for (const auto &worker_endpoint : this->worker_endpoints_) {
      all_connected &= this->comm_->connect(worker_endpoint);
    }

    if (!all_connected) {
      std::cout << "Failed to connect to all endpoints\n";
      return false;
    }

    std::cout << "Connected to all endpoints, sending stage configurations..." << std::endl;

    for (size_t i = 0; i < stage_configs_.size(); ++i) {
      deploy_stage_config(stage_configs_[i], worker_endpoints_[i]);
    }

    if (!join(CommandType::CONFIG_RECEIVED, this->worker_endpoints_.size(), 60)) {
      std::cerr << "Not all stages reported ready\n";
      throw std::runtime_error("Stage deployment failed");
    }

    return true;
  }

private:
  void initialize_partitions() {
    if (!partitioner_) {
      throw std::runtime_error("Partitioner must be set before initialization");
    }
    this->partitions_ = partitioner_->partition_model(this->model_->get_layers());
  }

  void initialize_topology() {
    auto splitted_layers = split(this->model_->get_layers(), this->partitions_);

    for (size_t i = 0; i < worker_endpoints_.size(); ++i) {
      Sequential stage_model("stage_" + std::to_string(i), std::move(splitted_layers[i]));
      StageConfig config;
      config.model_config = stage_model.get_config();
      config.optimizer_config = optimizer_->get_config();
      config.coordinator_endpoint = coordinator_endpoint_;

      if (parallel_mode_ == ParallelMode_t::DATA) {
        config.next_stage_endpoint = coordinator_endpoint_;
        config.prev_stage_endpoint = coordinator_endpoint_;

      } else if (parallel_mode_ == ParallelMode_t::PIPELINE) {
        if (i > 0) {
          config.prev_stage_endpoint = worker_endpoints_[i - 1];
        } else {
          config.prev_stage_endpoint = Endpoint::empty();
        }
        if (i < worker_endpoints_.size() - 1) {
          config.next_stage_endpoint = worker_endpoints_[i + 1];
        } else {
          config.next_stage_endpoint = coordinator_endpoint_;
        }

        // minor optimization
      }

      stage_configs_.push_back(config);
    }
  }

  bool deploy_stage_config(const StageConfig &config, const Endpoint &worker_endpoint) {
    try {
      auto config_msg = Message(CommandType::CONFIG_TRANSFER, config);
      this->comm_->send_message(std::move(config_msg), worker_endpoint);
      return true;
    } catch (const std::exception &e) {
      std::cout << "Failed to deploy config: " << e.what() << '\n';
      return false;
    }
  }

protected:
  // Components of the coordinator
  ParallelMode_t parallel_mode_ = ParallelMode_t::DATA;
  std::unique_ptr<Sequential> model_;
  std::unique_ptr<Optimizer> optimizer_;
  std::unique_ptr<Partitioner> partitioner_;
  std::unique_ptr<Worker> local_worker_;

  // Communication
  std::unique_ptr<Communicator> comm_;
  Endpoint coordinator_endpoint_;

  // Topology information
  std::vector<SeqPartition> partitions_;
  std::vector<Endpoint> worker_endpoints_;
  std::vector<StageConfig> stage_configs_;
  Logger logger_{"profiler", "logs/profiler.log", LogLevel::info};

  // Training Parameters
  bool should_stop_ = true;

  // Message synchronization
  mutable std::mutex message_notification_mutex_;
  mutable std::condition_variable message_notification_cv_;
};

}  // namespace tnn