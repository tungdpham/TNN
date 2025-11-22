/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device_type.hpp"
#include "nn/sequential.hpp"

#include "communicator.hpp"
#include "job.hpp"
#include "load_tracker.hpp"
#include "stage_config.hpp"
#include "utils/hardware_info.hpp"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <omp.h>
#include <string>

namespace tnn {

class PipelineStage {
public:
  explicit PipelineStage(std::unique_ptr<Sequential<float>> model,
                         std::unique_ptr<Communicator> communicator, const std::string &name = "")
      : model_(std::move(model)), communicator_(std::move(communicator)), name_(name),
        should_stop_(true) {}

  virtual ~PipelineStage() { stop(); }

protected:
  PipelineStage()
      : model_(nullptr), communicator_(nullptr), name_(""), should_stop_(true),
        is_configured_(false) {
    if (!cpu_info_.initialize()) {
      std::cerr << "Failed to initialize CPU information" << std::endl;
    }
  }

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

    message_loop();
  }

  virtual void stop() {
    should_stop_ = true;
    message_available_cv_.notify_all();
  }

  void message_loop() {
    std::cout << "Running event loop" << std::endl;
    while (!should_stop_) {
      std::unique_lock<std::mutex> lock(message_available_mutex_);
      message_available_cv_.wait(
          lock, [this]() { return communicator_->has_input_message() || should_stop_; });

      if (should_stop_) {
        std::cout << "Stage " << name_ << " stopping message loop" << std::endl;
        break;
      }

      while (communicator_->has_input_message()) {
        auto message = communicator_->dequeue_input_message();
        this->process_message(message);
      }
    }
  }

  bool is_configured() const { return is_configured_; }

  std::string get_stage_id() const { return stage_id_; }

  std::string name() const { return name_; }

protected:
  virtual void process_message(Message &message) {
    switch (message.header.command_type) {
    case CommandType::FORWARD_JOB: {
      Job<float> forward_job = message.get<Job<float>>();
      this->model_->forward(forward_job.data, forward_job.micro_batch_id); // apply in-place
      Message output_message("next_stage", CommandType::FORWARD_JOB, forward_job);
      output_message.header.sender_id = name_;

      communicator_->send_message(output_message);
    } break;
    case CommandType::BACKWARD_JOB: {
      Job<float> backward_job = message.get<Job<float>>();
      this->model_->backward(backward_job.data, backward_job.micro_batch_id);
      Message output_message("prev_stage", CommandType::BACKWARD_JOB, backward_job);
      output_message.header.sender_id = name_;

      communicator_->send_message(output_message);
    } break;
    case CommandType::UPDATE_PARAMETERS: {
      model_->update_parameters();
      Message response("coordinator", CommandType::PARAMETERS_UPDATED, std::monostate{});
      response.header.sender_id = name_;
      communicator_->send_message(response);
    } break;
    case CommandType::TRAIN_MODE:
      this->model_->set_training(true);
      break;

    case CommandType::EVAL_MODE:
      this->model_->set_training(false);
      break;

    case CommandType::STATUS_REQUEST: {
      throw new std::runtime_error("Not implemented yet");
      break;
    }

    case CommandType::ERROR_REPORT:
      if (message.has_type<std::string>()) {
        std::cout << "Stage " << name_ << " received error: " << message.get<std::string>()
                  << " from " << message.header.sender_id << std::endl;
      }
      break;
    case CommandType::PRINT_PROFILING:
      if (model_) {
        model_->print_profiling_summary();
        Message outgoing_message(message.header.sender_id, CommandType::PROFILING_PRINTED,
                                 std::monostate{});
        outgoing_message.header.sender_id = name_;
        communicator_->send_message(outgoing_message);
      } else {
        std::cout << "Warning: No model available to print profiling data" << std::endl;
      }
      break;
    case CommandType::CLEAR_PROFILING:
      if (model_) {
        model_->clear_profiling_data();
        Message outgoing_message(message.header.sender_id, CommandType::PROFILING_CLEARED,
                                 std::monostate{});
        outgoing_message.header.sender_id = name_;
        communicator_->send_message(outgoing_message);
      } else {
        std::cout << "Warning: No model available to clear profiling data" << std::endl;
      }
      break;
    case CommandType::CONFIG_TRANSFER:
      handle_configuration(message);
      break;
    case CommandType::LOAD_PARAMS: {
      // // decode and deserialize parameters
      throw new std::runtime_error("Not implemented yet");
      break;
    }
    case CommandType::SEND_PARAMS: {
      try {

      } catch (const std::exception &e) {
        std::cerr << "Failed to send parameters: " << e.what() << std::endl;
        std::string error_text = std::string("Failed to send parameters: ") + e.what();
        Message error_msg(message.header.sender_id, CommandType::ERROR_REPORT, error_text);
        error_msg.header.sender_id = name_;
        communicator_->send_message(error_msg);
      }
      break;
    }
    case CommandType::UPDATE_LOAD: {
      // update load tracker info
      update_load_tracker();
      break;
    }
    case CommandType::REPORT_LOAD: {
      throw std::runtime_error("Not implemented yet");
      break;
    }
    case CommandType::SHUTDOWN:
      std::cout << "Stage " << name_ << " received SHUTDOWN command. Stopping." << std::endl;
      this->stop();
      break;
    default:
      throw std::runtime_error("Unknown command type received");
      break;
    }
  }

  void update_load_tracker() {
    if (!model_) {
      load_tracker_.avg_forward_time_ = 0;
      load_tracker_.avg_backward_time_ = 0;
    } else {
      const std::map<std::string, uint64_t> forward_times = model_->get_forward_times();
      const std::map<std::string, uint64_t> backward_times = model_->get_backward_times();

      uint64_t cummulative_forward_time = std::accumulate(
          forward_times.begin(), forward_times.end(), 0LL,
          [](uint64_t sum, const std::pair<std::string, uint64_t> &p) { return sum + p.second; });

      uint64_t cummulative_backward_time = std::accumulate(
          backward_times.begin(), backward_times.end(), 0LL,
          [](uint64_t sum, const std::pair<std::string, uint64_t> &p) { return sum + p.second; });

      load_tracker_.avg_forward_time_ =
          static_cast<float>(static_cast<double>(cummulative_forward_time) / 1000.0);
      load_tracker_.avg_backward_time_ =
          static_cast<float>(static_cast<double>(cummulative_backward_time) / 1000.0);
    }

    if (cpu_info_.update_dynamic_info()) {
      load_tracker_.avg_cpu_utilization_ = static_cast<float>(cpu_info_.get_overall_utilization());
      load_tracker_.max_memory_usage_ =
          static_cast<float>(cpu_info_.get_ram_info().used_memory_bytes / (1024 * 1024));
    } else {
      load_tracker_.avg_cpu_utilization_ = -1.0f;
      load_tracker_.max_memory_usage_ = -1.0f;
    }
  }

  void handle_configuration(Message &message) {
    if (!message.has_type<std::string>()) {
      std::cout << "Configuration message missing text data" << '\n';
      return;
    }

    try {
      // Parse configuration
      nlohmann::json config_json = nlohmann::json::parse(message.get<std::string>());

      StageConfig config = StageConfig::from_json(config_json);

      stage_id_ = config.stage_id;
      std::cout << "Received configuration for stage " << stage_id_ << '\n';
      this->model_ = std::make_unique<Sequential<float>>(
          Sequential<float>::load_from_config(config.model_config));

      this->model_->set_device(DeviceType::GPU);

      this->model_->initialize();

      this->model_->enable_profiling(true);

      std::cout << "Created model with " << this->model_->layer_size() << " layers" << '\n';

      setup_stage_connections(config);

      name_ = stage_id_;

      is_configured_ = true;

      Message ready_msg("coordinator", CommandType::CONFIG_RECEIVED, true);
      this->communicator_->send_message(ready_msg);
    } catch (const std::exception &e) {
      std::cout << "Failed to configure stage: " << e.what() << '\n';

      std::string error_text = std::string("Configuration failed: ") + e.what();
      Message error_msg("coordinator", CommandType::ERROR_REPORT, error_text);
      this->communicator_->send_message(error_msg);
    }
  }

  void setup_stage_connections(const StageConfig &config) {
    this->communicator_->connect("coordinator", config.coordinator_endpoint);
    this->communicator_->connect("next_stage", config.next_stage_endpoint);
    this->communicator_->connect("prev_stage", config.prev_stage_endpoint);
  }

  std::unique_ptr<Sequential<float>> model_;
  std::shared_ptr<Communicator> communicator_;
  std::string name_;
  std::atomic<bool> should_stop_;
  std::atomic<bool> is_configured_;
  std::string stage_id_;
  std::vector<StageConfig> stage_configs_;

  std::mutex message_available_mutex_;
  std::condition_variable message_available_cv_;

  HardwareInfo cpu_info_;
  LoadTracker load_tracker_;
  uint32_t update_interval = 10000;
  std::thread monitoring_thread_;
};

} // namespace tnn