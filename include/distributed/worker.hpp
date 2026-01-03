/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device_type.hpp"
#include "distributed/command_type.hpp"
#include "distributed/job_pool.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"

#include "communicator.hpp"
#include "job.hpp"
#include "load_tracker.hpp"
#include "message.hpp"
#include "profiling/event.hpp"
#include "profiling/profiler.hpp"
#include "stage_config.hpp"
#include "utils/hardware_info.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>

#include <string>

namespace tnn {

class Worker {
public:
  explicit Worker(std::unique_ptr<Sequential<float>> model,
                  std::unique_ptr<Communicator> communicator, const std::string &name = "")
      : model_(std::move(model)), communicator_(std::move(communicator)), id_(name),
        should_stop_(true) {}

  virtual ~Worker() { stop(); }

protected:
  Worker(bool use_gpu)
      : use_gpu_(use_gpu), model_(nullptr), communicator_(nullptr), id_(""), should_stop_(true),
        is_configured_(false) {
    if (!cpu_info_.initialize()) {
      std::cerr << "Failed to initialize CPU information" << std::endl;
    }
  }

public:
  virtual void start() {
    if (!should_stop_) {
      std::cerr << "Stage " << id_ << " is already running" << std::endl;
      return;
    }

    should_stop_ = false;

    communicator_->set_message_notification_callback([this]() {
      std::lock_guard<std::mutex> lock(message_available_mutex_);
      message_available_cv_.notify_one();
    });

    message_loop();
  }

  virtual void stop() {
    should_stop_ = true;
    message_available_cv_.notify_all();
  }

  void set_id(const std::string &id) {
    this->id_ = id;
    this->communicator_->set_id(id);
    std::cout << "Worker ID set to: " << id_ << std::endl;
  }

  void message_loop() {
    std::cout << "Running event loop" << std::endl;
    while (!should_stop_) {
      std::unique_lock<std::mutex> lock(message_available_mutex_);
      message_available_cv_.wait(
          lock, [this]() { return communicator_->has_input_message() || should_stop_; });

      if (should_stop_) {
        std::cout << "Stage " << id_ << " stopping message loop" << std::endl;
        break;
      }

      while (communicator_->has_input_message()) {
        auto message = communicator_->dequeue_input_message();
        this->process_message(std::move(message));
      }
    }
  }

  bool is_configured() const { return is_configured_; }

  std::string name() const { return id_; }

protected:
  virtual void process_message(Message &&message) {
    switch (message.header().command_type) {
    case CommandType::FORWARD_JOB: {
      auto forward_start = std::chrono::system_clock::now();
      const PooledJob<float> &forward_job = message.get<PooledJob<float>>();
      PooledJob<float> output = JobPool<float>::instance().get_job(forward_job->data.size());
      this->model_->forward(forward_job->data, output->data, forward_job->micro_batch_id);
      auto forward_end = std::chrono::system_clock::now();
      Profiler::instance().add_event(
          {EventType::COMPUTE, forward_start, forward_end, "Forward Pass", this->id_});
      output->micro_batch_id = forward_job->micro_batch_id;

      // recycle input message
      message.data() = MessageData(std::move(output));
      message.header() = MessageHeader{"next_stage", CommandType::FORWARD_JOB};
      message.header().sender_id = id_;

      communicator_->send_message(std::move(message));
    } break;
    case CommandType::BACKWARD_JOB: {
      auto backward_start = std::chrono::system_clock::now();
      const PooledJob<float> &backward_job = message.get<PooledJob<float>>();
      PooledJob<float> output = JobPool<float>::instance().get_job(backward_job->data.size());
      this->model_->backward(backward_job->data, output->data, backward_job->micro_batch_id);
      auto backward_end = std::chrono::system_clock::now();
      Profiler::instance().add_event(
          {EventType::COMPUTE, backward_start, backward_end, "Backward Pass", this->id_});

      output->micro_batch_id = backward_job->micro_batch_id;

      // recycle input message
      message.data() = MessageData(std::move(output));
      message.header() = MessageHeader{"prev_stage", CommandType::BACKWARD_JOB};
      message.header().sender_id = id_;

      communicator_->send_message(std::move(message));
    } break;
    case CommandType::UPDATE_PARAMETERS: {
      // implicitly clear grads
      this->optimizer_->update();
      this->optimizer_->clear_gradients();
      Message response("coordinator", CommandType::PARAMETERS_UPDATED, std::monostate{});
      response.header().sender_id = id_;
      communicator_->send_message(std::move(response));
    } break;
    case CommandType::TRAIN_MODE:
      this->model_->set_training(true);
      break;

    case CommandType::EVAL_MODE:
      this->model_->set_training(false);
      break;

    case CommandType::STATUS_REQUEST: {
      throw std::runtime_error("Not implemented yet");
      break;
    }

    case CommandType::ERROR_REPORT:
      if (message.has_type<std::string>()) {
        std::cout << "Stage " << id_ << " received error: " << message.get<std::string>()
                  << " from " << message.header().sender_id << std::endl;
      }
      break;
    case CommandType::START_PROFILING: {
      Profiler::instance().init_start_time(std::chrono::system_clock::now());
      Message response(message.header().sender_id, CommandType::PROFILING_STARTED,
                       std::monostate{});
      response.header().sender_id = id_;
      communicator_->send_message(std::move(response));
      break;
    }
    case CommandType::REPORT_PROFILING: {
      Message response(message.header().sender_id, CommandType::PROFILING_REPORTED,
                       Profiler::instance());
      response.header().sender_id = id_;
      communicator_->send_message(std::move(response));
      break;
    }
    case CommandType::PRINT_PROFILING:
      if (model_) {
        model_->print_profiling_summary();
        auto profile_data = communicator_->get_profile_data();
        std::cout << "Communicator profiling data:" << std::endl;
        for (const auto &[key, value] : profile_data) {
          std::cout << "  " << key << ": " << value << " us" << std::endl;
        }
        Message outgoing_message(message.header().sender_id, CommandType::PROFILING_PRINTED,
                                 std::monostate{});
        outgoing_message.header().sender_id = id_;
        communicator_->send_message(std::move(outgoing_message));
      } else {
        std::cout << "Warning: No model available to print profiling data" << std::endl;
      }
      break;
    case CommandType::CLEAR_PROFILING:
      if (model_) {
        model_->clear_profiling_data();
        Message outgoing_message(message.header().sender_id, CommandType::PROFILING_CLEARED,
                                 std::monostate{});
        outgoing_message.header().sender_id = id_;
        communicator_->send_message(std::move(outgoing_message));
      } else {
        std::cout << "Warning: No model available to clear profiling data" << std::endl;
      }
      break;
    case CommandType::CONFIG_TRANSFER:
      handle_configuration(message);
      break;
    case CommandType::LOAD_PARAMS: {
      // // decode and deserialize parameters
      throw std::runtime_error("Not implemented yet");
      break;
    }
    case CommandType::SEND_PARAMS: {
      try {

      } catch (const std::exception &e) {
        std::cerr << "Failed to send parameters: " << e.what() << std::endl;
        std::string error_text = std::string("Failed to send parameters: ") + e.what();
        Message error_msg(message.header().sender_id, CommandType::ERROR_REPORT, error_text);
        error_msg.header().sender_id = id_;
        communicator_->send_message(std::move(error_msg));
      }
      break;
    }
    case CommandType::REPORT_LOAD: {
      throw std::runtime_error("Not implemented yet");
      break;
    }
    case CommandType::HANDSHAKE: {
      // do nothing;
      break;
    }
    case CommandType::SHUTDOWN:
      std::cout << "Stage " << id_ << " received SHUTDOWN command. Stopping." << std::endl;
      this->stop();
      break;
    default:
      throw std::runtime_error("Unknown command type received");
      break;
    }
  }

  void handle_configuration(const Message &message) {
    if (!message.has_type<std::string>()) {
      std::cout << "Configuration message missing text data" << '\n';
      return;
    }

    try {
      // Parse configuration
      nlohmann::json config_json = nlohmann::json::parse(message.get<std::string>());
      StageConfig config = StageConfig::from_json(config_json);
      std::cout << "Received configuration for stage " << this->id_ << '\n';
      std::cout << config_json.dump(2) << std::endl;
      this->set_id(config.stage_id);

      // setup model, optimizer
      this->model_ = std::make_unique<Sequential<float>>(
          Sequential<float>::load_from_config(config.model_config));
      OptimizerConfig optimizer_config = OptimizerConfig::from_json(config.optimizer_config);
      this->optimizer_ = OptimizerFactory<float>::create_from_config(optimizer_config);
      if (use_gpu_) {
        this->model_->set_device(DeviceType::GPU);
      } else {
        this->model_->set_device(DeviceType::CPU);
      }
      this->model_->initialize();
      this->optimizer_->attach(this->model_->parameters(), this->model_->gradients());
      this->model_->enable_profiling(true);
      std::cout << "Created model with " << this->model_->layer_size() << " layers" << '\n';

      // setup connections
      setup_stage_connections(config);
      is_configured_ = true;
      Message ready_msg("coordinator", CommandType::CONFIG_RECEIVED, true);
      this->communicator_->send_message(std::move(ready_msg));

    } catch (const std::exception &e) {
      std::cout << "Failed to configure stage: " << e.what() << '\n';
      std::string error_text = std::string("Configuration failed: ") + e.what();
      Message error_msg("coordinator", CommandType::ERROR_REPORT, error_text);
      this->communicator_->send_message(std::move(error_msg));
    }
  }

  void setup_stage_connections(const StageConfig &config) {
    this->communicator_->connect("coordinator", config.coordinator_endpoint);
    this->communicator_->connect("next_stage", config.next_stage_endpoint);
    this->communicator_->connect("prev_stage", config.prev_stage_endpoint);
  }

  bool use_gpu_;
  std::unique_ptr<Sequential<float>> model_;
  std::unique_ptr<Optimizer<float>> optimizer_;
  std::shared_ptr<Communicator> communicator_;

  std::string id_;
  std::atomic<bool> should_stop_;
  std::atomic<bool> is_configured_;
  std::vector<StageConfig> stage_configs_;

  std::mutex message_available_mutex_;
  std::condition_variable message_available_cv_;

  HardwareInfo cpu_info_;
  LoadTracker load_tracker_;
  uint32_t update_interval = 10000;
  std::thread monitoring_thread_;
};

} // namespace tnn