/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device_manager.hpp"
#include "distributed/command_type.hpp"
#include "nn/optimizers.hpp"
#include "nn/sequential.hpp"

#include "communicator.hpp"
#include "job.hpp"
#include "message.hpp"
#include "profiling/event.hpp"
#include "profiling/profiler.hpp"
#include "stage_config.hpp"
#include "tensor/tensor.hpp"
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
  explicit Worker(std::unique_ptr<Sequential> model, std::unique_ptr<Communicator> communicator)
      : model_(std::move(model)), communicator_(std::move(communicator)), should_stop_(true) {}

  virtual ~Worker() { stop(); }

protected:
  Worker(bool use_gpu)
      : use_gpu_(use_gpu), model_(nullptr), communicator_(nullptr), should_stop_(true),
        is_configured_(false) {}

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
      const Job &forward_job = message.get<Job>();
      Tensor output_tensor = make_pooled_tensor_from_dtype(
          global_mem_pool(), forward_job.data->data_type(),
          this->model_->compute_output_shape(forward_job.data->shape()), model_->get_device());
      this->model_->forward(forward_job.data, output_tensor, forward_job.micro_batch_id);
      Tensor cpu_output_tensor = output_tensor->to_device(&getCPU());
      Job output(cpu_output_tensor, forward_job.micro_batch_id);
      auto forward_end = std::chrono::system_clock::now();
      GlobalProfiler::add_event(
          {EventType::COMPUTE, forward_start, forward_end, "Forward Pass", this->id_});
      message = Message(id_, "next_stage", CommandType::FORWARD_JOB, std::move(output));
      communicator_->send_message(std::move(message));
    } break;
    case CommandType::BACKWARD_JOB: {
      auto backward_start = std::chrono::system_clock::now();
      const Job &backward_job = message.get<Job>();
      Tensor output_tensor =
          make_pooled_tensor_from_dtype(global_mem_pool(), backward_job.data->data_type(),
                                        backward_job.data->shape(), model_->get_device());
      this->model_->backward(backward_job.data, output_tensor, backward_job.micro_batch_id);
      Tensor cpu_output_tensor = output_tensor->to_device(&getCPU());
      Job output(cpu_output_tensor, backward_job.micro_batch_id);
      auto backward_end = std::chrono::system_clock::now();
      GlobalProfiler::add_event(
          {EventType::COMPUTE, backward_start, backward_end, "Backward Pass", this->id_});
      message = Message(id_, "prev_stage", CommandType::BACKWARD_JOB, std::move(output));
      communicator_->send_message(std::move(message));
    } break;
    case CommandType::UPDATE_PARAMETERS: {
      auto update_start = std::chrono::system_clock::now();
      // implicitly clear grads
      this->optimizer_->update();
      this->optimizer_->clear_gradients();
      auto update_end = std::chrono::system_clock::now();
      GlobalProfiler::add_event(
          {EventType::COMPUTE, update_start, update_end, "Parameters Update", this->id_});
      Message response(id_, "coordinator", CommandType::PARAMETERS_UPDATED, std::monostate{});
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
      GlobalProfiler::init_start_time(std::chrono::system_clock::now());
      Message response(id_, message.header().sender_id, CommandType::PROFILING_STARTED);
      communicator_->send_message(std::move(response));
      break;
    }
    case CommandType::REPORT_PROFILING: {
      Message response(id_, message.header().sender_id, CommandType::PROFILING_REPORTED,
                       GlobalProfiler::get_profiler());
      communicator_->send_message(std::move(response));
      break;
    }
    case CommandType::PRINT_PROFILING:
      if (model_) {
        model_->print_profiling_info();
        Message outgoing_message(id_, message.header().sender_id, CommandType::PROFILING_PRINTED);
        communicator_->send_message(std::move(outgoing_message));
      } else {
        std::cout << "Warning: No model available to print profiling data" << std::endl;
      }
      break;
    case CommandType::CLEAR_PROFILING:
      if (model_) {
        model_->reset_profiling_info();
        Message outgoing_message(id_, message.header().sender_id, CommandType::PROFILING_CLEARED);
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
        // serialize and encode parameters
      } catch (const std::exception &e) {
        std::cerr << "Failed to send parameters: " << e.what() << std::endl;
        std::string error_text = std::string("Failed to send parameters: ") + e.what();
        Message error_msg(id_, message.header().sender_id, CommandType::ERROR_REPORT, error_text);
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
    case CommandType::HANDSHAKE_ACK: {
      // do nothing;
      break;
    }
    case CommandType::SHUTDOWN:
      std::cout << "Stage " << id_ << " received SHUTDOWN command. Stopping." << std::endl;
      this->stop();
      break;
    default:
      std::cerr << "Warning: Unknown command type "
                << static_cast<int>(message.header().command_type) << " received by stage " << id_
                << std::endl;
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
      LayerConfig model_config = LayerConfig::from_json(config.model_config);
      this->model_ = Sequential::create_from_config(model_config);
      OptimizerConfig optimizer_config = OptimizerConfig::from_json(config.optimizer_config);
      this->optimizer_ = OptimizerFactory::create_from_config(optimizer_config);
      if (use_gpu_) {
        this->model_->set_device(getGPU());
      } else {
        this->model_->set_device(getCPU());
      }
      this->model_->init();
      this->optimizer_->attach(this->model_->parameters(), this->model_->gradients());
      this->model_->enable_profiling(true);

      // setup connections
      setup_stage_connections(config);
      is_configured_ = true;
      Message ready_msg(id_, "coordinator", CommandType::CONFIG_RECEIVED, true);
      this->communicator_->send_message(std::move(ready_msg));

    } catch (const std::exception &e) {
      std::cout << "Failed to configure stage: " << e.what() << '\n';
      std::string error_text = std::string("Configuration failed: ") + e.what();
      Message error_msg(id_, "coordinator", CommandType::ERROR_REPORT, error_text);
      this->communicator_->send_message(std::move(error_msg));
    }
  }

  void setup_stage_connections(const StageConfig &config) {
    this->communicator_->connect("coordinator", config.coordinator_endpoint);
    this->communicator_->connect("next_stage", config.next_stage_endpoint);
    this->communicator_->connect("prev_stage", config.prev_stage_endpoint);
  }

  bool use_gpu_;
  std::unique_ptr<Sequential> model_;
  std::unique_ptr<Optimizer> optimizer_;
  std::shared_ptr<Communicator> communicator_;

  std::string id_;
  std::atomic<bool> should_stop_;
  std::atomic<bool> is_configured_;

  std::mutex message_available_mutex_;
  std::condition_variable message_available_cv_;
};

} // namespace tnn