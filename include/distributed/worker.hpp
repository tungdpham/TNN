/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <variant>

#include "communicator.hpp"
#include "device/device_manager.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "distributed/command_type.hpp"
#include "job.hpp"
#include "message.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/graph.hpp"
#include "nn/layers.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "profiling/profiler.hpp"
#include "stage_config.hpp"
#include "type/type.hpp"

namespace tunx {

class Worker {
public:
  explicit Worker(Sequential model, std::unique_ptr<Communicator> communicator,
                  const Device &device = getHost())
      : communicator_(std::move(communicator)),
        should_stop_(true) {
    graph_ = std::make_unique<Graph>();
    auto input = graph_->make_node("input");
    auto output = model(input);

    auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);
    graph_->compile(allocator);
  }

  virtual ~Worker() { stop(); }

protected:
  Worker(bool use_gpu)
      : use_gpu_(use_gpu),
        should_stop_(true),
        is_configured_(false) {}

public:
  void start() {
    if (!should_stop_) {
      std::cerr << "Stage " << id_ << " is already running" << std::endl;
      return;
    }
    should_stop_ = false;
    communicator_->set_callback([this]() {
      std::lock_guard<std::mutex> lock(message_available_mutex_);
      message_available_cv_.notify_one();
    });
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

  void stop() {
    should_stop_ = true;
    message_available_cv_.notify_all();
  }

  bool is_configured() const { return is_configured_; }

  void set_config(const StageConfig &config) {
    auto &device = use_gpu_ ? getGPU() : getHost();
    auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);
    std::istringstream graph_stream(config.graph_state, std::ios::binary);
    graph_ = std::make_unique<Graph>(Graph::load_state(graph_stream, allocator));
    for (const Edge &edge : graph_->edges()) {
      std::cout << "Input nodes: ";
      for (const auto &producer : edge->producers()) {
        std::cout << producer->uid() << " ";
      }
      std::cout << "| Layer: " << edge->layer()->name() << " | Output nodes: ";
      for (const auto &consumer : edge->consumers()) {
        std::cout << consumer->uid() << " ";
      }
      std::cout << std::endl;
    }
    input_uids_ = config.input_uids;
    output_uids_ = config.output_uids;

    this->optimizer_ = OptimizerFactory::create_from_config(config.optimizer_config);
    this->optimizer_->attach(*graph_);

    this->scheduler_ =
        SchedulerFactory::create_from_config(config.scheduler_config, this->optimizer_.get());

    // setup connections
    coordinator_endpoint_ = config.coordinator_endpoint;
    next_stage_endpoint_ = config.next_stage_endpoint;
    prev_stage_endpoint_ = config.prev_stage_endpoint;

    if (coordinator_endpoint_) this->communicator_->connect(coordinator_endpoint_);
    if (next_stage_endpoint_) this->communicator_->connect(next_stage_endpoint_);
    if (prev_stage_endpoint_) this->communicator_->connect(prev_stage_endpoint_);

    is_configured_ = true;
  }

  void set_next_stage_endpoint(const Endpoint &endpoint) { next_stage_endpoint_ = endpoint; }

  void set_prev_stage_endpoint(const Endpoint &endpoint) { prev_stage_endpoint_ = endpoint; }

  void set_coordinator_endpoint(const Endpoint &endpoint) { coordinator_endpoint_ = endpoint; }

  Endpoint endpoint() const { return communicator_->endpoint(); }

  Communicator *get_communicator() const { return communicator_.get(); }

protected:
  void process_message(Message &&message) {
    switch (message.header().command_type) {
      case CommandType::FORWARD_JOB: {
        const Job &forward_job = message.get<Job>();
        TensorBundle inputs = forward_job.data;
        auto outputs = graph_->forward(inputs, forward_job.pid);
        Job output_job(outputs, forward_job.pid);
        message = Message(CommandType::FORWARD_JOB, std::move(output_job));
        communicator_->send_message(std::move(message), next_stage_endpoint_);
      } break;
      case CommandType::BACKWARD_JOB: {
        const Job &backward_job = message.get<Job>();
        TensorBundle output_grads = backward_job.data;
        auto outputs = graph_->backward(output_grads, backward_job.pid);
        if (prev_stage_endpoint_ == Endpoint::empty()) {
          // only send backward complete if there is no previous stage
          Message complete_msg(CommandType::BACKWARD_COMPLETE);
          communicator_->send_message(std::move(complete_msg), coordinator_endpoint_);
          break;
        }
        Job output_job(outputs, backward_job.pid);
        message = Message(CommandType::BACKWARD_JOB, std::move(output_job));
        communicator_->send_message(std::move(message), prev_stage_endpoint_);
      } break;
      case CommandType::UPDATE_PARAMETERS: {
        update_count_++;

        if (scheduler_) {
          this->scheduler_->step();
        }
        this->optimizer_->update();
        this->optimizer_->zero_grads();
        this->graph_->device().getFlow(defaultFlowHandle)->synchronize();

        Message response(CommandType::PARAMETERS_UPDATED, std::monostate{});
        communicator_->send_message(std::move(response), coordinator_endpoint_);
      } break;
      case CommandType::TRAIN_MODE:
        std::cout << fmt::format("Stage {}: Switching to TRAIN mode.", id_) << std::endl;
        this->graph_->set_mode(ExecutionMode::TRAIN);
        communicator_->send_message(Message(CommandType::MODE_CHANGED), coordinator_endpoint_);
        break;
      case CommandType::EVAL_MODE:
        std::cout << fmt::format("Stage {}: Switching to EVAL mode.", id_) << std::endl;
        this->graph_->set_mode(ExecutionMode::EVAL);
        communicator_->send_message(Message(CommandType::MODE_CHANGED), coordinator_endpoint_);
        break;
      case CommandType::STATUS_REQUEST: {
        throw std::runtime_error("Not implemented yet");
        break;
      }
      case CommandType::PRINT_LR: {
        if (scheduler_) {
          float lr = scheduler_->get_lr();
          std::cout << fmt::format("Stage {}: Current learning rate: {:.6e}", id_, lr) << std::endl;
          Message response(CommandType::LR_PRINTED, std::monostate{});
          communicator_->send_message(std::move(response), coordinator_endpoint_);
        } else {
          std::cout << "Warning: No scheduler available to get learning rate" << std::endl;
        }
        break;
      }
      case CommandType::ERROR_REPORT:
        if (message.has_type<std::string>()) {
          std::cout << "Stage " << id_ << " received error: " << message.get<std::string>()
                    << std::endl;
        }
        break;
      case CommandType::START_PROFILING: {
        GlobalProfiler::init_start_time(std::chrono::system_clock::now());
        Message response(CommandType::PROFILING_STARTED);
        communicator_->send_message(std::move(response), coordinator_endpoint_);
        break;
      }
      case CommandType::REPORT_PROFILING: {
        Message response(CommandType::PROFILING_REPORTED, GlobalProfiler::get_profiler());
        communicator_->send_message(std::move(response), coordinator_endpoint_);
        break;
      }
      case CommandType::PRINT_PROFILING:
        if (graph_) {
          Message outgoing_message(CommandType::PROFILING_PRINTED);
          communicator_->send_message(std::move(outgoing_message), coordinator_endpoint_);
        } else {
          std::cout << "Warning: No graph available to print profiling data" << std::endl;
        }
        break;
      case CommandType::CLEAR_PROFILING:
        if (graph_) {
          Message outgoing_message(CommandType::PROFILING_CLEARED);
          communicator_->send_message(std::move(outgoing_message), coordinator_endpoint_);
        } else {
          std::cout << "Warning: No graph available to clear profiling data" << std::endl;
        }
        break;
      case CommandType::CONFIG_TRANSFER: {
        handle_configuration(message);
        Message ready_msg(CommandType::CONFIG_RECEIVED, true);
        this->communicator_->send_message(std::move(ready_msg), coordinator_endpoint_);
        break;
      }
      case CommandType::LOAD_PARAMS: {
        // decode and deserialize parameters
        throw std::runtime_error("Not implemented yet");
        break;
      }
      case CommandType::SEND_PARAMS: {
        try {
          // serialize and encode parameters
        } catch (const std::exception &e) {
          std::cerr << "Failed to send parameters: " << e.what() << std::endl;
          std::string error_text = std::string("Failed to send parameters: ") + e.what();
          Message error_msg(CommandType::ERROR_REPORT, error_text);
          communicator_->send_message(std::move(error_msg), coordinator_endpoint_);
        }
        break;
      }
      case CommandType::REPORT_LOAD: {
        throw std::runtime_error("Not implemented yet");
        break;
      }
      case CommandType::PRINT_LOGS: {
        Message response(CommandType::LOGS_PRINTED);
        communicator_->send_message(std::move(response), coordinator_endpoint_);
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
      case CommandType::SAVE_TO_FILE: {
        try {
          const std::string &filepath = message.get<std::string>();
          std::ofstream file(filepath, std::ios::binary);
          if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
          }
          this->graph_->save_state(file);
          file.close();
          std::cout << "Model saved to " << filepath << std::endl;
        } catch (const std::exception &e) {
          std::cerr << "Failed to save model: " << e.what() << std::endl;
          std::string error_text = std::string("Failed to save model: ") + e.what();
          Message error_msg(CommandType::ERROR_REPORT, error_text);
          communicator_->send_message(std::move(error_msg), coordinator_endpoint_);
        }
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
    try {
      // Parse configuration
      StageConfig config = message.get<StageConfig>();
      set_config(config);
    } catch (const std::exception &e) {
      std::cout << "Failed to configure stage: " << e.what() << '\n';
      std::string error_text = std::string("Configuration failed: ") + e.what();
      Message error_msg(CommandType::ERROR_REPORT, error_text);
      this->communicator_->send_message(std::move(error_msg), coordinator_endpoint_);
    }
  }

  bool use_gpu_;
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<Optimizer> optimizer_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<Communicator> communicator_;

  std::string id_;
  int forward_step_ = 0;
  int backward_step_ = 0;
  size_t update_count_ = 0;
  Endpoint coordinator_endpoint_;
  Endpoint next_stage_endpoint_;
  Endpoint prev_stage_endpoint_;
  Vec<std::string> input_uids_;
  Vec<std::string> output_uids_;
  std::atomic<bool> should_stop_;
  std::atomic<bool> is_configured_;

  std::mutex message_available_mutex_;
  std::condition_variable message_available_cv_;
};

}  // namespace tunx