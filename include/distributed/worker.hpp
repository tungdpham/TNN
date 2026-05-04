/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "communicator.hpp"
#include "device/device_manager.hpp"
#include "device/flow.hpp"
#include "device/pool_allocator.hpp"
#include "distributed/command_type.hpp"
#include "job.hpp"
#include "message.hpp"
#include "nn/blocks_impl/sequential.hpp"
#include "nn/graph.hpp"
#include "nn/graph_builder.hpp"
#include "nn/layers.hpp"
#include "nn/op_node.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "profiling/profiler.hpp"
#include "stage_config.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

namespace tnn {

class Worker {
public:
  explicit Worker(std::unique_ptr<Sequential> model, std::unique_ptr<Communicator> communicator)
      : communicator_(std::move(communicator)),
        should_stop_(true) {
    GraphBuilder builder;
    auto &node = builder.add_layer(std::move(model));
    model_ = &node;
    // Assume GPU for distributed worker for now
    auto &allocator = PoolAllocator::instance(
        DeviceManager::getInstance().getDevice(DeviceType::GPU), defaultFlowHandle);
    graph_ = std::make_unique<Graph>(builder.compile(allocator));
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
    // setup model, optimizer, criterion
    LayerConfig model_config = config.model_config;
    auto &device = use_gpu_ ? getGPU() : getHost();
    auto &allocator = PoolAllocator::instance(device, defaultFlowHandle);
    // this->graph_ = std::make_unique<Graph>();
    GraphBuilder builder;
    auto &node = builder.add_layer(Sequential::create_from_config(model_config));
    this->model_ = &node;
    // this->model_->set_seed(123456);
    this->graph_ = std::make_unique<Graph>(builder.compile(allocator));

    auto parsed_config = this->model_->get_config();
    std::cout << parsed_config.to_json().dump(4) << std::endl;

    OptimizerConfig optimizer_config = config.optimizer_config;
    this->optimizer_ = OptimizerFactory::create_from_config(optimizer_config);
    this->optimizer_->attach(this->graph_->context());

    this->scheduler_ =
        SchedulerFactory::create_from_config(config.scheduler_config, this->optimizer_.get());

    std::cout << "[TNNDBG][worker_config]"
              << " stage=" << id_
              << " optimizer=" << this->optimizer_->name()
              << " params=" << this->optimizer_->debug_num_parameters()
              << " init_lr=" << std::setprecision(10) << this->optimizer_->get_learning_rate()
              << " scheduler=" << (this->scheduler_ ? this->scheduler_->name() : std::string("none"))
              << " sched_step=" << (this->scheduler_ ? this->scheduler_->get_current_step() : 0)
              << std::endl;

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
  virtual void process_message(Message &&message) {
    switch (message.header().command_type) {
      case CommandType::FORWARD_JOB: {
        const Job &forward_job = message.get<Job>();
        auto outputs = this->model_->forward({forward_job.data}, forward_job.mb_id);
        Job output(outputs[0], forward_job.mb_id);
        message = Message(CommandType::FORWARD_JOB, std::move(output));
        communicator_->send_message(std::move(message), next_stage_endpoint_);
      } break;
      case CommandType::BACKWARD_JOB: {
        const Job &backward_job = message.get<Job>();
        auto outputs = this->model_->backward({backward_job.data}, backward_job.mb_id);
        if (prev_stage_endpoint_ == Endpoint::empty()) {
          // only send backward complete if there is no previous stage
          Message complete_msg(CommandType::BACKWARD_COMPLETE);
          communicator_->send_message(std::move(complete_msg), coordinator_endpoint_);
          break;
        }
        Job output(outputs[0], backward_job.mb_id);
        message = Message(CommandType::BACKWARD_JOB, std::move(output));
        communicator_->send_message(std::move(message), prev_stage_endpoint_);
      } break;
      case CommandType::UPDATE_PARAMETERS: {
        update_count_++;

        const char *dbg_env = std::getenv("TNN_DEBUG_LR");
        const bool debug_lr = dbg_env && std::string(dbg_env) != "0";
        int debug_interval = 100;
        if (const char *interval_env = std::getenv("TNN_DEBUG_INTERVAL")) {
          debug_interval = std::max(1, std::atoi(interval_env));
        }

        const char *stats_env = std::getenv("TNN_DEBUG_STATS");
        const bool debug_stats = stats_env && std::string(stats_env) != "0";

        const bool should_print_debug =
            debug_lr && (update_count_ <= 5 || update_count_ % debug_interval == 0);

        const float lr_before = this->optimizer_->get_learning_rate();
        const size_t sched_step_before = scheduler_ ? scheduler_->get_current_step() : 0;

        // Match PyTorch-style semantics more closely: compute the LR for this
        // optimizer update before applying the parameter update. This avoids
        // the first update using lr=0 when WarmupCosineAnnealing starts from 0.
        if (scheduler_) {
          this->scheduler_->step();
        }

        const float lr_after = this->optimizer_->get_learning_rate();
        const size_t sched_step_after = scheduler_ ? scheduler_->get_current_step() : 0;

        double grad_abs_mean = 0.0;
        double param_abs_mean_before = 0.0;
        if (should_print_debug && debug_stats) {
          grad_abs_mean = this->optimizer_->debug_abs_mean_grad();
          param_abs_mean_before = this->optimizer_->debug_abs_mean_param();
        }

        // implicitly clear grads
        this->optimizer_->update();

        double param_abs_mean_after = 0.0;
        if (should_print_debug && debug_stats) {
          param_abs_mean_after = this->optimizer_->debug_abs_mean_param();
        }

        this->optimizer_->zero_grads();

        if (should_print_debug) {
          std::cout << "[TNNDBG][worker_update]"
                    << " stage=" << id_
                    << " update=" << update_count_
                    << " optimizer=" << this->optimizer_->name()
                    << " scheduler=" << (scheduler_ ? scheduler_->name() : std::string("none"))
                    << " sched_step=" << sched_step_before << "->" << sched_step_after
                    << " lr=" << std::setprecision(10) << lr_before << "->" << lr_after;
          if (debug_stats) {
            std::cout << " grad_abs_mean=" << std::setprecision(8) << grad_abs_mean
                      << " param_abs_mean=" << param_abs_mean_before << "->" << param_abs_mean_after
                      << " param_abs_delta=" << std::abs(param_abs_mean_after - param_abs_mean_before);
          }
          std::cout << std::endl;
        }

        Message response(CommandType::PARAMETERS_UPDATED, std::monostate{});
        communicator_->send_message(std::move(response), coordinator_endpoint_);
      } break;
      case CommandType::TRAIN_MODE:
        std::cout << "Stage " << id_ << " switching to TRAIN mode." << std::endl;
        this->model_->set_training(true);
        break;
      case CommandType::EVAL_MODE:
        std::cout << "Stage " << id_ << " switching to EVAL mode." << std::endl;
        this->model_->set_training(false);
        break;
      case CommandType::STATUS_REQUEST: {
        throw std::runtime_error("Not implemented yet");
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
        if (model_) {
          Message outgoing_message(CommandType::PROFILING_PRINTED);
          communicator_->send_message(std::move(outgoing_message), coordinator_endpoint_);
        } else {
          std::cout << "Warning: No model available to print profiling data" << std::endl;
        }
        break;
      case CommandType::CLEAR_PROFILING:
        if (model_) {
          Message outgoing_message(CommandType::PROFILING_CLEARED);
          communicator_->send_message(std::move(outgoing_message), coordinator_endpoint_);
        } else {
          std::cout << "Warning: No model available to clear profiling data" << std::endl;
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
          this->model_->save_state(file);
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
  OpNode *model_;
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
  std::atomic<bool> should_stop_;
  std::atomic<bool> is_configured_;

  std::mutex message_available_mutex_;
  std::condition_variable message_available_cv_;
};

}  // namespace tnn