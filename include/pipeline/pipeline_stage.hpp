/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "nn/sequential.hpp"

#include "communicator.hpp"
#include "load_tracker.hpp"
#include "old_binary_serializer.hpp"
#include "task.hpp"
#include "utils/hardware_info.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <omp.h>
#include <queue>
#include <string>

namespace tpipeline {

template <typename T = float> class PipelineStage {
public:
  explicit PipelineStage(
      std::unique_ptr<tnn::Sequential<T>> model,
      std::unique_ptr<PipelineCommunicator<T>, std::function<void(PipelineCommunicator<T> *)>>
          communicator,
      const std::string &name = "")
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

    monitoring_thread_ = std::thread(&PipelineStage::monitoring_loop, this);
  }

  virtual void stop() {
    should_stop_ = true;
    message_available_cv_.notify_all();

    if (monitoring_thread_.joinable()) {
      monitoring_thread_.join();
    }
  }

  void message_loop() {
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

  void monitoring_loop() {
    while (!should_stop_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(update_interval));
      if (should_stop_) {
        break;
      }
      update_load_tracker();
    }
  }

  bool is_configured() const { return is_configured_; }

  std::string get_stage_id() const { return stage_id_; }

  std::string name() const { return name_; }

protected:
  virtual void process_message(const tpipeline::Message<T> &message) {
    switch (message.command_type) {
    case CommandType::FORWARD_TASK: {
      const Task<T> &forward_task = message.get_task();
      Tensor<T> output_data = this->model_->forward(forward_task.data, forward_task.micro_batch_id);
      Task<T> output_task(TaskType::FORWARD, std::move(output_data), forward_task.micro_batch_id);

      auto output_message = Message<T>::forward_task(output_task, name_, "next_stage");
      output_message.sequence_number = message.sequence_number;

      communicator_->send_message(output_message);
    } break;
    case CommandType::BACKWARD_TASK: {
      const Task<T> &backward_task = message.get_task();
      Tensor<T> output_data =
          this->model_->backward(backward_task.data, backward_task.micro_batch_id);
      Task<T> output_task(TaskType::BACKWARD, std::move(output_data), backward_task.micro_batch_id);

      auto output_message = Message<T>::backward_task(output_task, name_, "prev_stage");
      output_message.sequence_number = message.sequence_number;

      communicator_->send_message(output_message);
    } break;
    case CommandType::UPDATE_PARAMETERS: {
      model_->update_parameters();
      auto response = Message<T>::parameters_updated(this->name_, "coordinator");
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
      if (message.has_text()) {
        std::cout << "Stage " << name_ << " received error: " << message.get_text() << " from "
                  << message.sender_id << std::endl;
      }
      break;
    case CommandType::PRINT_PROFILING:
      if (model_) {
        model_->print_profiling_summary();
        auto outgoing_message = Message<T>::create_status_message(CommandType::PROFILING_PRINTED,
                                                                  name_, message.sender_id);
        communicator_->send_message(outgoing_message);
      } else {
        std::cout << "Warning: No model available to print profiling data" << std::endl;
      }
      break;
    case CommandType::CLEAR_PROFILING:
      if (model_) {
        model_->clear_profiling_data();
        auto outgoing_message = Message<T>::create_status_message(CommandType::PROFILING_CLEARED,
                                                                  name_, message.sender_id);
        communicator_->send_message(outgoing_message);
      } else {
        std::cout << "Warning: No model available to clear profiling data" << std::endl;
      }
      break;
    case CommandType::CONFIG_TRANSFER:
      this->handle_configuration(message);
      break;
    case CommandType::LOAD_PARAMS: {
      // decode and deserialize
      std::vector<uint8_t> params = message.get_binary();
      std::vector<Tensor<T>> parameters = BinarySerializer::deserialize_parameters<T>(params);

      model_->load_parameters(std::move(parameters));

      // send confirmation
      auto response =
          Message<T>::create_status_message(CommandType::PARAMS_LOADED, name_, message.sender_id);
      communicator_->send_message(response);
      break;
    }
    case CommandType::SEND_PARAMS: {
      try {
        std::vector<Tensor<T> *> param_ptrs = model_->parameters();
        std::vector<Tensor<T>> params_copy;
        params_copy.reserve(param_ptrs.size());
        for (const auto &param_ptr : param_ptrs) {
          if (param_ptr) {
            params_copy.emplace_back(param_ptr->clone());
          }
        }

        auto serialized_params = BinarySerializer::serialize_parameters(params_copy);
        auto params_msg = Message<T>(CommandType::PARAMS_TRANSFER, serialized_params);
        params_msg.sender_id = name_;
        params_msg.recipient_id = message.sender_id;

        communicator_->send_message(params_msg);
        std::cout << "Sent " << params_copy.size() << " parameters to " << message.sender_id
                  << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "Failed to send parameters: " << e.what() << std::endl;
        auto error_msg = Message<T>::error_message(
            std::string("Failed to send parameters: ") + e.what(), name_, message.sender_id);
        communicator_->send_message(error_msg);
      }
      break;
    }
    case CommandType::REPORT_LOAD: {
      std::vector<uint8_t> serialized_tracker = LoadTracker::serialize(load_tracker_);
      Message<T> load_msg(CommandType::LOAD_REPORT, serialized_tracker);
      load_msg.sender_id = name_;
      load_msg.recipient_id = "coordinator";
      communicator_->send_message(load_msg);
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
      const std::map<std::string, int64_t> forward_times = model_->get_forward_times();
      const std::map<std::string, int64_t> backward_times = model_->get_backward_times();

      int64_t cummulative_forward_time = std::accumulate(
          forward_times.begin(), forward_times.end(), 0LL,
          [](int64_t sum, const std::pair<std::string, int64_t> &p) { return sum + p.second; });

      int64_t cummulative_backward_time = std::accumulate(
          backward_times.begin(), backward_times.end(), 0LL,
          [](int64_t sum, const std::pair<std::string, int64_t> &p) { return sum + p.second; });

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

  void handle_configuration(const Message<T> &message) {
    if (!message.has_text()) {
      std::cout << "Configuration message missing text data" << '\n';
      return;
    }

    try {
      nlohmann::json config_json = nlohmann::json::parse(message.get_text());

      // std::cout << "Received configuration JSON: " << config_json.dump(2) << '\n';

      StageConfig config = StageConfig::from_json(config_json);

      stage_id_ = config.stage_id;

      std::cout << "Received configuration for stage " << stage_id_ << '\n';

      this->model_ = std::make_unique<tnn::Sequential<T>>(
          tnn::Sequential<T>::load_from_config(config.model_config));

      this->model_->print_config();
      this->model_->enable_profiling(true);

      std::cout << "Created model with " << this->model_->layer_size() << " layers" << '\n';

      setup_stage_connections(config);

      this->name_ = stage_id_;

      is_configured_ = true;

      auto ready_msg = Message<T>::create_signal_message(CommandType::CONFIG_RECEIVED, true,
                                                         stage_id_, "coordinator");
      this->communicator_->enqueue_output_message(ready_msg);
      this->communicator_->flush_output_messages();
    } catch (const std::exception &e) {
      std::cout << "Failed to configure stage: " << e.what() << '\n';

      auto error_msg =
          Message<T>::error_message(std::string("Configuration failed: ") + e.what(),
                                    stage_id_.empty() ? "unknown" : stage_id_, "coordinator");
      this->communicator_->enqueue_output_message(error_msg);
      this->communicator_->flush_output_messages();
    }
  }

  virtual void setup_stage_connections(const StageConfig &config) = 0;

  std::unique_ptr<tnn::Sequential<T>> model_;
  std::unique_ptr<PipelineCommunicator<T>, std::function<void(PipelineCommunicator<T> *)>>
      communicator_;
  std::string name_;
  std::atomic<bool> should_stop_;
  std::atomic<bool> is_configured_;
  std::string stage_id_;

  std::mutex message_available_mutex_;
  std::condition_variable message_available_cv_;

  utils::HardwareInfo cpu_info_;
  LoadTracker load_tracker_;
  uint32_t update_interval = 1000;
  std::thread monitoring_thread_;
};

} // namespace tpipeline