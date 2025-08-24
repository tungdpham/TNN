#pragma once

#include "../nn/optimizers.hpp"
#include "../nn/sequential.hpp"
#include "pipeline_communicator.hpp"
#include "pipeline_stage.hpp"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

namespace tpipeline {

template <typename T = float> class PipelineCoordinator {
public:
  PipelineCoordinator(int num_stages = 4, int num_microbatches = 4)
      : num_stages_(num_stages), num_microbatches_(num_microbatches) {
    if (num_stages < 1 || num_microbatches < 1) {
      throw std::invalid_argument(
          "Number of stages and microbatches must be at least 1");
    }
  }

  virtual ~PipelineCoordinator() = default;

  
  virtual void start() = 0;

  virtual void stop() = 0;

  virtual void join(bool direction) = 0;

  virtual void forward(const Tensor<T> &input, size_t microbatch_id) = 0;

  virtual void backward(const Tensor<T> &gradient, size_t microbatch_id) = 0;

  virtual void update_parameters() = 0;

  virtual void print_profiling_on_all_stages() = 0;

  virtual void clear_profiling_data() = 0;

  
  void send_message_to_stage(const std::string &stage_id,
                             const Message<T> &message) {
    coordinator_comm_->send_message(stage_id, message);
  }

  std::vector<Message<T>> get_task_messages() {
    std::vector<Message<T>> task_messages;
    while (coordinator_comm_->has_task_message()) {
      try {
        Message<T> message = coordinator_comm_->dequeue_task_message();
        task_messages.push_back(message);
      } catch (const std::runtime_error &e) {
        break; 
      }
    }
    return task_messages;
  }

  
  std::vector<Message<T>> get_status_messages() {
    std::vector<Message<T>> status_messages;
    while (this->coordinator_comm_->has_status_message()) {
      try {
        Message<T> message = this->coordinator_comm_->dequeue_status_message();
        status_messages.push_back(message);
      } catch (const std::runtime_error &e) {
        break; 
      }
    }
    return status_messages;
  }

  
  std::vector<Message<T>> get_parameter_update_messages() {
    std::vector<Message<T>> param_messages;
    while (this->coordinator_comm_->has_parameter_update_message()) {
      try {
        Message<T> message =
            this->coordinator_comm_->dequeue_parameter_update_message();
        param_messages.push_back(message);
      } catch (const std::runtime_error &e) {
        break; 
      }
    }
    return param_messages;
  }

protected:
  int num_stages_;
  int num_microbatches_;
  std::shared_ptr<PipelineCommunicator<T>>
      coordinator_comm_;                 
  std::vector<std::string> stage_names_; 
};

template <typename T = float>
class InProcessPipelineCoordinator : public PipelineCoordinator<T> {
public:
  InProcessPipelineCoordinator(tnn::Sequential<T> model, int num_stages = 4,
                               int num_microbatches = 4)
      : PipelineCoordinator<T>(num_stages, num_microbatches) {

    if (model.get_layers().size() < num_stages) {
      throw std::invalid_argument(
          "Model must have at least as many layers as stages");
    }

    
    auto splitted_models = model.split(num_stages);

    
    this->coordinator_comm_ =
        std::make_shared<InProcessPipelineCommunicator<T>>();

    
    this->coordinator_comm_->set_message_notification_callback([this]() {
      std::lock_guard<std::mutex> lock(message_notification_mutex_);
      message_notification_cv_.notify_all();
    });

    
    for (int i = 0; i < this->num_stages_; ++i) {
      this->stage_names_.push_back("stage_" + std::to_string(i));
    }

    
    std::vector<std::shared_ptr<PipelineCommunicator<T>>> stage_communicators;

    for (int i = 0; i < this->num_stages_; ++i) {
      
      splitted_models[i].enable_profiling(true);
      auto model_ptr =
          std::make_unique<tnn::Sequential<T>>(std::move(splitted_models[i]));

      
      auto stage_communicator =
          std::make_shared<InProcessPipelineCommunicator<T>>();
      stage_communicators.push_back(stage_communicator);

      
      auto stage_comm_unique =
          std::unique_ptr<PipelineCommunicator<T>,
                          std::function<void(PipelineCommunicator<T> *)>>(
              stage_communicator.get(),
              [](PipelineCommunicator<T> *) {  });

      auto stage = std::make_unique<PipelineStage<T>>(
          std::move(model_ptr), std::move(stage_comm_unique),
          this->stage_names_[i]);

      
      temp_stages_.emplace_back(std::move(stage));

      std::cout << "Created stage: " << this->stage_names_[i]
                << " with model layers: "
                << temp_stages_.back()->get_model()->get_layers().size()
                << std::endl;
    }

    
    setup_communication_network(stage_communicators);

    std::cout << "Pipeline coordinator initialized with " << this->num_stages_
              << " stages" << std::endl;
  }

  
  std::vector<std::unique_ptr<PipelineStage<T>>> get_stages() {
    auto stages = std::move(temp_stages_);
    temp_stages_.clear();
    return stages;
  }

  
  bool stages_transferred() const { return temp_stages_.empty(); }

  void start() override {
    if (!stages_transferred()) {
      throw std::runtime_error(
          "Must call get_stages() first to transfer stage ownership");
    }

    
    for (const auto &stage_name : this->stage_names_) {
      auto start_msg = Message<T>::create_control_message(
          CommandType::START_TRAINING, "coordinator", stage_name);
      this->send_message_to_stage(stage_name, start_msg);
    }

    
    this->coordinator_comm_->flush_output_messages();

    std::cout << "Started all " << this->num_stages_ << " pipeline stages"
              << std::endl;
  }

  void stop() override {
    
    for (const auto &stage_name : this->stage_names_) {
      auto stop_msg = Message<T>::create_control_message(
          CommandType::STOP_TRAINING, "coordinator", stage_name);
      this->send_message_to_stage(stage_name, stop_msg);
    }

    this->coordinator_comm_->flush_output_messages();
    std::cout << "Stopped all pipeline stages" << std::endl;
  }

  void forward(const Tensor<T> &input, size_t microbatch_id) override {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &first_stage = this->stage_names_[0];

    
    Task<T> task{TaskType::FORWARD, input, microbatch_id};
    auto forward_msg =
        Message<T>::forward_task(task, "coordinator", first_stage);
    forward_msg.sequence_number = microbatch_id;

    this->send_message_to_stage(first_stage, forward_msg);

    this->coordinator_comm_->flush_output_messages();
  }

  void backward(const Tensor<T> &gradient, size_t microbatch_id) override {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &last_stage = this->stage_names_.back();

    Task<T> task{TaskType::BACKWARD, gradient, microbatch_id};
    auto backward_msg =
        Message<T>::backward_task(task, "coordinator", last_stage);
    backward_msg.sequence_number = microbatch_id;

    this->send_message_to_stage(last_stage, backward_msg);

    this->coordinator_comm_->flush_output_messages();
  }

  void join(bool direction) override {
    
    expected_task_count_ = this->num_microbatches_;

    std::unique_lock<std::mutex> lock(message_notification_mutex_);

    
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);

    bool success =
        message_notification_cv_.wait_until(lock, timeout, [this, direction]() {
          if (direction) {
            return this->coordinator_comm_->forward_message_count() >=
                   expected_task_count_.load();
          } else {
            return this->coordinator_comm_->backward_message_count() >=
                   expected_task_count_.load();
          }
        });

    if (!success) {
      std::cout << "Warning: join() timed out waiting for task messages. "
                << "Expected: " << expected_task_count_.load() << ", Got: "
                << (direction
                        ? this->coordinator_comm_->forward_message_count()
                        : this->coordinator_comm_->backward_message_count())
                << '\n';
    }
    return;
  }

  void print_profiling_on_all_stages() override {
    for (const auto &stage_names_ : this->stage_names_) {
      auto profiling_msg = Message<T>::create_control_message(
          CommandType::PRINT_PROFILING, "coordinator", stage_names_);
      this->send_message_to_stage(stage_names_, profiling_msg);
    }
    std::cout << "Sent profiling request to all stages" << std::endl;
  }

  void clear_profiling_data() {
    for (const auto &stage_name : this->stage_names_) {
      auto clear_msg = Message<T>::create_control_message(
          CommandType::CLEAR_PROFILING, "coordinator", stage_name);
      this->send_message_to_stage(stage_name, clear_msg);
    }
    std::cout << "Sent clear profiling data request to all stages" << std::endl;
  }

  
  void request_status_from_all_stages() {
    for (const auto &stage_name : this->stage_names_) {
      auto status_msg = Message<T>(CommandType::STATUS_REQUEST, true,
                                   "coordinator", stage_name);
      this->send_message_to_stage(stage_name, status_msg);
    }
  }

  void update_parameters() override {
    for (const auto &stage_name : this->stage_names_) {
      auto update_msg = Message<T>(CommandType::UPDATE_PARAMETERS, true,
                                   "coordinator", stage_name);
      this->send_message_to_stage(stage_name, update_msg);
    }
    
    wait_for_parameter_updates();
  }

private:
  
  std::vector<std::unique_ptr<PipelineStage<T>>> temp_stages_;
  std::vector<std::shared_ptr<PipelineCommunicator<T>>>
      stage_comm_refs_; 

  
  mutable std::mutex message_notification_mutex_;
  mutable std::condition_variable message_notification_cv_;
  std::atomic<int> expected_task_count_{0};

  void setup_communication_network(
      const std::vector<std::shared_ptr<PipelineCommunicator<T>>>
          &stage_comms) {

    
    stage_comm_refs_ = stage_comms;

    auto coordinator_comm_shared =
        std::static_pointer_cast<InProcessPipelineCommunicator<T>>(
            this->coordinator_comm_);

    
    for (int i = 0; i < this->num_stages_; ++i) {
      auto stage_comm =
          std::static_pointer_cast<InProcessPipelineCommunicator<T>>(
              stage_comms[i]);
      stage_comm->register_communicator("coordinator", this->coordinator_comm_);
    }

    
    for (int i = 0; i < this->num_stages_; ++i) {
      auto current_comm =
          std::static_pointer_cast<InProcessPipelineCommunicator<T>>(
              stage_comms[i]);

      
      if (i > 0) {
        current_comm->register_communicator("prev_stage", stage_comms[i - 1]);
        current_comm->register_recipient(
            "prev_stage", StageEndpoint::in_process(this->stage_names_[i - 1]));
      } else {
        
        current_comm->register_communicator("prev_stage",
                                            this->coordinator_comm_);
        current_comm->register_recipient(
            "prev_stage", StageEndpoint::in_process("coordinator"));
      }

      
      if (i < this->num_stages_ - 1) {
        current_comm->register_communicator("next_stage", stage_comms[i + 1]);
        current_comm->register_recipient(
            "next_stage", StageEndpoint::in_process(this->stage_names_[i + 1]));
      } else {
        
        current_comm->register_communicator("next_stage",
                                            this->coordinator_comm_);
        current_comm->register_recipient(
            "next_stage", StageEndpoint::in_process("coordinator"));
      }

      
      current_comm->register_recipient(
          "coordinator", StageEndpoint::in_process("coordinator"));
    }

    
    for (int i = 0; i < this->num_stages_; ++i) {
      coordinator_comm_shared->register_communicator(this->stage_names_[i],
                                                     stage_comms[i]);
      coordinator_comm_shared->register_recipient(
          this->stage_names_[i],
          StageEndpoint::in_process(this->stage_names_[i]));
    }
  }

  void wait_for_parameter_updates() {
    int confirmations = 0;
    auto start_time = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(10);

    while (confirmations < this->num_stages_) {
      
      auto param_messages = this->get_parameter_update_messages();
      for (const auto &message : param_messages) {
        if (message.command_type == CommandType::PARAMETERS_UPDATED) {
          confirmations++;
        }
      }

      auto elapsed = std::chrono::steady_clock::now() - start_time;
      if (elapsed > timeout) {
        std::cout << "Warning: Timeout waiting for parameter updates. Got "
                  << confirmations << "/" << this->num_stages_
                  << " confirmations" << std::endl;
        break;
      }

      if (confirmations < this->num_stages_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      } else if (confirmations == this->num_stages_) {
        
        break;
      }
    }
  }
};

} 