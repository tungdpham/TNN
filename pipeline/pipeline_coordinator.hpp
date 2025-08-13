#pragma once

#include "../nn/optimizers.hpp"
#include "../nn/sequential.hpp"
#include "pipeline_communicator.hpp"
#include "pipeline_stage.hpp"
#include <chrono>
#include <condition_variable>
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

  // Core interface - only communicates through messages
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void join(bool direction) = 0;
  virtual void forward(const Tensor<T> &batch) = 0;
  virtual void backward(const std::vector<Tensor<T>> &gradients) = 0;

  // Message-based communication only
  void send_message_to_stage(const std::string &stage_id,
                             const Message<T> &message) {
    coordinator_comm_->send_message(stage_id, message);
  }

  std::vector<Message<T>> get_task_messages(){
    std::vector<Message<T>> task_messages;
    while (coordinator_comm_->has_task_message()) {
      try {
        Message<T> message = coordinator_comm_->dequeue_task_message();
        task_messages.push_back(message);
      } catch (const std::runtime_error &e) {
        break; // No more task messages available
      }
    }
    return task_messages;
  }

  // Get all status messages specifically
  std::vector<Message<T>> get_status_messages() {
    std::vector<Message<T>> status_messages;
    while (this->coordinator_comm_->has_status_message()) {
      try {
        Message<T> message = this->coordinator_comm_->dequeue_status_message();
        status_messages.push_back(message);
      } catch (const std::runtime_error &e) {
        break; // No more status messages available
      }
    }
    return status_messages;
  }

  // Get all parameter update messages specifically
  std::vector<Message<T>> get_parameter_update_messages() {
    std::vector<Message<T>> param_messages;
    while (this->coordinator_comm_->has_parameter_update_message()) {
      try {
        Message<T> message = this->coordinator_comm_->dequeue_parameter_update_message();
        param_messages.push_back(message);
      } catch (const std::runtime_error &e) {
        break; // No more parameter update messages available
      }
    }
    return param_messages;
  }

protected:
  int num_stages_;
  int num_microbatches_;
  std::shared_ptr<PipelineCommunicator<T>>
      coordinator_comm_;                 // Changed to shared_ptr
  std::vector<std::string> stage_names_; // Only keep stage identifiers
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

    // Split the model into stages
    auto splitted_models = model.split(num_stages);

    // Create coordinator communicator as shared_ptr
    this->coordinator_comm_ =
        std::make_shared<InProcessPipelineCommunicator<T>>();

    // Set up notification callback for task message arrivals
    this->coordinator_comm_->set_message_notification_callback([this]() {
      std::lock_guard<std::mutex> lock(task_notification_mutex_);
      task_notification_cv_.notify_all();
    });

    // Generate stage names
    for (int i = 0; i < this->num_stages_; ++i) {
      this->stage_names_.push_back("stage_" + std::to_string(i));
    }

    // Create stages and establish communication network
    std::vector<std::shared_ptr<PipelineCommunicator<T>>> stage_communicators;

    for (int i = 0; i < this->num_stages_; ++i) {
      // Create stage model
      splitted_models[i].enable_profiling(true);
      auto model_ptr =
          std::make_unique<tnn::Sequential<T>>(std::move(splitted_models[i]));

      // Create stage communicator
      auto stage_communicator =
          std::make_shared<InProcessPipelineCommunicator<T>>();
      stage_communicators.push_back(stage_communicator);

      // Create stage - use raw pointer and manage lifetime manually
      auto stage_comm_unique =
          std::unique_ptr<PipelineCommunicator<T>,
                          std::function<void(PipelineCommunicator<T> *)>>(
              stage_communicator.get(),
              [](PipelineCommunicator<T> *) { /* no-op deleter */ });

      auto stage = std::make_unique<PipelineStage<T>>(
          std::move(model_ptr), std::move(stage_comm_unique),
          this->stage_names_[i]);

      // Store the stage temporarily for setup only
      temp_stages_.emplace_back(std::move(stage));

      printf("Created stage %d with name: %s\n", i,
             this->stage_names_[i].c_str());
    }

    // Set up communication topology
    setup_communication_network(stage_communicators);

    // NOTE: Don't clear temp_stages_ here - they need to stay alive!
    // Call get_stages() to transfer ownership to caller

    printf("Pipeline coordinator initialized with %d stages\n",
           this->num_stages_);
  }

  // Transfer ownership of stages to caller and clear internal references
  std::vector<std::unique_ptr<PipelineStage<T>>> get_stages() {
    auto stages = std::move(temp_stages_);
    temp_stages_.clear();
    return stages;
  }

  // Check if stages have been transferred
  bool stages_transferred() const { return temp_stages_.empty(); }

  void start() override {
    if (!stages_transferred()) {
      throw std::runtime_error(
          "Must call get_stages() first to transfer stage ownership");
    }

    // Send start command to all stages
    for (const auto &stage_name : this->stage_names_) {
      auto start_msg = Message<T>::create_control_message(
          CommandType::START_TRAINING, "coordinator", stage_name);
      this->send_message_to_stage(stage_name, start_msg);
    }

    // Flush all outgoing messages
    this->coordinator_comm_->flush_output_messages();

    printf("Started all %d pipeline stages\n", this->num_stages_);
  }

  void stop() override {
    // Send stop command to all stages
    for (const auto &stage_name : this->stage_names_) {
      auto stop_msg = Message<T>::create_control_message(
          CommandType::STOP_TRAINING, "coordinator", stage_name);
      this->send_message_to_stage(stage_name, stop_msg);
    }

    this->coordinator_comm_->flush_output_messages();
    printf("Stopped all pipeline stages\n");
  }

  void forward(const Tensor<T> &batch) override {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    // Split batch into microbatches
    auto microbatches = batch.split(this->num_microbatches_);

    // printf("Forwarding %zu microbatches to first stage\n", microbatches.size());

    // Send each microbatch to the first stage
    for (int i = 0; i < this->num_microbatches_; ++i) {
      Task<T> task{TaskType::FORWARD, microbatches[i],
                   i}; // batch_id=0, microbatch_id=i
      auto forward_msg =
          Message<T>::forward_task(task, "coordinator", this->stage_names_[0]);
      forward_msg.sequence_number = i;

      this->send_message_to_stage(this->stage_names_[0], forward_msg);
    }

    this->coordinator_comm_->flush_output_messages();
  }

  void backward(const std::vector<Tensor<T>> &gradients) override {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &last_stage = this->stage_names_[this->num_stages_ - 1];
    
    // Send each gradient to the last stage
    for (size_t i = 0; i < gradients.size(); ++i) {
      Task<T> task{TaskType::BACKWARD, gradients[i], static_cast<int>(i)};
      auto backward_msg =
          Message<T>::backward_task(task, "coordinator", last_stage);
      backward_msg.sequence_number = static_cast<int>(i);

      this->send_message_to_stage(last_stage, backward_msg);
    }

    this->coordinator_comm_->flush_output_messages();
  }

  void join(bool direction) override {
    // Set expected task count based on direction
    expected_task_count_ = this->num_microbatches_;

    std::unique_lock<std::mutex> lock(task_notification_mutex_);
    
    // Wait with timeout for the expected number of task messages to arrive
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    
    bool success = task_notification_cv_.wait_until(lock, timeout, [this]() {
      return this->coordinator_comm_->actual_task_message_count() >= 
             static_cast<size_t>(expected_task_count_);
    });

    if (!success) {
      printf("Warning: join() timed out waiting for task messages. "
             "Expected: %d, Got: %zu\n", 
             expected_task_count_.load(),
             this->coordinator_comm_->actual_task_message_count());
    } else {
      // printf("Successfully received %zu task messages\n",
      //        this->coordinator_comm_->actual_task_message_count());
    }
  }

  void print_profiling_on_all_stages() {
    for (const auto &stage_names_ : this->stage_names_) {
      auto profiling_msg = Message<T>::create_control_message(
          CommandType::PRINT_PROFILING, "coordinator", stage_names_);
      this->send_message_to_stage(stage_names_, profiling_msg);
    }
    this->coordinator_comm_->flush_output_messages();
    printf("Sent profiling request to all stages\n");
  }

  // Status and monitoring through messages only
  void request_status_from_all_stages() {
    for (const auto &stage_name : this->stage_names_) {
      auto status_msg = Message<T>(CommandType::STATUS_REQUEST, true,
                                   "coordinator", stage_name);
      this->send_message_to_stage(stage_name, status_msg);
    }
    this->coordinator_comm_->flush_output_messages();
  }

  void update_parameters() {
    for (const auto &stage_name : this->stage_names_) {
      auto update_msg = Message<T>(CommandType::UPDATE_PARAMETERS, true,
                                   "coordinator", stage_name);
      this->send_message_to_stage(stage_name, update_msg);
    }
    this->coordinator_comm_->flush_output_messages();

    // Wait for confirmations
    wait_for_parameter_updates();
  }

private:
  // Temporary storage for stages during initialization only
  std::vector<std::unique_ptr<PipelineStage<T>>> temp_stages_;
  std::vector<std::shared_ptr<PipelineCommunicator<T>>>
      stage_comm_refs_; // Keep refs alive

  // Synchronization for event-driven join()
  mutable std::mutex task_notification_mutex_;
  mutable std::condition_variable task_notification_cv_;
  std::atomic<int> expected_task_count_{0};

  void setup_communication_network(
      const std::vector<std::shared_ptr<PipelineCommunicator<T>>>
          &stage_comms) {

    // Store references to keep communicators alive
    stage_comm_refs_ = stage_comms;

    auto coordinator_comm_shared =
        std::static_pointer_cast<InProcessPipelineCommunicator<T>>(
            this->coordinator_comm_);

    // Register coordinator in all stage communicators
    for (int i = 0; i < this->num_stages_; ++i) {
      auto stage_comm =
          std::static_pointer_cast<InProcessPipelineCommunicator<T>>(
              stage_comms[i]);
      stage_comm->register_communicator("coordinator", this->coordinator_comm_);
    }

    // Set up inter-stage communication
    for (int i = 0; i < this->num_stages_; ++i) {
      auto current_comm =
          std::static_pointer_cast<InProcessPipelineCommunicator<T>>(
              stage_comms[i]);

      // Register previous stage
      if (i > 0) {
        current_comm->register_communicator("prev_stage", stage_comms[i - 1]);
        current_comm->register_recipient(
            "prev_stage", StageEndpoint::in_process(this->stage_names_[i - 1]));
      } else {
        // First stage receives from coordinator
        current_comm->register_communicator("prev_stage",
                                            this->coordinator_comm_);
        current_comm->register_recipient(
            "prev_stage", StageEndpoint::in_process("coordinator"));
      }

      // Register next stage
      if (i < this->num_stages_ - 1) {
        current_comm->register_communicator("next_stage", stage_comms[i + 1]);
        current_comm->register_recipient(
            "next_stage", StageEndpoint::in_process(this->stage_names_[i + 1]));
      } else {
        // Last stage sends to coordinator
        current_comm->register_communicator("next_stage",
                                            this->coordinator_comm_);
        current_comm->register_recipient(
            "next_stage", StageEndpoint::in_process("coordinator"));
      }

      // Register coordinator
      current_comm->register_recipient(
          "coordinator", StageEndpoint::in_process("coordinator"));
    }

    // Register stages in coordinator
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
      // Get specifically parameter update messages
      auto param_messages = this->get_parameter_update_messages();
      for (const auto &message : param_messages) {
        if (message.command_type == CommandType::PARAMETERS_UPDATED) {
          confirmations++;
        }
      }

      auto elapsed = std::chrono::steady_clock::now() - start_time;
      if (elapsed > timeout) {
        printf("Warning: Timeout waiting for parameter updates. Got %d/%d "
               "confirmations\n",
               confirmations, this->num_stages_);
        break;
      }

      if (confirmations < this->num_stages_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      } else if (confirmations == this->num_stages_) {
        // printf("All stages confirmed parameter updates\n");
        break;
      }
    }
  }
};

} // namespace tpipeline