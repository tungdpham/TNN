#pragma once

#include "../nn/optimizers.hpp"
#include "../nn/sequential.hpp"
#include "pipeline_stage.hpp" // Includes communicator and endpoint

namespace tpipeline {
template <typename T = float> class PipelineCoordinator {
public:
  PipelineCoordinator(int num_stages = 4, int num_microbatches = 4)
      : num_stages_(num_stages), num_microbatches_(num_microbatches) {
    // Ensure the model has enough layers to split into stages
    if (num_stages < 1 || num_microbatches < 1) {
      throw std::invalid_argument(
          "Number of stages and microbatches must be at least 1");
    }
  }

  // Start all stages
  virtual void start() = 0;

  // Stop all stages
  virtual void stop() = 0;

  // Wait for all stages to finish processing
  virtual void join() = 0;

  // Get all messages from coordinator communicator
  std::vector<Message<T>> get_all_messages() {
    std::vector<Message<T>> all_messages;
    while (coordinator_comm_->has_input_message()) {
      try {
        Message<T> message = coordinator_comm_->dequeue_input_message();
        all_messages.push_back(message);
      } catch (const std::runtime_error &e) {
        break;
      }
    }
    return all_messages;
  }

  // Backward compatibility
  std::vector<Task<T>> get_all_tasks() {
    std::vector<Task<T>> all_tasks;
    auto messages = get_all_messages();
    for (const auto& message : messages) {
      if (message.is_task()) {
        all_tasks.push_back(message.template get_payload<Task<T>>());
      }
    }
    return all_tasks;
  }

  virtual void forward(const Tensor<T> &batch) = 0;
  virtual void backward(const std::vector<Tensor<T>> gradients) = 0;

protected:
  int num_stages_;
  int num_microbatches_;
  std::unique_ptr<PipelineCommunicator<T>> coordinator_comm_;
};

template <typename T = float>
class InProcessPipelineCoordinator : public PipelineCoordinator<T> {
public:
  InProcessPipelineCoordinator(tnn::Sequential<T> model, int num_stages = 4,
                               int num_microbatches = 4)
      : PipelineCoordinator<T>(num_stages, num_microbatches) {
    // Ensure the model has enough layers to split into stages
    if (model.get_layers().size() < num_stages) {
      throw std::invalid_argument("Model must have at least as many layers as "
                                  "the number of stages");
    }

    // Split the model into stages
    auto splitted_models_ = model.split(num_stages);

    // Create the coordinator communicator
    this->coordinator_comm_ =
        std::make_unique<InProcessPipelineCommunicator<T>>();

    // Create stages and their communicators
    for (int i = 0; i < this->num_stages_; ++i) {
      splitted_models_[i].enable_profiling(true);
      auto model_ptr =
          std::make_unique<tnn::Sequential<T>>(std::move(splitted_models_[i]));
      auto stage_communicator =
          std::make_unique<InProcessPipelineCommunicator<T>>();

      this->stages_.emplace_back(std::make_unique<PipelineStage<T>>(
          std::move(model_ptr), std::move(stage_communicator),
          "stage_" + std::to_string(i)));
      printf("Created stage %d with model: %s\n", i,
             this->stages_.back()->name().c_str());
      for (const auto &layer :
           this->stages_.back()->get_model()->get_layers()) {
        printf("  Layer: %s\n", layer->name().c_str());
      }
    }

    // Now set up inter-stage communication links using endpoints
    for (int i = 0; i < this->num_stages_; ++i) {
      auto *current_comm = this->stages_[i]->get_communicator();

      // Endpoint for the previous stage
      if (i > 0) {
        auto *prev_comm = this->stages_[i - 1]->get_communicator();
        InProcessStageEndpointDetails<T> prev_details{prev_comm};
        StageEndpoint prev_endpoint("stage_" + std::to_string(i - 1),
                                    prev_details);
        current_comm->set_prev_stage_endpoint(prev_endpoint);
      }

      // Endpoint for the next stage
      if (i < this->num_stages_ - 1) {
        auto *next_comm = this->stages_[i + 1]->get_communicator();
        InProcessStageEndpointDetails<T> next_details{next_comm};
        StageEndpoint next_endpoint("stage_" + std::to_string(i + 1),
                                    next_details);
        current_comm->set_next_stage_endpoint(next_endpoint);
      } else {
        // The last stage sends to the coordinator
        InProcessStageEndpointDetails<T> next_details{
            this->coordinator_comm_.get()};
        StageEndpoint next_endpoint("coordinator", next_details);
        current_comm->set_next_stage_endpoint(next_endpoint);
      }
    }

    // Set up coordinator's endpoints
    auto *first_stage_comm = this->stages_[0]->get_communicator();
    InProcessStageEndpointDetails<T> first_stage_details{first_stage_comm};
    StageEndpoint first_stage_endpoint("stage_0", first_stage_details);
    this->coordinator_comm_->set_next_stage_endpoint(first_stage_endpoint);

    auto *last_stage_comm =
        this->stages_[this->num_stages_ - 1]->get_communicator();
    InProcessStageEndpointDetails<T> last_stage_details{last_stage_comm};
    StageEndpoint last_stage_endpoint(
        "stage_" + std::to_string(this->num_stages_ - 1), last_stage_details);
    this->coordinator_comm_->set_prev_stage_endpoint(last_stage_endpoint);

    // Create optimizers for each stage
    for (int i = 0; i < this->num_stages_; ++i) {
      // Example: Create a simple SGD optimizer for each stage
      auto optimizer =
          std::make_unique<tnn::Adam<T>>(0.01f, 0.9f, 0.999f, 1e-8f);
      optimizers_.push_back(std::move(optimizer));
    }
  }

  void start() override {
    for (auto &stage : this->stages_) {
      stage->start();
    }
  }

  void stop() override {
    for (auto &stage : this->stages_) {
      stage->stop();
    }
  }

  void forward(const Tensor<T> &batch) override {
    if (this->stages_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    // Split the batch into microbatches
    auto microbatches = batch.split(this->num_microbatches_);

    // Enqueue each microbatch to the first stage's communicator
    auto *first_stage_comm = this->stages_[0]->get_communicator();

    for (int i = 0; i < this->num_microbatches_; ++i) {
      tpipeline::Task<T> task(tpipeline::TaskType::Forward, microbatches[i], i);
      Message<T> message(CommandType::FORWARD_TASK, task, "coordinator", "stage_0");
      first_stage_comm->enqueue_input_message(message);
    }
  }

  void backward(const std::vector<Tensor<T>> gradients) override {
    if (this->stages_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    // Enqueue each gradient to the last stage's communicator
    auto *last_stage_comm =
        this->stages_[this->num_stages_ - 1]->get_communicator();

    for (size_t i = 0; i < gradients.size(); ++i) {
      tpipeline::Task<T> task(tpipeline::TaskType::Backward, gradients[i], i);
      Message<T> message(CommandType::BACKWARD_TASK, task, 
                        "coordinator", 
                        "stage_" + std::to_string(this->num_stages_ - 1));
      last_stage_comm->enqueue_input_message(message);
    }
  }

  void update_params() {
    int idx = 0;
    for (auto &stage : this->stages_) {
      auto model = stage->get_model();
      model->update_parameters();
      ++idx;
    }
  }

  // Send control commands to all stages
  void send_start_training_command() {
    Message<T> command(CommandType::START_TRAINING, true, "coordinator");
    for (auto &stage : this->stages_) {
      command.recipient_id = stage->name();
      stage->get_communicator()->enqueue_input_message(command);
    }
  }

  void send_stop_training_command() {
    Message<T> command(CommandType::STOP_TRAINING, true, "coordinator");
    for (auto &stage : this->stages_) {
      command.recipient_id = stage->name();
      stage->get_communicator()->enqueue_input_message(command);
    }
  }

  // Request status from all stages
  void request_status_from_all_stages() {
    Message<T> request(CommandType::STATUS_REQUEST, true, "coordinator");
    for (auto &stage : this->stages_) {
      request.recipient_id = stage->name();
      stage->get_communicator()->enqueue_input_message(request);
    }
  }

  // Get status responses
  std::vector<tpipeline::StatusInfo> get_status_responses() {
    std::vector<tpipeline::StatusInfo> statuses;
    auto messages = this->get_all_messages();
    for (const auto& message : messages) {
      if (message.command_type == CommandType::STATUS_RESPONSE && 
          message.template has_payload<tpipeline::StatusInfo>()) {
        statuses.push_back(message.template get_payload<tpipeline::StatusInfo>());
      }
    }
    return statuses;
  }

  void join() override {
    // Wait for all stages to complete processing all tasks
    bool all_done = false;
    while (!all_done) {
      all_done = true;
      for (const auto &stage : this->stages_) {
        if (stage->get_communicator()->has_input_message() ||
            stage->get_communicator()->has_output_message() ||
            stage->is_processing()) {
          all_done = false;
          break;
        }
      }
      if (all_done)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
  }

private:
  std::vector<std::unique_ptr<tnn::Optimizer<T>>> optimizers_;
  std::vector<std::unique_ptr<PipelineStage<T>>> stages_;
};

} // namespace tpipeline