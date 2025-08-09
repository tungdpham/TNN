#pragma once

#include "../nn/sequential.hpp"
#include "pipeline_communicator.hpp"
#include "pipeline_stage.hpp"
#include "../nn/optimizers.hpp"

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
    create_optimizers();
  }

  // Add a stage to the pipeline
  void add_stage(std::unique_ptr<PipelineStage<T>> stage) {
    stages_.push_back(std::move(stage));
  }

  // Start all stages
  void start() {
    for (auto &stage : stages_) {
      printf("Starting stage: %s\n", stage->name().c_str());
      stage->start();
    }
  }

  // Stop all stages
  void stop() {
    for (auto &stage : stages_) {
      stage->stop();
    }
  }

  void join() {
    // Check if all stages have input tasks
    while (true) {
      bool all_empty = true;
      for (const auto &stage : stages_) {
        if (stage->get_communicator()->has_input_task() || stage->get_communicator()->has_output_task() || stage->is_processing()) {
          all_empty = false;
          // printf("Stage %s has input tasks or output tasks to process.\n",
                //  stage->name().c_str());
        }
      }
      if(all_empty) {
        printf("All stages have no tasks. Exiting join loop.\n");
        break;
      }
      // printf("Waiting for stages to finish processing...\n");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Get all tasks from coordinator communicator
  std::vector<Task<T>> get_all_tasks() {
    std::vector<Task<T>> all_tasks;
    while (coordinator_comm_->has_input_task()) {
      try {
        Task<T> task = coordinator_comm_->dequeue_input_task();
        printf("Coordinator dequeued task: %d\n", task.micro_batch_id);
        all_tasks.push_back(task);
      } catch (const std::runtime_error &e) {
        // Ignore empty queue errors
        printf("No more tasks in coordinator queue.\n");
        break;
      }
    }
    return all_tasks;
  }

  void create_optimizers(){
    // Create optimizers for each stage
    for (int i=0; i < num_stages_; ++i) {
      // Example: Create a simple SGD optimizer for each stage
      auto optimizer = std::make_unique<tnn::Adam<T>>(0.01f, 0.9f, 0.999f, 1e-8f);
      optimizers_.push_back(std::move(optimizer));
    }
  }

  virtual void forward(const Tensor<T> &batch, int num_microbatches = 4) = 0;
  virtual void backward(const std::vector<Tensor<T>> gradients) = 0;

protected:
  int num_stages_;
  int num_microbatches_;
  std::vector<std::unique_ptr<PipelineStage<T>>> stages_;
  std::vector<std::unique_ptr<PipelineCommunicator<T>>> communicators_;
  std::unique_ptr<PipelineCommunicator<T>> coordinator_comm_;
  std::vector<std::unique_ptr<tnn::Optimizer<T>>> optimizers_;

  // Helper function to split batch into microbatches
  std::vector<Tensor<T>> split_into_microbatches(const Tensor<T> &batch,
                                                 int num_microbatches) {
    std::vector<Tensor<T>> microbatches;
    printf("Splitting batch of size %zu into %d microbatches\n",
           batch.batch_size(), num_microbatches);
    size_t batch_size = batch.batch_size() / num_microbatches;

    for (int i = 0; i < num_microbatches; ++i) {
      size_t start = i * batch_size;
      size_t end = (i == num_microbatches - 1) ? batch.batch_size() - 1
                                               : start + batch_size - 1;
      microbatches.push_back(batch.slice_batch(start, end));
    }

    return microbatches;
  }
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
        std::make_unique<InProcessPipelineCommunicator<T>>(nullptr, nullptr);

    // Create stages and their communicators
    for (int i = 0; i < this->num_stages_; ++i) {
      auto model_ptr =
          std::make_unique<tnn::Sequential<T>>(std::move(splitted_models_[i]));
      auto stage_communicator =
          std::make_unique<InProcessPipelineCommunicator<T>>(nullptr, nullptr);

      this->stages_.emplace_back(std::make_unique<InProcessPipelineStage<T>>(
          std::move(model_ptr), std::move(stage_communicator),
          "stage_" + std::to_string(i)));
      printf("Created stage %d with model: %s\n", i,
             this->stages_.back()->name().c_str());
      for (const auto &layer :
           this->stages_.back()->get_model()->get_layers()) {
        printf("  Layer: %s\n", layer->name().c_str());
      }
    }

    // Now set up inter-stage communication links
    for (int i = 0; i < this->num_stages_; ++i) {
      auto *current_comm = static_cast<InProcessPipelineCommunicator<T> *>(
          this->stages_[i]->get_communicator());
      auto *prev_comm = (i > 0)
                            ? static_cast<InProcessPipelineCommunicator<T> *>(
                                  this->stages_[i - 1]->get_communicator())
                            : nullptr;
      auto *next_comm = (i < this->num_stages_ - 1)
                            ? static_cast<InProcessPipelineCommunicator<T> *>(
                                  this->stages_[i + 1]->get_communicator())
                            : this->coordinator_comm_.get();

      current_comm->set_prev_stage(
          static_cast<InProcessPipelineCommunicator<T> *>(prev_comm));
      current_comm->set_next_stage(
          static_cast<InProcessPipelineCommunicator<T> *>(next_comm));
    }

    // Set up coordinator communication links
    auto *first_stage_comm = static_cast<InProcessPipelineCommunicator<T> *>(
        this->stages_[0]->get_communicator());
    auto *last_stage_comm = static_cast<InProcessPipelineCommunicator<T> *>(
        this->stages_[this->num_stages_ - 1]->get_communicator());

    auto *coordinator_comm_raw =
        static_cast<InProcessPipelineCommunicator<T> *>(
            this->coordinator_comm_.get());

    coordinator_comm_raw->set_next_stage(first_stage_comm);
    coordinator_comm_raw->set_prev_stage(last_stage_comm);
  }

  void forward(const Tensor<T> &batch,
              int num_microbatches = 4) override {
    if (this->stages_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    // Split the batch into microbatches
    auto microbatches = this->split_into_microbatches(batch, num_microbatches);

    // Enqueue each microbatch to the first stage's communicator
    auto *first_stage_comm =
        static_cast<InProcessPipelineCommunicator<T> *>(
            this->stages_[0]->get_communicator());

    for (int i = 0; i < num_microbatches; ++i) {
      tpipeline::Task<T> task(tpipeline::TaskType::Forward, microbatches[i], i);

      printf("Enqueuing forward task for microbatch %d\n", i);
      first_stage_comm->enqueue_task(task);
    }
  }

  void backward(const std::vector<Tensor<T>> gradients) override {
    if (this->stages_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    // Enqueue each microbatch to the last stage's communicator
    auto *last_stage_comm =
        static_cast<InProcessPipelineCommunicator<T> *>(
            this->stages_[this->num_stages_ - 1]->get_communicator());

    for (int i = 0; i < gradients.size(); ++i) {
      tpipeline::Task<T> task(tpipeline::TaskType::Backward, gradients[i], i);

      printf("Enqueuing backward task for microbatch %d\n", i);
      last_stage_comm->enqueue_task(task);
    }
  }

  void update_params() {
    int idx = 0;
    for(auto &stage : this->stages_) {
      printf("Updating parameters for stage: %s\n", stage->name().c_str());
      auto model = stage->get_model();
      std::vector<Tensor<T>*> params = model->parameters();
      std::vector<Tensor<T>*> grads = model->gradients();
      for(int i=0;i<params.size();++i) {
        printf("Parameter %d shape: ", i);
        params[i]->print_info();
      }
      for(int i=0;i<grads.size();++i) {
        printf("Gradient %d shape: ", i);
        grads[i]->print_info();
      }
      this->optimizers_[idx]->update(params, grads);
      printf("Updated parameters for stage: %s\n", stage->name().c_str());
      ++idx;
    }
  }
};

} // namespace tpipeline