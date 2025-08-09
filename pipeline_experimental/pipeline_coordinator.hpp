#pragma once

#include "../nn/sequential.hpp"
#include "pipeline_communicator.hpp"
#include "pipeline_stage.hpp"

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

  // Process a batch by splitting it into microbatches and sending to first
  // stage
  void process_batch(const Tensor<T> &batch, int num_microbatches = 4) {
    if (stages_.empty())
      return;

    // Split batch into microbatches
    std::vector<Tensor<T>> microbatches =
        split_into_microbatches(batch, num_microbatches);

    // Send each microbatch to the first stage
    auto *first_stage_comm = stages_[0]->get_communicator();
    for (int i = 0; i < microbatches.size(); ++i) {
      tpipeline::Task<T> task(tpipeline::TaskType::Forward, microbatches[i], i);
      first_stage_comm->enqueue_task(task);
    }
  }

protected:
  int num_stages_;
  int num_microbatches_;
  std::vector<std::unique_ptr<PipelineStage<T>>> stages_;
  std::vector<std::unique_ptr<PipelineCommunicator<T>>> communicators_;
  std::unique_ptr<PipelineCommunicator<T>> coordinator_comm_;

  // Helper function to split batch into microbatches
  std::vector<Tensor<T>> split_into_microbatches(const Tensor<T> &batch,
                                                 int num_microbatches) {
    std::vector<Tensor<T>> microbatches;
    size_t batch_size = batch.batch_size() / num_microbatches;

    for (int i = 0; i < num_microbatches; ++i) {
      size_t start = i * batch_size;
      size_t end =
          (i == num_microbatches - 1) ? batch.batch_size() : start + batch_size;
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
          std::move(model_ptr), std::move(stage_communicator), "stage_" + std::to_string(i)));
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
                            : this->coordinator_comm_.get();
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
};

} // namespace tpipeline