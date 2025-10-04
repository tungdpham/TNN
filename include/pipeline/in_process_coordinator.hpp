/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include "in_process_communicator.hpp"
#include "nn/partitioner.hpp"
#include "pipeline_coordinator.hpp"

namespace tpipeline {

// Concrete implementation of PipelineStage for in-process communication
template <typename T = float> class InProcessPipelineStage : public PipelineStage<T> {
public:
  InProcessPipelineStage(
      std::unique_ptr<tnn::Sequential<T>> model,
      std::unique_ptr<PipelineCommunicator<T>, std::function<void(PipelineCommunicator<T> *)>>
          communicator,
      const std::string &name)
      : PipelineStage<T>(std::move(model), std::move(communicator), name) {}

protected:
  void setup_stage_connections(const StageConfig &config) override {
    // No-op for in-process communication - connections are set up by the coordinator
  }
};

template <typename T = float> class InProcessPipelineCoordinator : public PipelineCoordinator<T> {
public:
  InProcessPipelineCoordinator(tnn::Sequential<T> model, int num_stages = 4,
                               int num_microbatches = 4)
      : PipelineCoordinator<T>(num_stages, num_microbatches, std::move(model)) {

    if (model.get_layers().size() < static_cast<size_t>(num_stages)) {
      throw std::invalid_argument("Model must have at least as many layers as stages");
    }

    auto partitions = tnn::NaivePartitioner::get_partitions(model.get_layers(), num_stages);
    auto splitted_models = model.split(partitions);

    this->coordinator_comm_ = std::make_shared<InProcessPipelineCommunicator<T>>();

    for (int i = 0; i < this->num_stages_; ++i) {
      this->stage_names_.push_back("stage_" + std::to_string(i));
    }

    std::vector<std::shared_ptr<PipelineCommunicator<T>>> stage_communicators;

    for (int i = 0; i < this->num_stages_; ++i) {

      splitted_models[i].enable_profiling(true);
      auto model_ptr = std::make_unique<tnn::Sequential<T>>(std::move(splitted_models[i]));

      auto stage_communicator = std::make_shared<InProcessPipelineCommunicator<T>>();
      stage_communicators.push_back(stage_communicator);

      auto stage_comm_unique =
          std::unique_ptr<PipelineCommunicator<T>, std::function<void(PipelineCommunicator<T> *)>>(
              stage_communicator.get(), [](PipelineCommunicator<T> *) {});

      auto stage = std::make_unique<InProcessPipelineStage<T>>(
          std::move(model_ptr), std::move(stage_comm_unique), this->stage_names_[i]);

      temp_stages_.emplace_back(std::move(stage));
    }

    setup_communication_network(stage_communicators);

    this->add_message_callback();

    std::cout << "Pipeline coordinator initialized with " << this->num_stages_ << " stages"
              << std::endl;
  }

  std::vector<std::unique_ptr<PipelineStage<T>>> get_stages() {
    // Convert InProcessPipelineStage to base PipelineStage
    std::vector<std::unique_ptr<PipelineStage<T>>> stages;
    for (auto &stage : temp_stages_) {
      stages.emplace_back(std::move(stage));
    }
    temp_stages_.clear();
    return stages;
  }

private:
  std::vector<std::unique_ptr<InProcessPipelineStage<T>>> temp_stages_;
  std::vector<std::shared_ptr<PipelineCommunicator<T>>> stage_comm_refs_;

  void setup_communication_network(
      const std::vector<std::shared_ptr<PipelineCommunicator<T>>> &stage_comms) {

    stage_comm_refs_ = stage_comms;

    auto coordinator_comm_shared =
        std::static_pointer_cast<InProcessPipelineCommunicator<T>>(this->coordinator_comm_);

    for (int i = 0; i < this->num_stages_; ++i) {
      auto stage_comm = std::static_pointer_cast<InProcessPipelineCommunicator<T>>(stage_comms[i]);
      stage_comm->register_communicator("coordinator", this->coordinator_comm_);
    }

    for (int i = 0; i < this->num_stages_; ++i) {
      auto current_comm =
          std::static_pointer_cast<InProcessPipelineCommunicator<T>>(stage_comms[i]);

      if (i > 0) {
        current_comm->register_communicator("prev_stage", stage_comms[i - 1]);
        current_comm->register_recipient("prev_stage",
                                         StageEndpoint::in_process(this->stage_names_[i - 1]));
      } else {

        current_comm->register_communicator("prev_stage", this->coordinator_comm_);
        current_comm->register_recipient("prev_stage", StageEndpoint::in_process("coordinator"));
      }

      if (i < this->num_stages_ - 1) {
        current_comm->register_communicator("next_stage", stage_comms[i + 1]);
        current_comm->register_recipient("next_stage",
                                         StageEndpoint::in_process(this->stage_names_[i + 1]));
      } else {

        current_comm->register_communicator("next_stage", this->coordinator_comm_);
        current_comm->register_recipient("next_stage", StageEndpoint::in_process("coordinator"));
      }

      current_comm->register_recipient("coordinator", StageEndpoint::in_process("coordinator"));
    }

    for (int i = 0; i < this->num_stages_; ++i) {
      coordinator_comm_shared->register_communicator(this->stage_names_[i], stage_comms[i]);
      coordinator_comm_shared->register_recipient(this->stage_names_[i],
                                                  StageEndpoint::in_process(this->stage_names_[i]));
    }
  }

  void setup_stage_connections(const StageConfig &config) {
    // No-op for in-process communication
  }
};

} // namespace tpipeline