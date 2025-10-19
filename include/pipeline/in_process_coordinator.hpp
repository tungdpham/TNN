/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include "coordinator.hpp"
#include "in_process_communicator.hpp"

namespace tpipeline {

// Concrete implementation of PipelineStage for in-process communication
class InProcessPipelineStage : public PipelineStage {
public:
  InProcessPipelineStage(std::unique_ptr<tnn::Sequential<float>> model,
                         std::unique_ptr<Communicator> communicator, const std::string &name)
      : PipelineStage(std::move(model), std::move(communicator), name) {}

protected:
  void setup_stage_connections(const StageConfig &config) override {
    // No-op for in-process communication - connections are set up by the coordinator
  }
};

class InProcessCoordinator : public Coordinator {
public:
  InProcessCoordinator(tnn::Sequential<float> model, int num_stages = 4, int num_microbatches = 4)
      : Coordinator(num_stages, num_microbatches, std::move(model)) {

    if (model.get_layers().size() < static_cast<size_t>(num_stages)) {
      throw std::invalid_argument("Model must have at least as many layers as stages");
    }

    if (this->partitioner_ == nullptr) {
      throw std::runtime_error("Partitioner is not set");
    }

    auto partitions = partitioner_->get_partitions(this->model_.get_layers(), this->num_stages_);
    auto splitted_models = model.split(partitions);

    this->coordinator_comm_ = std::make_shared<InProcessCommunicator>();

    for (int i = 0; i < this->num_stages_; ++i) {
      this->stage_names_.push_back("stage_" + std::to_string(i));
    }

    std::vector<std::shared_ptr<Communicator>> stage_communicators;

    for (int i = 0; i < this->num_stages_; ++i) {

      splitted_models[i].enable_profiling(true);
      auto model_ptr = std::make_unique<tnn::Sequential<float>>(std::move(splitted_models[i]));

      auto stage_communicator = std::make_shared<InProcessCommunicator>();
      stage_communicators.push_back(stage_communicator);

      auto stage_comm_unique = std::unique_ptr<Communicator>(stage_communicator.get());

      auto stage = std::make_unique<InProcessPipelineStage>(
          std::move(model_ptr), std::move(stage_comm_unique), this->stage_names_[i]);

      temp_stages_.emplace_back(std::move(stage));
    }

    setup_communication_network(stage_communicators);

    this->add_message_callback();

    std::cout << "Pipeline coordinator initialized with " << this->num_stages_ << " stages"
              << std::endl;
  }

  std::vector<std::unique_ptr<PipelineStage>> get_stages() {
    // Convert InProcessPipelineStage to base PipelineStage
    std::vector<std::unique_ptr<PipelineStage>> stages;
    for (auto &stage : temp_stages_) {
      stages.emplace_back(std::move(stage));
    }
    temp_stages_.clear();
    return stages;
  }

private:
  std::vector<std::unique_ptr<InProcessPipelineStage>> temp_stages_;
  std::vector<std::shared_ptr<Communicator>> stage_comm_refs_;

  void setup_communication_network(const std::vector<std::shared_ptr<Communicator>> &stage_comms) {

    stage_comm_refs_ = stage_comms;

    auto coordinator_comm_shared =
        std::static_pointer_cast<InProcessCommunicator>(this->coordinator_comm_);

    for (int i = 0; i < this->num_stages_; ++i) {
      auto stage_comm = std::static_pointer_cast<InProcessCommunicator>(stage_comms[i]);
      stage_comm->register_communicator("coordinator", this->coordinator_comm_);
    }

    for (int i = 0; i < this->num_stages_; ++i) {
      auto current_comm = std::static_pointer_cast<InProcessCommunicator>(stage_comms[i]);

      if (i > 0) {
        current_comm->register_communicator("prev_stage", stage_comms[i - 1]);
        current_comm->connect("prev_stage", Endpoint::in_process());
      } else {
        current_comm->register_communicator("prev_stage", this->coordinator_comm_);
        current_comm->connect("prev_stage", Endpoint::in_process());
      }

      if (i < this->num_stages_ - 1) {
        current_comm->register_communicator("next_stage", stage_comms[i + 1]);
        current_comm->connect("next_stage", Endpoint::in_process());
      } else {

        current_comm->register_communicator("next_stage", this->coordinator_comm_);
        current_comm->connect("next_stage", Endpoint::in_process());
      }

      current_comm->connect("coordinator", Endpoint::in_process());
    }

    for (int i = 0; i < this->num_stages_; ++i) {
      coordinator_comm_shared->register_communicator(this->stage_names_[i], stage_comms[i]);
      coordinator_comm_shared->connect(this->stage_names_[i], Endpoint::in_process());
    }
  }

  void setup_stage_connections(const StageConfig &config) {
    // No-op for in-process communication
  }
};

} // namespace tpipeline