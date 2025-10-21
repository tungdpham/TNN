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
  InProcessPipelineStage(std::unique_ptr<Communicator> communicator)
      : PipelineStage(nullptr, std::move(communicator), "") {}
};

class InProcessCoordinator : public Coordinator {
public:
  InProcessCoordinator(tnn::Sequential<float> model, const size_t num_stages)
      : Coordinator(std::move(model)) {
    // Initialize in-process communicator for the coordinator
    this->coordinator_comm_ = std::make_unique<InProcessCommunicator>();
    this->add_message_callback();

    this->coordinator_endpoint_ = Endpoint::in_process(this->coordinator_comm_.get());

    // Initialize in-process communicators for each stage
    std::vector<std::unique_ptr<InProcessCommunicator>> stage_comms(num_stages);
    for (size_t i = 0; i < num_stages; ++i) {
      stage_comms[i] = std::make_unique<InProcessCommunicator>();
    }
    temp_stages_.resize(num_stages);
    // Initialize remote endpoints and stages
    for (size_t i = 0; i < num_stages; ++i) {
      this->remote_endpoints_[i] = Endpoint::in_process(stage_comms[i].get());
      temp_stages_[i] = std::make_unique<InProcessPipelineStage>(std::move(stage_comms[i]));
    }
    this->num_stages_ = static_cast<int>(num_stages);
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
};

} // namespace tpipeline