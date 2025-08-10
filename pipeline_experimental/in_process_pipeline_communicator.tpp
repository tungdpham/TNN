#pragma once

#include "pipeline_communicator.hpp"

namespace tpipeline {

template <typename T>
void InProcessPipelineCommunicator<T>::send_output_task() {
  std::lock_guard<std::mutex> lock(this->out_task_mutex_);
  if (!this->out_task_queue_.empty()) {
    tpipeline::Task<T> task = this->out_task_queue_.front();
    this->out_task_queue_.pop();

    auto send_to_endpoint = [&](const StageEndpoint &endpoint) {
      if (!endpoint.address.empty()) {
        auto details =
            std::any_cast<tpipeline::InProcessStageEndpointDetails<T>>(
                endpoint.details);
        if (details.communicator) {
          details.communicator->enqueue_input_task(task);
        }
      }
    };

    if (task.type == tpipeline::TaskType::Forward) {
      send_to_endpoint(this->next_stage_endpoint_);
    } else if (task.type == tpipeline::TaskType::Backward) {
      send_to_endpoint(this->prev_stage_endpoint_);
    }
  }
}

} // namespace tpipeline
