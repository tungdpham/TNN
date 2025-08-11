#pragma once

#include "pipeline_communicator.hpp"

namespace tpipeline {

template <typename T>
void InProcessPipelineCommunicator<T>::send_output_message() {
  std::lock_guard<std::mutex> lock(this->out_message_mutex_);
  if (!this->out_message_queue_.empty()) {
    tpipeline::Message<T> message = this->out_message_queue_.front();
    this->out_message_queue_.pop();

    auto send_to_endpoint = [&](const StageEndpoint &endpoint) {
      if (!endpoint.address.empty()) {
        auto details =
            std::any_cast<tpipeline::InProcessStageEndpointDetails<T>>(
                endpoint.details);
        if (details.communicator) {
          details.communicator->enqueue_input_message(message);
        }
      }
    };

    // Route message based on command type
    if (message.command_type == CommandType::FORWARD_TASK) {
      send_to_endpoint(this->next_stage_endpoint_);
    } else if (message.command_type == CommandType::BACKWARD_TASK) {
      send_to_endpoint(this->prev_stage_endpoint_);
    } else {
      // For control messages, send to both endpoints if relevant
      // This can be customized based on your specific routing needs
      if (message.recipient_id.empty()) {
        // Broadcast to next stage by default for control messages
        send_to_endpoint(this->next_stage_endpoint_);
      } else {
        // Send to specific recipient based on recipient_id
        // This would require a more sophisticated routing mechanism
        // For now, default to next stage
        send_to_endpoint(this->next_stage_endpoint_);
      }
    }
  }
}

} // namespace tpipeline
