#pragma once

#include <any>
#include <string>

namespace tpipeline {

// Forward declaration
template <typename T> class PipelineCommunicator;

/**
 * @brief A generic endpoint for a pipeline stage.
 *
 * This struct holds the information needed to send a task to another stage.
 * It is designed to be abstract, allowing different communication methods
 * (in-process, network, etc.) to store their specific details.
 */
struct StageEndpoint {
  // A unique identifier for the endpoint, e.g., "stage_1_comm",
  // "192.168.1.10:8080"
  std::string address;

  // A type-erased container to hold communication-specific details.
  // For in-process, this might hold a pointer. For network, a config struct.
  std::any details;

  StageEndpoint() = default;
  StageEndpoint(std::string addr, std::any d)
      : address(std::move(addr)), details(std::move(d)) {}
};

/**
 * @brief A concrete details structure for in-process communication.
 *
 * This will be stored in the `details` field of a `StageEndpoint` when
 * communicating between stages within the same process.
 */
template <typename T> struct InProcessStageEndpointDetails {
  PipelineCommunicator<T> *communicator = nullptr;
};

} // namespace tpipeline

