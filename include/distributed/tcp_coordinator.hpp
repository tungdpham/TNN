/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <asio.hpp>
#include <memory>

#include "coordinator.hpp"
#include "tcp_communicator.hpp"

namespace tnn {

/**
 * @brief Distributed pipeline coordinator for network-based stage deployment
 *
 * Handles deployment of pipeline stages to remote machines, establishes
 * network communication topology, and coordinates distributed training.
 */
class NetworkCoordinator : public Coordinator {
public:
  /**
   * @brief Constructor for distributed coordinator
   * @param model The neural network model to distribute
   * @param coordinator_endpoint The endpoint for the coordinator
   * @param endpoints The list of worker endpoints
   * @param io_threads Number of IO threads for the TCP communicator (default: 1)
   */
  NetworkCoordinator(CoordinatorConfig config,
                     TCPCommunicator::Config tcp_config = TCPCommunicator::Config())
      : Coordinator(std::move(config)) {
    auto &allocator = PoolAllocator::instance(getCPU());
    // Initialize TCP communicator for the coordinator
    auto communicator =
        std::make_unique<TCPCommunicator>(this->coordinator_endpoint_, allocator, tcp_config);
    communicator->start_server();
    this->comm_ = std::move(communicator);
    this->add_message_callback();
  }

  ~NetworkCoordinator() = default;
};

}  // namespace tnn
