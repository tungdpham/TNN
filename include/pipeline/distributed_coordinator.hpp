/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "coordinator.hpp"
#include "endpoint.hpp"
#include "network_serialization.hpp"
#include "nn/sequential.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "tcp_communicator.hpp"
#include <asio.hpp>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace tpipeline {

/**
 * @brief Distributed pipeline coordinator for network-based stage deployment
 *
 * Handles deployment of pipeline stages to remote machines, establishes
 * network communication topology, and coordinates distributed training.
 */
class DistributedCoordinator : public Coordinator {
public:
  DistributedCoordinator(tnn::Sequential<float> model,
                         Endpoint coordinator_endpoint = Endpoint::network("localhost", 8000),
                         const std::vector<Endpoint> &endpoints = {})
      : Coordinator(std::move(model)) {
    // Initialize coordinator and remote endpoints
    this->coordinator_endpoint_ = coordinator_endpoint;
    this->remote_endpoints_ = endpoints;
    this->num_stages_ = static_cast<int>(endpoints.size());

    // Initialize TCP communicator for the coordinator
    this->coordinator_comm_ = std::make_unique<TcpCommunicator>(coordinator_endpoint_);
    this->add_message_callback();
  }

  ~DistributedCoordinator() = default;
};

} // namespace tpipeline
