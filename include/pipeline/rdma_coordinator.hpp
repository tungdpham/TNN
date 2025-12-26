/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "coordinator.hpp"
#include "endpoint.hpp"
#include "nn/sequential.hpp"
#include "rdma_communicator.hpp"
#include <memory>
#include <string>
#include <vector>

namespace tnn {

/**
 * @brief Distributed pipeline coordinator using RDMA
 *
 * Handles deployment of pipeline stages to remote machines using RDMA communication.
 */
class RdmaNetworkCoordinator : public Coordinator {
public:
  /**
   * @brief Constructor for RDMA distributed coordinator
   * @param model The neural network model to distribute
   * @param optimizer The optimizer
   * @param coordinator_endpoint The endpoint for the coordinator
   * @param endpoints The list of worker endpoints
   */
  RdmaNetworkCoordinator(Sequential<float> model, std::unique_ptr<Optimizer<float>> optimizer,
                         Endpoint coordinator_endpoint, const std::vector<Endpoint> &endpoints = {})
      : Coordinator(std::move(model), std::move(optimizer)) {
    // Initialize coordinator and remote endpoints
    this->coordinator_endpoint_ = coordinator_endpoint;
    this->remote_endpoints_ = endpoints;
    this->num_stages_ = static_cast<int>(endpoints.size());

    // Initialize RDMA communicator for the coordinator
    auto communicator = std::make_unique<RdmaCommunicator>(coordinator_endpoint);
    this->coordinator_comm_ = std::move(communicator);
    this->add_message_callback();
  }

  ~RdmaNetworkCoordinator() = default;
};

} // namespace tnn
