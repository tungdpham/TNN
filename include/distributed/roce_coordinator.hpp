/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "coordinator.hpp"
#include "device/device_manager.hpp"
#include "endpoint.hpp"
#include "nn/sequential.hpp"
#include "roce_communicator.hpp"
#include <memory>
#include <vector>

namespace tnn {

/**
 * @brief Distributed pipeline coordinator for RoCE-based stage deployment
 *
 * Handles deployment of pipeline stages to remote machines, establishes
 * RDMA communication topology, and coordinates distributed training.
 */
class RoceCoordinator : public Coordinator {
public:
  /**
   * @brief Constructor for distributed coordinator using RoCE
   * @param id Coordinator ID
   * @param model The neural network model to distribute
   * @param optimizer The optimizer
   * @param host Hostname or IP to bind to
   * @param port TCP port for initial connection setup
   * @param device_name IB device name
   * @param gid_index GID index for RoCE
   * @param endpoints The list of worker endpoints
   */
  RoceCoordinator(std::unique_ptr<Sequential> model, std::unique_ptr<Optimizer> optimizer,
                  Endpoint coordinator_endpoint, const std::vector<Endpoint> &endpoints = {})
      : Coordinator(std::move(model), std::move(optimizer)) {

    // Initialize coordinator and remote endpoints
    this->coordinator_endpoint_ = coordinator_endpoint;
    this->worker_endpoints_ = endpoints;
    this->num_stages_ = static_cast<int>(endpoints.size());

    // Initialize RoCE communicator for the coordinator
    // Coordinators typically use CPU, workers use GPU for GPU Direct RDMA
    auto communicator = std::make_unique<RoceCommunicator>(coordinator_endpoint, &getCPU(),
                                                           512 * 1024 * 1024); // 512MB slab
    communicator->start_server();
    this->comm_ = std::move(communicator);
    this->add_message_callback();
  }

  ~RoceCoordinator() = default;
};

} // namespace tnn
