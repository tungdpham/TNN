/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <memory>

#include "coordinator.hpp"
#include "roce_communicator.hpp"

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
  RoceCoordinator(CoordinatorConfig config)
      : Coordinator(std::move(config)) {
    // Initialize RoCE communicator for the coordinator
    auto &allocator = PoolAllocator::instance(getCPU());
    auto communicator = std::make_unique<RoceCommunicator>(this->coordinator_endpoint_, allocator);
    communicator->start_server();
    this->comm_ = std::move(communicator);
    this->add_message_callback();
  }

  ~RoceCoordinator() = default;
};

}  // namespace tnn
