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

namespace synet {

struct RoCEConfig {
  std::string host = "";
  int port = 0;
  std::string local_worker_host = "";
  int local_worker_port = 0;
  std::string device_name = "";
  int gid_index = -1;
  uint64_t master_slab_size = 1024ULL * 1024 * 1024;  // 1 GB
  uint32_t num_io_threads = 4;
  Vec<Endpoint> worker_endpoints;
  Vec<size_t> partition_ratios;

  void load_from_json(const std::string &config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open RoCE config file: " + config_path);
    }
    nlohmann::json j;
    file >> j;

    host = j.value("host", host);
    port = j.value("port", port);
    local_worker_host = j.value("local_worker_host", local_worker_host);
    local_worker_port = j.value("local_worker_port", local_worker_port);
    device_name = j.value("device_name", device_name);
    gid_index = j.value("gid_index", gid_index);
    master_slab_size = j.value("master_slab_size", master_slab_size);
    num_io_threads = j.value("num_io_threads", num_io_threads);
    if (j.contains("worker_endpoints")) {
      for (const auto &ep_json : j["worker_endpoints"]) {
        worker_endpoints.push_back(Endpoint::from_json(ep_json));
      }
    }
    if (j.contains("partition_ratios")) {
      partition_ratios = j["partition_ratios"].get<Vec<size_t>>();
    }
  }
};

/**
 * @brief Distributed pipeline coordinator for RoCE-based stage deployment
 *
 * Handles deployment of pipeline stages to remote machines, establishes
 * RDMA communication topology, and coordinates distributed training.
 */
class RoCECoordinator : public Coordinator {
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
  RoCECoordinator(RoCEConfig roce_config, CoordinatorConfig coordinator_config)
      : Coordinator(std::move(coordinator_config.model), std::move(coordinator_config.optimizer),
                    std::move(coordinator_config.scheduler),
                    std::move(coordinator_config.partitioner),
                    std::move(coordinator_config.local_worker),
                    Endpoint::roce(roce_config.host, roce_config.port),
                    std::move(coordinator_config.worker_endpoints), ParallelMode_t::PIPELINE) {
    auto communicator = RoCECommunicator::create(
        this->coordinator_endpoint_,
        RoCECommunicator::Config{roce_config.master_slab_size, roce_config.num_io_threads});

    communicator->start_server();
    this->comm_ = std::move(communicator);
    this->add_message_callback();
  }

  ~RoCECoordinator() = default;
};

}  // namespace synet
