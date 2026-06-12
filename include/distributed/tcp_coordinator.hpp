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

namespace synet {

struct TCPConfig {
  std::string host = "";
  int port = 0;
  std::string local_worker_host = "";
  int local_worker_port = 0;
  uint32_t num_io_threads = 4;
  uint32_t max_packet_size = 4 * 1024 * 1024;  // 4 MB
  uint32_t skts_per_endpoint = 1;
  Vec<Endpoint> worker_endpoints = {};
  Vec<size_t> partition_ratios;

  void load_from_json(const std::string &config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open config file: " + config_path);
    }
    nlohmann::json j;
    file >> j;

    host = j.value("host", host);
    port = j.value("port", port);
    local_worker_host = j.value("local_worker_host", local_worker_host);
    local_worker_port = j.value("local_worker_port", local_worker_port);
    num_io_threads = j.value("num_io_threads", num_io_threads);
    max_packet_size = j.value("max_packet_size", max_packet_size);
    skts_per_endpoint = j.value("skts_per_endpoint", skts_per_endpoint);
    if (j.contains("worker_endpoints")) {
      for (const auto &ep_json : j["worker_endpoints"]) {
        worker_endpoints.push_back(Endpoint::from_json(ep_json));
      }
    } else {
      std::cerr << "Warning: No worker_endpoints specified in config file." << std::endl;
    }
    if (j.contains("partition_ratios")) {
      partition_ratios = j["partition_ratios"].get<Vec<size_t>>();
    } else {
      throw std::runtime_error("TCP config must contain partition_ratios");
    }
  }
};

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
  NetworkCoordinator(TCPConfig tcp_config, CoordinatorConfig coordinator_config)
      : Coordinator(std::move(coordinator_config.model), std::move(coordinator_config.optimizer),
                    std::move(coordinator_config.scheduler),
                    std::move(coordinator_config.partitioner),
                    std::move(coordinator_config.local_worker),
                    Endpoint::tcp(tcp_config.host, tcp_config.port),
                    std::move(coordinator_config.worker_endpoints), ParallelMode_t::PIPELINE) {
    auto &allocator = PoolAllocator::instance(getHost(), defaultFlowHandle);
    // Initialize TCP communicator for the coordinator
    auto communicator =
        std::make_unique<TCPCommunicator>(this->coordinator_endpoint_, allocator,
                                          TCPCommunicator::Config{
                                              .num_io_threads = tcp_config.num_io_threads,
                                              .max_packet_size = tcp_config.max_packet_size,
                                              .skts_per_endpoint = tcp_config.skts_per_endpoint,
                                          });
    communicator->start_server();
    this->comm_ = std::move(communicator);
    this->add_message_callback();
  }

  ~NetworkCoordinator() = default;
};

}  // namespace synet
