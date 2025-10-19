/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "network_serialization.hpp"
#include "nn/sequential.hpp"
#include "partitioner/naive_partitioner.hpp"
#include "pipeline_coordinator.hpp"
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
class DistributedPipelineCoordinator : public PipelineCoordinator {
public:
  struct RemoteEndpoint {
    std::string host;
    int port;
    std::string stage_id;

    RemoteEndpoint(std::string h, int p, std::string id)
        : host(std::move(h)), port(p), stage_id(std::move(id)) {}
  };

  DistributedPipelineCoordinator(tnn::Sequential<float> model,
                                 const std::vector<RemoteEndpoint> &endpoints,
                                 int num_microbatches = 4,
                                 const std::string &coordinator_host = "localhost",
                                 int coordinator_port = 8000)
      : PipelineCoordinator(endpoints.size(), num_microbatches, std::move(model)),
        remote_endpoints_(endpoints), coordinator_host_(coordinator_host),
        coordinator_port_(coordinator_port), is_deployed_(false) {}

  ~DistributedPipelineCoordinator() {
    this->stop();

    this->should_stop_ = true;
    this->message_notification_cv_.notify_all();

    this->coordinator_comm_.reset();
  }

  void initialize_communicator() override {
    // Communicator now manages its own io_context, work_guard, and io_thread
    this->coordinator_comm_ =
        std::make_unique<TcpPipelineCommunicator>(coordinator_host_, coordinator_port_);

    this->add_message_callback();
  }

  void initialize_topology() override {
    auto splitted_model = this->model_.split(this->partitions_);

    for (size_t i = 0; i < remote_endpoints_.size(); ++i) {
      StageConfig config;
      config.stage_id = remote_endpoints_[i].stage_id;
      config.model_config = splitted_model[i].get_config();
      config.model_config["name"] = remote_endpoints_[i].stage_id;
      config.coordinator_endpoint = coordinator_host_ + ":" + std::to_string(coordinator_port_);

      if (i > 0) {
        config.prev_stage_endpoint =
            remote_endpoints_[i - 1].host + ":" + std::to_string(remote_endpoints_[i - 1].port);
      } else {

        config.prev_stage_endpoint = coordinator_host_ + ":" + std::to_string(coordinator_port_);
      }
      if (i < remote_endpoints_.size() - 1) {
        config.next_stage_endpoint =
            remote_endpoints_[i + 1].host + ":" + std::to_string(remote_endpoints_[i + 1].port);
      } else {
        config.next_stage_endpoint = coordinator_host_ + ":" + std::to_string(coordinator_port_);
      }

      stage_configs_.push_back(config);
      this->stage_names_.push_back(config.stage_id);
    }
  }

  bool deploy_stages() {
    if (is_deployed_) {
      std::cout << "Stages already deployed\n";
      return true;
    }

    std::cout << "Starting deployment of distributed pipeline stages...\n";

    std::vector<std::future<bool>> connection_futures;

    for (const auto &endpoint : remote_endpoints_) {
      auto future = std::async(std::launch::async,
                               [this, &endpoint]() { return connect_to_endpoint(endpoint); });
      connection_futures.push_back(std::move(future));
    }

    bool all_connected = true;
    for (auto &future : connection_futures) {
      if (!future.get()) {
        all_connected = false;
      }
    }

    if (!all_connected) {
      std::cout << "Failed to connect to all endpoints\n";
      return false;
    }

    std::cout << "Connected to all endpoints, sending stage configurations...\n" << std::endl;

    std::vector<std::future<bool>> deployment_futures;

    for (size_t i = 0; i < remote_endpoints_.size(); ++i) {
      auto future = std::async(std::launch::async, [this, i]() {
        return deploy_stage_config(remote_endpoints_[i].stage_id, stage_configs_[i]);
      });
      deployment_futures.push_back(std::move(future));
    }

    bool all_deployed = true;
    for (auto &future : deployment_futures) {
      if (!future.get()) {
        all_deployed = false;
      }
    }

    if (!all_deployed) {
      std::cout << "Failed to deploy all stages\n";
      return false;
    }

    if (!this->wait_for_config_received()) {
      std::cerr << "Not all stages reported ready\n";
      return false;
    }

    is_deployed_ = true;
    std::cout << "All stages deployed and ready!\n";
    // clear model
    this->model_.clear();
    return true;
  }

  bool is_deployed() const { return is_deployed_; }

private:
  std::vector<RemoteEndpoint> remote_endpoints_;
  std::vector<StageConfig> stage_configs_;
  std::string coordinator_host_;
  int coordinator_port_;

  std::atomic<bool> is_deployed_;

  bool connect_to_endpoint(const RemoteEndpoint &endpoint) {
    try {
      std::cout << "Connecting to stage " << endpoint.stage_id << " at " << endpoint.host << ":"
                << endpoint.port << std::endl;
      auto tcp_comm = static_cast<TcpPipelineCommunicator *>(this->coordinator_comm_.get());
      bool connected = tcp_comm->connect_to_peer(endpoint.stage_id, endpoint.host, endpoint.port);

      if (connected) {
        std::cout << "Connected to stage " << endpoint.stage_id << " at " << endpoint.host << ":"
                  << endpoint.port << std::endl;
      }

      return connected;

    } catch (const std::exception &e) {
      std::cout << "Failed to connect to " << endpoint.host << ":" << endpoint.port << " - "
                << e.what() << '\n';
      return false;
    }
  }

  bool deploy_stage_config(const std::string &stage_id, const StageConfig &config) {
    try {

      std::string config_json = config.to_json().dump();
      auto config_msg = Message(stage_id, CommandType::CONFIG_TRANSFER, config_json);

      this->coordinator_comm_->send_message(config_msg);

      std::cout << "Sent configuration to stage " << stage_id << '\n';

      return true;
    } catch (const std::exception &e) {
      std::cout << "Failed to deploy config to stage " << stage_id << ": " << e.what() << '\n';
      return false;
    }
  }
};

} // namespace tpipeline
