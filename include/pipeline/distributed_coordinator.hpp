/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "network_serialization.hpp"
#include "nn/partitioner.hpp"
#include "nn/sequential.hpp"
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
template <typename T = float> class DistributedPipelineCoordinator : public PipelineCoordinator<T> {
public:
  struct RemoteEndpoint {
    std::string host;
    int port;
    std::string stage_id;

    RemoteEndpoint(std::string h, int p, std::string id)
        : host(std::move(h)), port(p), stage_id(std::move(id)) {}
  };

  DistributedPipelineCoordinator(tnn::Sequential<T> model,
                                 const std::vector<RemoteEndpoint> &endpoints,
                                 int num_microbatches = 4,
                                 const std::string &coordinator_host = "localhost",
                                 int coordinator_port = 8000)
      : PipelineCoordinator<T>(endpoints.size(), num_microbatches, std::move(model)),
        remote_endpoints_(endpoints), coordinator_port_(coordinator_port), io_context_(),
        work_guard_(asio::make_work_guard(io_context_)), is_deployed_(false) {

    if (this->model_.get_layers().size() < endpoints.size()) {
      std::cout << "Error: Model has fewer layers (" << this->model_.get_layers().size()
                << ") than remote endpoints (" << endpoints.size() << ")\n";
      throw std::invalid_argument("Model must have at least as many layers as remote endpoints");
    }

    this->partitions_ = tnn::NaivePartitioner::get_partitions(
        this->model_.get_layers(), static_cast<size_t>(endpoints.size()));

    auto splitted_models = this->model_.split(this->partitions_);

    for (size_t i = 0; i < endpoints.size(); ++i) {
      StageConfig config;
      config.stage_id = endpoints[i].stage_id;
      config.stage_index = static_cast<int>(i);
      config.model_config = splitted_models[i].get_config();
      config.model_config["name"] = endpoints[i].stage_id;
      config.coordinator_endpoint = coordinator_host + ":" + std::to_string(coordinator_port_);

      if (i > 0) {
        config.prev_stage_endpoint =
            endpoints[i - 1].host + ":" + std::to_string(endpoints[i - 1].port);
      } else {

        config.prev_stage_endpoint = coordinator_host + ":" + std::to_string(coordinator_port_);
      }
      if (i < endpoints.size() - 1) {
        config.next_stage_endpoint =
            endpoints[i + 1].host + ":" + std::to_string(endpoints[i + 1].port);
      } else {
        config.next_stage_endpoint = coordinator_host + ":" + std::to_string(coordinator_port_);
      }

      stage_configs_.push_back(config);
      this->stage_names_.push_back(config.stage_id);
    }

    this->coordinator_comm_ = std::make_unique<TcpPipelineCommunicator<T>>(
        io_context_, coordinator_host, coordinator_port_);

    this->add_message_callback();

    io_thread_ = std::thread([this]() { io_context_.run(); });

    std::cout << "Distributed coordinator initialized with " << endpoints.size()
              << " remote endpoints\n";
  }

  ~DistributedPipelineCoordinator() {
    this->stop();

    this->should_stop_ = true;
    this->message_notification_cv_.notify_all();

    work_guard_.reset();
    io_context_.stop();
    if (io_thread_.joinable()) {
      io_thread_.join();
    }

    this->coordinator_comm_.reset();
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

    std::vector<std::future<bool>> params_futures;

    for (size_t i = 0; i < remote_endpoints_.size(); ++i) {
      auto future = std::async(std::launch::async, [this, i]() {
        return this->send_params(remote_endpoints_[i].stage_id, this->partitions_[i]);
      });
      params_futures.push_back(std::move(future));
    }

    bool all_params_sent = true;

    for (auto &future : params_futures) {
      if (!future.get()) {
        all_params_sent = false;
      }
    }
    if (!all_params_sent) {
      std::cout << "Failed to send parameters to all stages\n";
      return false;
    }

    if (!this->wait_for_params_loaded()) {
      std::cerr << "Failed to receive parameters confirmation from all stages\n";
      return false;
    }

    is_deployed_ = true;
    std::cout << "All stages deployed and ready!\n";
    return true;
  }

  bool is_deployed() const { return is_deployed_; }

private:
  std::vector<RemoteEndpoint> remote_endpoints_;
  std::vector<StageConfig> stage_configs_;
  int coordinator_port_;

  asio::io_context io_context_;
  asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
  std::thread io_thread_;

  std::atomic<bool> is_deployed_;

  bool connect_to_endpoint(const RemoteEndpoint &endpoint) {
    try {
      std::cout << "Connecting to stage " << endpoint.stage_id << " at " << endpoint.host << ":"
                << endpoint.port << std::endl;
      auto tcp_comm = static_cast<TcpPipelineCommunicator<T> *>(this->coordinator_comm_.get());
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
      auto config_msg = Message<T>::create_text_message(CommandType::CONFIG_TRANSFER, config_json,
                                                        "coordinator", stage_id);

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
