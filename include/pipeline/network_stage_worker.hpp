/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "asio.hpp"
#include "network_serialization.hpp"
#include "nn/sequential.hpp"
#include "pipeline_stage.hpp"
#include "tcp_communicator.hpp"
#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>

namespace tpipeline {

/**
 * @brief Network-based pipeline stage worker
 *
 * Standalone worker process that listens for stage configurations
 * from a coordinator and processes distributed pipeline tasks.
 */
template <typename T = float> class NetworkStageWorker : public PipelineStage<T> {
public:
  explicit NetworkStageWorker(int listen_port)
      : PipelineStage<T>(), listen_port_(listen_port), io_context_(),
        work_guard_(asio::make_work_guard(io_context_)) {
    tcp_communicator_ =
        std::make_unique<TcpPipelineCommunicator<T>>(io_context_, "localhost", listen_port_);

    this->communicator_ =
        std::unique_ptr<PipelineCommunicator<T>, std::function<void(PipelineCommunicator<T> *)>>(
            tcp_communicator_.get(), [](PipelineCommunicator<T> *) {});
  }

  ~NetworkStageWorker() { stop(); }

  void start() override {
    if (!this->should_stop_)
      return;

    PipelineStage<T>::start();

    std::cout << "Starting network stage worker on port " << listen_port_ << '\n';

    io_thread_ = std::thread([this]() {
// Ignore SIGPIPE to prevent crashes on broken connections
#ifdef __linux__
      std::signal(SIGPIPE, SIG_IGN);
#endif
      io_context_.run();
    });

    std::cout << "Network stage worker listening on port " << listen_port_ << '\n';
  }

  void stop() override {
    if (this->should_stop_)
      return;
    PipelineStage<T>::stop();
    tcp_communicator_->stop();

    work_guard_.reset();
    io_context_.stop();

    if (io_thread_.joinable()) {
      io_thread_.join();
    }

    std::cout << "Network stage worker stopped" << '\n';
  }

private:
  int listen_port_;
  asio::io_context io_context_;
  asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
  std::thread io_thread_;
  std::unique_ptr<TcpPipelineCommunicator<T>> tcp_communicator_;

  void setup_stage_connections(const StageConfig &config) {

    if (!config.coordinator_endpoint.empty()) {
      auto [host, port] = parse_endpoint(config.coordinator_endpoint);
      if (!tcp_communicator_->connect_to_peer("coordinator", host, port)) {
        throw std::runtime_error("Failed to connect to coordinator");
      }
      std::cout << "Connected to coordinator at " << config.coordinator_endpoint << '\n';
    }

    if (!config.next_stage_endpoint.empty()) {
      auto [host, port] = parse_endpoint(config.next_stage_endpoint);
      if (!tcp_communicator_->connect_to_peer("next_stage", host, port)) {
        std::cout << "Warning: Failed to connect to next stage at " << config.next_stage_endpoint
                  << '\n';
      } else {
        std::cout << "Connected to next stage at " << config.next_stage_endpoint << '\n';
      }
    }

    if (!config.prev_stage_endpoint.empty()) {
      auto [host, port] = parse_endpoint(config.prev_stage_endpoint);
      if (!tcp_communicator_->connect_to_peer("prev_stage", host, port)) {
        std::cout << "Warning: Failed to connect to previous stage at "
                  << config.prev_stage_endpoint << '\n';
      } else {
        std::cout << "Connected to previous stage at " << config.prev_stage_endpoint << '\n';
      }
    }
  }

  std::pair<std::string, int> parse_endpoint(const std::string &endpoint) const {
    size_t colon_pos = endpoint.find(':');
    if (colon_pos == std::string::npos) {
      throw std::invalid_argument("Invalid endpoint format: " + endpoint);
    }

    std::string host = endpoint.substr(0, colon_pos);
    int port = std::stoi(endpoint.substr(colon_pos + 1));

    return {host, port};
  }
};

/**
 * @brief Standalone network stage worker application
 *
 * Creates a worker that listens on a specified port and waits for
 * stage configuration from a distributed coordinator.
 */
template <typename T = float> class StandaloneNetworkWorker {
public:
  static int run_worker(int listen_port) {
    try {
      NetworkStageWorker<T> worker(listen_port);

      std::signal(SIGINT, [](int) {
        std::cout << '\n' << "Received interrupt signal, shutting down..." << '\n';
        std::exit(0);
      });

      worker.start();

      std::cout << "Network stage worker started on port " << listen_port << std::endl;

      worker.message_loop();

      return 0;
    } catch (const std::exception &e) {
      std::cout << "Worker failed: " << e.what() << '\n';
      return -1;
    }
  }
};

} // namespace tpipeline
