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
#include "utils/hardware_info.hpp"
#include "utils/thread_affinity.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <future>
#include <iostream>
#include <memory>

namespace tpipeline {

/**
 * @brief Network-based pipeline stage worker
 *
 * Standalone worker process that listens for stage configurations
 * from a coordinator and processes distributed pipeline tasks.
 */
template <typename T = float> class NetworkStageWorker : public PipelineStage {
public:
  /**
   * @brief Constructor with optional thread affinity configuration
   * @param listen_port Port to listen on for connections
   * @param use_ecore_affinity Whether to bind worker threads to E-cores for efficiency
   * @param max_ecore_threads Maximum number of E-cores to use (-1 for all available)
   */
  explicit NetworkStageWorker(int listen_port, bool use_ecore_affinity = false,
                              int max_ecore_threads = -1)
      : PipelineStage(), listen_port_(listen_port), use_ecore_affinity_(use_ecore_affinity),
        max_ecore_threads_(max_ecore_threads) {

    // Initialize hardware info for affinity
    if (use_ecore_affinity_) {
      hw_info_.initialize();
      thread_affinity_ = std::make_unique<utils::ThreadAffinity>(hw_info_);

      if (!thread_affinity_->has_efficiency_cores()) {
        std::cout << "Warning: E-core affinity requested but no E-cores detected. "
                  << "Will use P-cores instead." << std::endl;
        use_ecore_affinity_ = false;
      } else {
        std::cout << "E-core affinity enabled. Available E-cores: "
                  << thread_affinity_->get_efficiency_core_count() << std::endl;
      }
    }

    tcp_communicator_ = std::make_unique<TcpCommunicator>("localhost", listen_port_);

    this->communicator_ = std::unique_ptr<Communicator, std::function<void(Communicator *)>>(
        tcp_communicator_.get(), [](Communicator *) {});
  }

  ~NetworkStageWorker() { stop(); }

  /**
   * @brief Enable or disable E-core affinity at runtime
   * @param enable Whether to enable E-core affinity
   * @param max_threads Maximum number of E-cores to use (-1 for all)
   */
  void set_ecore_affinity(bool enable, int max_threads = -1) {
    use_ecore_affinity_ = enable;
    max_ecore_threads_ = max_threads;

    if (enable && !thread_affinity_) {
      hw_info_.initialize();
      thread_affinity_ = std::make_unique<utils::ThreadAffinity>(hw_info_);
    }
  }

  /**
   * @brief Get hardware information
   * @return Reference to hardware info object
   */
  const utils::HardwareInfo &get_hardware_info() const { return hw_info_; }

  /**
   * @brief Print affinity information for debugging
   */
  void print_affinity_info() const {
    if (thread_affinity_) {
      thread_affinity_->print_affinity_info();
    } else {
      std::cout << "Thread affinity not configured" << std::endl;
    }
  }

  void start() override {
    if (!this->should_stop_)
      return;

    PipelineStage::start();

    std::cout << "Starting network stage worker on port " << listen_port_ << '\n';
    std::cout << "Network stage worker listening on port " << listen_port_ << '\n';
  }

  void stop() override {
    std::cout << "Stopping network stage worker." << '\n';

    PipelineStage::stop();

    if (tcp_communicator_) {
      tcp_communicator_->stop();
    }

    std::cout << "Network stage worker stopped" << '\n';
  }

private:
  int listen_port_;
  bool use_ecore_affinity_;
  int max_ecore_threads_;
  utils::HardwareInfo hw_info_;
  std::unique_ptr<utils::ThreadAffinity> thread_affinity_;
  std::unique_ptr<TcpCommunicator> tcp_communicator_;

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
  /**
   * @brief Run network worker with E-core affinity configuration
   * @param listen_port Port to listen on
   * @param use_ecore_affinity Whether to bind to E-cores
   * @param max_ecore_threads Maximum E-cores to use (-1 for all)
   * @return Exit code (0 for success)
   */
  static int run_worker(int listen_port, bool use_ecore_affinity, int max_ecore_threads = -1) {
    try {
      NetworkStageWorker worker(listen_port, use_ecore_affinity, max_ecore_threads);

      std::signal(SIGINT, [](int) {
        std::cout << '\n' << "Received interrupt signal, shutting down..." << '\n';
        std::exit(0);
      });

      if (use_ecore_affinity) {
        std::cout << "Network stage worker configured with E-core affinity" << std::endl;
        worker.print_affinity_info();
      }

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
