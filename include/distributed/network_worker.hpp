/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "tcp_communicator.hpp"
#include "worker.hpp"
#include <csignal>
#include <iostream>
#include <memory>

namespace tnn {

/**
 * @brief Network-based pipeline stage worker
 *
 * Standalone worker process that listens for stage configurations
 * from a coordinator and processes distributed pipeline jobs.
 */
template <typename T = float> class NetworkStageWorker : public Worker {
public:
  /**
   * @brief Constructor with optional thread affinity configuration
   * @param listen_port Port to listen on for connections
   * @param use_gpu Whether to use GPU for processing
   * @param use_ecore_affinity Whether to bind worker threads to E-cores for efficiency
   * @param max_ecore_threads Maximum number of E-cores to use (-1 for all available)
   * @param io_threads Number of IO threads for the TCP communicator (default: 1)
   */
  explicit NetworkStageWorker(int listen_port, bool use_gpu, size_t io_threads = 1)
      : Worker(use_gpu), listen_port_(listen_port), io_threads_(io_threads) {

    auto communicator = std::make_unique<TcpCommunicator>(
        this->id_, Endpoint::network("localhost", listen_port_), io_threads_);

    communicator->start_server();

    this->communicator_ = std::move(communicator);
  }

  ~NetworkStageWorker() {}

  void start() override {
    if (!this->should_stop_)
      return;

    Worker::start();
  }

  void stop() override {
    std::cout << "Stopping network stage worker." << '\n';

    Worker::stop();

    std::cout << "Network stage worker stopped" << '\n';
  }

private:
  int listen_port_;
  size_t io_threads_;
};
} // namespace tnn
