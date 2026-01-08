/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "roce_communicator.hpp"
#include "worker.hpp"
#include <iostream>
#include <memory>
#include <string>

namespace tnn {

/**
 * @brief RoCE-based pipeline stage worker
 *
 * Standalone worker process that listens for stage configurations
 * from a coordinator and processes distributed pipeline jobs using RDMA.
 */
template <typename T = float> class RoceWorker : public Worker {
public:
  /**
   * @brief Constructor for RoCE worker
   * @param host Hostname or IP to bind to (for TCP handshake)
   * @param port TCP port for initial connection setup
   * @param device_name IB device name (e.g., "mlx5_0")
   * @param gid_index GID index for RoCE
   * @param use_gpu Whether to use GPU for processing
   */
  explicit RoceWorker(Endpoint worker_endpoint, bool use_gpu)
      : Worker(use_gpu), worker_endpoint_(worker_endpoint) {

    auto communicator = std::make_unique<RoceCommunicator>(this->id_, worker_endpoint_);

    communicator->start_server();

    this->communicator_ = std::move(communicator);
  }

  ~RoceWorker() override { stop(); }

  void start() override {
    if (!this->should_stop_)
      return;

    Worker::start();
  }

  void stop() override {
    std::cout << "Stopping RoCE worker." << '\n';
    Worker::stop();
    std::cout << "RoCE worker stopped" << '\n';
  }

private:
  Endpoint worker_endpoint_;
};

} // namespace tnn
