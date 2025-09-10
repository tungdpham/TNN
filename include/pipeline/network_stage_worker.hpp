/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "network_serialization.hpp"
#include "nn/sequential.hpp"
#include "pipeline_stage.hpp"
#include "tcp_communicator.hpp"
#define ASIO_STANDALONE
#include "../third_party/asio/asio/include/asio.hpp"
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
template <typename T = float>
class NetworkStageWorker : public PipelineStage<T> {
public:
  explicit NetworkStageWorker(int listen_port)
      : PipelineStage<T>(), listen_port_(listen_port), io_context_(),
        work_guard_(asio::make_work_guard(io_context_)), is_running_(false),
        is_configured_(false) {

    tcp_communicator_ = std::make_unique<TcpPipelineCommunicator<T>>(
        io_context_, "localhost", listen_port_);

    this->communicator_ =
        std::unique_ptr<PipelineCommunicator<T>,
                        std::function<void(PipelineCommunicator<T> *)>>(
            tcp_communicator_.get(), [](PipelineCommunicator<T> *) {});
  }

  ~NetworkStageWorker() { stop(); }

  void start() override {
    if (!this->should_stop_)
      return;

    PipelineStage<T>::start();

    std::cout << "Starting network stage worker on port " << listen_port_
              << '\n';

    io_thread_ = std::thread([this]() { io_context_.run(); });

    std::cout << "Network stage worker listening on port " << listen_port_
              << '\n';
  }

  void stop() override {
    PipelineStage<T>::stop();

    tcp_communicator_->stop();

    work_guard_.reset();
    io_context_.stop();

    if (io_thread_.joinable()) {
      io_thread_.join();
    }

    std::cout << "Network stage worker stopped" << '\n';
  }

  bool is_configured() const { return is_configured_; }
  std::string get_stage_id() const { return stage_id_; }

private:
  int listen_port_;
  asio::io_context io_context_;
  asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
  std::thread io_thread_;

  std::unique_ptr<TcpPipelineCommunicator<T>> tcp_communicator_;

  std::atomic<bool> is_running_;
  std::atomic<bool> is_configured_;
  std::string stage_id_;

  void process_message(const Message<T> &message) override {
    switch (message.command_type) {
    case CommandType::CONFIG_TRANSFER:
      std::cout << "Handling configuration message" << '\n';
      handle_configuration(message);
      break;

    case CommandType::HANDSHAKE_REQUEST:
      std::cout << "Handling handshake request" << '\n';
      handle_handshake(message);
      break;
    default:
      if (is_configured_ && this->get_model()) {
        PipelineStage<T>::process_message(message);
      } else {
        std::cout << "Received message type "
                  << static_cast<int>(message.command_type)
                  << " but stage not configured" << '\n';
      }
      break;
    }
  }

  void handle_configuration(const Message<T> &message) {
    if (!message.has_text()) {
      std::cout << "Configuration message missing text data" << '\n';
      return;
    }

    try {
      nlohmann::json config_json = nlohmann::json::parse(message.get_text());

      std::cout << "Received configuration JSON: " << config_json.dump(2)
                << '\n';

      StageConfig config = StageConfig::from_json(config_json);

      stage_id_ = config.stage_id;

      std::cout << "Received configuration for stage " << stage_id_ << '\n';

      this->model_ = std::make_unique<tnn::Sequential<T>>(
          tnn::Sequential<T>::load_from_config(config.model_config));

      this->model_->enable_profiling(true);

      std::cout << "Created model with " << this->model_->size() << " layers"
                << '\n';

      setup_stage_connections(config);

      this->name_ = stage_id_;

      is_configured_ = true;

      auto ready_msg = Message<T>::create_signal_message(
          CommandType::CONFIG_RECEIVED, true, stage_id_, "coordinator");
      tcp_communicator_->enqueue_output_message(ready_msg);
      tcp_communicator_->flush_output_messages();

      std::cout << "Stage " << stage_id_ << " configured and ready" << '\n';

    } catch (const std::exception &e) {
      std::cout << "Failed to configure stage: " << e.what() << '\n';

      auto error_msg = Message<T>::error_message(
          std::string("Configuration failed: ") + e.what(),
          stage_id_.empty() ? "unknown" : stage_id_, "coordinator");
      tcp_communicator_->enqueue_output_message(error_msg);
      tcp_communicator_->flush_output_messages();
    }
  }

  void handle_handshake(const Message<T> &message) {

    auto response = Message<T>::create_control_message(
        CommandType::HANDSHAKE_RESPONSE,
        stage_id_.empty() ? "worker" : stage_id_, message.sender_id);

    tcp_communicator_->enqueue_output_message(response);
    tcp_communicator_->flush_output_messages();

    std::cout << "Responded to handshake from " << message.sender_id << '\n';
  }

  void setup_stage_connections(const StageConfig &config) {

    if (!config.coordinator_endpoint.empty()) {
      auto [host, port] = parse_endpoint(config.coordinator_endpoint);
      if (!tcp_communicator_->connect_to_peer("coordinator", host, port)) {
        throw std::runtime_error("Failed to connect to coordinator");
      }
      std::cout << "Connected to coordinator at " << config.coordinator_endpoint
                << '\n';
    }

    if (!config.next_stage_endpoint.empty()) {
      auto [host, port] = parse_endpoint(config.next_stage_endpoint);
      if (!tcp_communicator_->connect_to_peer("next_stage", host, port)) {
        std::cout << "Warning: Failed to connect to next stage at "
                  << config.next_stage_endpoint << '\n';
      } else {
        std::cout << "Connected to next stage at " << config.next_stage_endpoint
                  << '\n';
      }
    }

    if (!config.prev_stage_endpoint.empty()) {
      auto [host, port] = parse_endpoint(config.prev_stage_endpoint);
      if (!tcp_communicator_->connect_to_peer("prev_stage", host, port)) {
        std::cout << "Warning: Failed to connect to previous stage at "
                  << config.prev_stage_endpoint << '\n';
      } else {
        std::cout << "Connected to previous stage at "
                  << config.prev_stage_endpoint << '\n';
      }
    }
  }

  std::pair<std::string, int> parse_endpoint(const std::string &endpoint) {
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
        std::cout << '\n'
                  << "Received interrupt signal, shutting down..." << '\n';
        std::exit(0);
      });

      worker.start();

      std::cout << "Network stage worker started on port " << listen_port
                << std::endl;

      worker.message_loop();

      return 0;
    } catch (const std::exception &e) {
      std::cout << "Worker failed: " << e.what() << '\n';
      return -1;
    }
  }
};

} // namespace tpipeline
