#pragma once

#include "../nn/sequential.hpp"
#include "network_serialization.hpp"
#include "pipeline_coordinator.hpp"
#include "tcp_communicator.hpp"
#define ASIO_STANDALONE
#include "../third_party/asio/asio/include/asio.hpp"
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
template <typename T = float>
class DistributedPipelineCoordinator : public PipelineCoordinator<T> {
public:
  struct RemoteEndpoint {
    std::string host;
    int port;
    std::string stage_id;

    RemoteEndpoint(std::string h, int p, std::string id)
        : host(std::move(h)), port(p), stage_id(std::move(id)) {}
  };

  DistributedPipelineCoordinator(
      tnn::Sequential<T> model, const std::vector<RemoteEndpoint> &endpoints,
      int num_microbatches = 4,
      const std::string &coordinator_host = "localhost",
      int coordinator_port = 8000)
      : PipelineCoordinator<T>(static_cast<int>(endpoints.size()),
                               num_microbatches),
        remote_endpoints_(endpoints), coordinator_port_(coordinator_port),
        io_context_(), work_guard_(asio::make_work_guard(io_context_)),
        is_deployed_(false) {

    if (model.get_layers().size() < endpoints.size()) {
      std::cout << "Error: Model has fewer layers ("
                << model.get_layers().size() << ") than remote endpoints ("
                << endpoints.size() << ")\n";
      throw std::invalid_argument(
          "Model must have at least as many layers as remote endpoints");
    }

    // Split the model into stages
    auto splitted_models = model.split(static_cast<int>(endpoints.size()));

    // Create stage configurations
    for (size_t i = 0; i < endpoints.size(); ++i) {
      StageConfig config;
      config.stage_id = endpoints[i].stage_id;
      config.stage_index = static_cast<int>(i);
      config.model_config = splitted_models[i].get_config();
      config.model_config["name"] = endpoints[i].stage_id; // Set stage name
      config.coordinator_endpoint =
          coordinator_host + ":" + std::to_string(coordinator_port_);

      // Set up stage interconnections
      if (i > 0) {
        config.prev_stage_endpoint =
            endpoints[i - 1].host + ":" + std::to_string(endpoints[i - 1].port);
      } else {
        // first stage connnects to coordinator
        config.prev_stage_endpoint =
            coordinator_host + ":" + std::to_string(coordinator_port_);
      }
      if (i < endpoints.size() - 1) {
        config.next_stage_endpoint =
            endpoints[i + 1].host + ":" + std::to_string(endpoints[i + 1].port);
      } else {
        config.next_stage_endpoint =
            coordinator_host + ":" + std::to_string(coordinator_port_);
      }

      stage_configs_.push_back(config);
      this->stage_names_.push_back(config.stage_id);
    }

    // Create coordinator communicator
    this->coordinator_comm_ = std::make_unique<TcpPipelineCommunicator<T>>(
        io_context_, coordinator_host, coordinator_port_);

    // Set up message notification callback for event-driven processing
    this->coordinator_comm_->set_message_notification_callback([this]() {
      std::lock_guard<std::mutex> lock(message_notification_mutex);
      message_notification_cv.notify_all();
    });

    // Start IO context in background thread
    io_thread_ = std::thread([this]() { io_context_.run(); });

    std::cout << "Distributed coordinator initialized with " << endpoints.size()
              << " remote endpoints\n";
  }

  ~DistributedPipelineCoordinator() {
    // First stop the coordinator and clean up connections
    stop();

    // Reset the coordinator communicator before destroying io_context
    this->coordinator_comm_.reset();

    // Then stop the io_context and cleanup
    work_guard_.reset();
    io_context_.stop();
    if (io_thread_.joinable()) {
      io_thread_.join();
    }
  }

  // Deploy stages to remote machines
  bool deploy_stages() {
    if (is_deployed_) {
      std::cout << "Stages already deployed\n";
      return true;
    }

    std::cout << "Starting deployment of distributed pipeline stages...\n";

    // Connect to all remote endpoints
    std::vector<std::future<bool>> connection_futures;

    for (const auto &endpoint : remote_endpoints_) {
      auto future = std::async(std::launch::async, [this, &endpoint]() {
        return connect_to_endpoint(endpoint);
      });
      connection_futures.push_back(std::move(future));
    }

    // Wait for all connections
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

    std::cout << "Connected to all endpoints, sending stage configurations...\n"
              << std::endl;

    // Send stage configurations
    std::vector<std::future<bool>> deployment_futures;

    for (size_t i = 0; i < remote_endpoints_.size(); ++i) {
      auto future = std::async(std::launch::async, [this, i]() {
        return deploy_stage_config(remote_endpoints_[i], stage_configs_[i]);
      });
      deployment_futures.push_back(std::move(future));
    }

    // Wait for all deployments
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

    // Wait for readiness confirmations
    if (!wait_for_stage_readiness()) {
      std::cout << "Not all stages reported ready\n";
      return false;
    }

    is_deployed_ = true;
    std::cout << "All stages deployed and ready!\n";
    return true;
  }

  void start() override {
    if (!is_deployed_) {
      throw std::runtime_error("Must deploy stages before starting");
    }

    // Send start command to all stages
    for (const auto &stage_name : this->stage_names_) {
      auto start_msg = Message<T>::create_control_message(
          CommandType::START_TRAINING, "coordinator", stage_name);
      this->send_message_to_stage(stage_name, start_msg);
    }

    // this->coordinator_comm_->flush_output_messages();
    std::cout << "Started all " << this->num_stages_
              << " distributed pipeline stages\n";
  }

  void stop() override {
    // Send stop command to all stages
    for (const auto &stage_name : this->stage_names_) {
      auto stop_msg = Message<T>::create_control_message(
          CommandType::STOP_TRAINING, "coordinator", stage_name);
      this->send_message_to_stage(stage_name, stop_msg);
    }

    std::cout << "Stopped all distributed pipeline stages\n";
  }

  void forward(const Tensor<T> &input, size_t microbatch_id) override {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &first_stage = this->stage_names_[0];

    // Create task for the first stage
    Task<T> task(TaskType::FORWARD, input, microbatch_id);
    auto forward_msg =
        Message<T>::forward_task(task, "coordinator", first_stage);
    forward_msg.sequence_number = microbatch_id;

    this->send_message_to_stage(first_stage, forward_msg);
  }

  void backward(const Tensor<T> &gradient, size_t microbatch_id) override {
    if (this->stage_names_.empty()) {
      throw std::runtime_error("No stages available for processing");
    }

    const std::string &last_stage = this->stage_names_.back();

    Task<T> task(TaskType::BACKWARD, gradient, microbatch_id);
    auto backward_msg =
        Message<T>::backward_task(task, "coordinator", last_stage);
    backward_msg.sequence_number = microbatch_id;

    this->send_message_to_stage(last_stage, backward_msg);

    // this->coordinator_comm_->flush_output_messages();
  }

  void join(bool direction) override {
    // Set expected task count based on direction
    expected_task_count_ = this->num_microbatches_;

    std::unique_lock<std::mutex> lock(message_notification_mutex);

    // Wait with timeout for the expected number of task messages to arrive
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);

    bool success =
        message_notification_cv.wait_until(lock, timeout, [this, direction]() {
          if (direction) {
            return this->coordinator_comm_->forward_message_count() >=
                   expected_task_count_.load();
          } else {
            return this->coordinator_comm_->backward_message_count() >=
                   expected_task_count_.load();
          }
        });

    if (!success) {
      std::cout << "Warning: join() timed out waiting for task messages. "
                << "Expected: " << expected_task_count_.load() << ", Got: "
                << (direction
                        ? this->coordinator_comm_->forward_message_count()
                        : this->coordinator_comm_->backward_message_count())
                << '\n';
    }
    return;
  }

  void update_parameters() override {
    for (const auto &stage_name : this->stage_names_) {
      auto update_msg = Message<T>(CommandType::UPDATE_PARAMETERS, true,
                                   "coordinator", stage_name);
      this->send_message_to_stage(stage_name, update_msg);
    }

    // Wait for confirmations
    wait_for_parameter_updates();
  }

  void async_process_batch(std::vector<Tensor<T>> &microbatch_inputs,
                           std::vector<Tensor<T>> &microbatch_labels) {
    if (microbatch_inputs.size() != this->num_microbatches_ ||
        microbatch_labels.size() != this->num_microbatches_) {
      throw std::invalid_argument(
          "Microbatch size mismatch with coordinator configuration");
    }

    if (loss_function_ == nullptr) {
      throw std::runtime_error(
          "Loss function not set for distributed coordinator");
    }

    for (size_t i = 0; i < this->num_microbatches_; ++i) {

      forward(microbatch_inputs[i], i);
    }

    // Backward on completion of any microbatch
    size_t processed_microbatches_ = 0;
    while (processed_microbatches_ < this->num_microbatches_) {
      std::unique_lock<std::mutex> lock(message_notification_mutex);
      message_notification_cv.wait(lock, [this]() {
        return this->coordinator_comm_->forward_message_count() > 0;
      });

      auto forward_msg = this->coordinator_comm_->dequeue_message_by_type(
          CommandType::FORWARD_TASK);

      if (forward_msg.has_task()) {
        ++processed_microbatches_;
        // Process the forward task
        const auto &task = forward_msg.task.value();

        // Compute loss and prepare backward task
        Tensor<T> predictions = task.data; // Assuming data contains predictions
        Tensor<T> targets = microbatch_labels[task.micro_batch_id];
        Tensor<T> gradient =
            loss_function_->compute_gradient(predictions, targets);

        // Send backward task
        backward(gradient, task.micro_batch_id);
      } else {
        throw std::runtime_error("Received forward message without task data");
      }
    }

    std::unique_lock<std::mutex> lock(message_notification_mutex);

    // Wait for all backward tasks to complete
    message_notification_cv.wait(lock, [this]() {
      return this->coordinator_comm_->backward_message_count() >=
             this->num_microbatches_;
    });

    this->get_task_messages();
  }

  void print_profiling_on_all_stages() override {
    if (!is_deployed_) {
      throw std::runtime_error("Must deploy stages before printing profiling");
    }

    // Send profiling request to all stages
    for (const auto &stage_name : this->stage_names_) {
      auto profiling_msg = Message<T>::create_control_message(
          CommandType::PRINT_PROFILING, "coordinator", stage_name);
      this->send_message_to_stage(stage_name, profiling_msg);
    }
  }

  void clear_profiling_data() override {
    if (!is_deployed_) {
      throw std::runtime_error(
          "Must deploy stages before clearing profiling data");
    }

    // Send clear profiling request to all stages
    for (const auto &stage_name : this->stage_names_) {
      auto clear_msg = Message<T>::create_control_message(
          CommandType::CLEAR_PROFILING, "coordinator", stage_name);
      this->send_message_to_stage(stage_name, clear_msg);
    }
  }

  bool is_deployed() const { return is_deployed_; }

  void set_loss_function_function(std::unique_ptr<tnn::Loss<T>> loss) {
    loss_function_ = std::move(loss);
  }

  const std::unique_ptr<tnn::Loss<T>> &get_loss_function() const {
    return loss_function_;
  }

private:
  std::unique_ptr<tnn::Loss<T>> loss_function_;
  std::vector<RemoteEndpoint> remote_endpoints_;
  std::vector<StageConfig> stage_configs_;
  int coordinator_port_;

  asio::io_context io_context_;
  asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
  std::thread io_thread_;

  std::atomic<bool> is_deployed_;
  std::atomic<int> expected_task_count_{0};

  mutable std::mutex message_notification_mutex;
  mutable std::condition_variable message_notification_cv;

  bool connect_to_endpoint(const RemoteEndpoint &endpoint) {
    try {
      std::cout << "Connecting to stage " << endpoint.stage_id << " at "
                << endpoint.host << ":" << endpoint.port << std::endl;
      auto tcp_comm = static_cast<TcpPipelineCommunicator<T> *>(
          this->coordinator_comm_.get());
      bool connected = tcp_comm->connect_to_peer(endpoint.stage_id,
                                                 endpoint.host, endpoint.port);

      if (connected) {
        std::cout << "Connected to stage " << endpoint.stage_id << " at "
                  << endpoint.host << ":" << endpoint.port << std::endl;
      }

      return connected;

    } catch (const std::exception &e) {
      std::cout << "Failed to connect to " << endpoint.host << ":"
                << endpoint.port << " - " << e.what() << '\n';
      return false;
    }
  }

  bool deploy_stage_config(const RemoteEndpoint &endpoint,
                           const StageConfig &config) {
    try {
      // Send stage configuration as a text message
      std::string config_json = config.to_json().dump();
      auto config_msg = Message<T>::create_text_message(
          CommandType::CONFIG_RECEIVED, config_json, "coordinator",
          endpoint.stage_id);

      this->send_message_to_stage(endpoint.stage_id, config_msg);

      std::cout << "Sent configuration to stage " << endpoint.stage_id << '\n';
      return true;

    } catch (const std::exception &e) {
      std::cout << "Failed to deploy config to stage " << endpoint.stage_id
                << ": " << e.what() << '\n';
      return false;
    }
  }

  bool wait_for_stage_readiness() {
    std::unique_lock<std::mutex> lock(message_notification_mutex);

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(60);

    // Use condition variable instead of polling
    bool success = message_notification_cv.wait_until(lock, timeout, [this]() {
      return this->coordinator_comm_->message_count_by_type(
                 CommandType::READY_SIGNAL) >=
             static_cast<size_t>(this->num_stages_);
    });

    if (!success) {
      std::cout << "Timeout waiting for stage readiness\n";
      return false;
    }

    std::cout << "All stages reported ready!\n";

    return true;
  }

  void wait_for_parameter_updates() {
    int confirmations = 0;

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);

    std::unique_lock<std::mutex> lock(message_notification_mutex);

    bool success = message_notification_cv.wait_until(lock, timeout, [this]() {
      return this->coordinator_comm_->params_updated_count() >=
             static_cast<size_t>(this->num_stages_);
    });

    if (!success) {
      std::cout << "Warning: wait_for_parameter_updates() timed out. "
                << "Expected: " << this->num_stages_
                << ", Got: " << this->coordinator_comm_->params_updated_count()
                << '\n';
      return;
    }
    return;
  }
};

} // namespace tpipeline
