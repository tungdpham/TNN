#pragma once

#include "pipeline_coordinator.hpp"
#include "tcp_communicator.hpp"
#include "network_serialization.hpp"
#include "../nn/sequential.hpp"
#define ASIO_STANDALONE
#include "../third_party/asio/asio/include/asio.hpp"
#include <thread>
#include <future>
#include <vector>
#include <string>

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
    
    DistributedPipelineCoordinator(tnn::Sequential<T> model, 
                                 const std::vector<RemoteEndpoint>& endpoints,
                                 int num_microbatches = 4,
                                const std::string& coordinator_host = "localhost",
                                 int coordinator_port = 8000)
        : PipelineCoordinator<T>(static_cast<int>(endpoints.size()), num_microbatches),
          remote_endpoints_(endpoints),
          coordinator_port_(coordinator_port),
          io_context_(),
          work_guard_(asio::make_work_guard(io_context_)),
          is_deployed_(false) {
        
        if (model.get_layers().size() < endpoints.size()) {
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
            config.coordinator_endpoint = "localhost:" + std::to_string(coordinator_port_);
            
            // Set up stage interconnections
            if (i > 0) {
                config.prev_stage_endpoint = endpoints[i-1].host + ":" + std::to_string(endpoints[i-1].port);
            }else {
              // first stage connnects to coordinator
                config.prev_stage_endpoint = coordinator_host + ":" + std::to_string(coordinator_port_);
            }
            if (i < endpoints.size() - 1) {
                config.next_stage_endpoint = endpoints[i+1].host + ":" + std::to_string(endpoints[i+1].port);
            }else {
                config.next_stage_endpoint = coordinator_host + ":" + std::to_string(coordinator_port_);
            }
            
            stage_configs_.push_back(config);
            this->stage_names_.push_back(config.stage_id);
        }
        
        // Create coordinator communicator
        this->coordinator_comm_ = std::make_unique<TcpPipelineCommunicator<T>>(
            io_context_, "localhost", coordinator_port_);
        
        // Set up message notification callback for event-driven processing
        this->coordinator_comm_->set_message_notification_callback([this]() {
            std::lock_guard<std::mutex> lock(task_notification_mutex_);
            task_notification_cv_.notify_all();
        });
        
        // Start IO context in background thread
        io_thread_ = std::thread([this]() {
            io_context_.run();
        });
        
        printf("Distributed coordinator initialized with %zu remote endpoints\n", endpoints.size());
    }
    
    ~DistributedPipelineCoordinator() {
        stop();
        work_guard_.reset();
        io_context_.stop();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }
    
    // Deploy stages to remote machines
    bool deploy_stages() {
        if (is_deployed_) {
            printf("Stages already deployed\n");
            return true;
        }
        
        printf("Starting stage deployment...\n");
        
        // Connect to all remote endpoints
        std::vector<std::future<bool>> connection_futures;
        
        for (const auto& endpoint : remote_endpoints_) {
            auto future = std::async(std::launch::async, [this, &endpoint]() {
                return connect_to_endpoint(endpoint);
            });
            connection_futures.push_back(std::move(future));
        }
        
        // Wait for all connections
        bool all_connected = true;
        for (auto& future : connection_futures) {
            if (!future.get()) {
                all_connected = false;
            }
        }
        
        if (!all_connected) {
            printf("Failed to connect to all endpoints\n");
            return false;
        }
        
        printf("Connected to all endpoints, sending stage configurations...\n");
        
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
        for (auto& future : deployment_futures) {
            if (!future.get()) {
                all_deployed = false;
            }
        }
        
        if (!all_deployed) {
            printf("Failed to deploy all stages\n");
            return false;
        }
        
        // Wait for readiness confirmations
        if (!wait_for_stage_readiness()) {
            printf("Not all stages reported ready\n");
            return false;
        }
        
        is_deployed_ = true;
        printf("All stages deployed and ready!\n");
        return true;
    }
    
    void start() override {
        if (!is_deployed_) {
            throw std::runtime_error("Must deploy stages before starting");
        }
        
        // Send start command to all stages
        for (const auto& stage_name : this->stage_names_) {
            auto start_msg = Message<T>::create_control_message(
                CommandType::START_TRAINING, "coordinator", stage_name);
            this->send_message_to_stage(stage_name, start_msg);
        }
        
        this->coordinator_comm_->flush_output_messages();
        printf("Started all %d distributed pipeline stages\n", this->num_stages_);
    }
    
    void stop() override {
        // Send stop command to all stages
        for (const auto& stage_name : this->stage_names_) {
            auto stop_msg = Message<T>::create_control_message(
                CommandType::STOP_TRAINING, "coordinator", stage_name);
            this->send_message_to_stage(stage_name, stop_msg);
        }
        
        this->coordinator_comm_->flush_output_messages();
        printf("Stopped all distributed pipeline stages\n");
    }
    
    void forward(const Tensor<T>& batch) override {
        if (this->stage_names_.empty()) {
            throw std::runtime_error("No stages available for processing");
        }
        
        // Split batch into microbatches
        auto microbatches = batch.split(this->num_microbatches_);
        
        // Send each microbatch to the first stage
        for (int i = 0; i < this->num_microbatches_; ++i) {
            Task<T> task{TaskType::FORWARD, microbatches[i], i};
            auto forward_msg = Message<T>::forward_task(task, "coordinator", this->stage_names_[0]);
            forward_msg.sequence_number = i;
            this->send_message_to_stage(this->stage_names_[0], forward_msg);
        }
        
        this->coordinator_comm_->flush_output_messages();
    }
    
    void backward(const std::vector<Tensor<T>>& gradients) override {
        if (this->stage_names_.empty()) {
            throw std::runtime_error("No stages available for processing");
        }
        
        const std::string& last_stage = this->stage_names_[this->num_stages_ - 1];
        
        // Send each gradient to the last stage
        for (size_t i = 0; i < gradients.size(); ++i) {
            Task<T> task{TaskType::BACKWARD, gradients[i], static_cast<int>(i)};
            auto backward_msg = Message<T>::backward_task(task, "coordinator", last_stage);
            backward_msg.sequence_number = static_cast<int>(i);
            this->send_message_to_stage(last_stage, backward_msg);
        }
        
        this->coordinator_comm_->flush_output_messages();
    }
    
    void join(bool direction) override {
        // Set expected task count based on direction
        expected_task_count_ = this->num_microbatches_;
        
        std::unique_lock<std::mutex> lock(task_notification_mutex_);
        
        // Wait with timeout for the expected number of task messages to arrive
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(30); // Longer timeout for network
        
        bool success = task_notification_cv_.wait_until(lock, timeout, [this]() {
            return this->coordinator_comm_->actual_task_message_count() >= expected_task_count_.load();
        });
        
        if (!success) {
            printf("Warning: Timeout waiting for %d task messages (got %zu)\n", 
                   expected_task_count_.load(), this->coordinator_comm_->actual_task_message_count());
        } else {
            printf("Received all %d expected task messages\n", expected_task_count_.load());
        }
    }
    
    void update_parameters() {
        for (const auto& stage_name : this->stage_names_) {
            auto update_msg = Message<T>::create_control_message(
                CommandType::UPDATE_PARAMETERS, "coordinator", stage_name);
            this->send_message_to_stage(stage_name, update_msg);
        }
        this->coordinator_comm_->flush_output_messages();
        
        // Wait for confirmations
        wait_for_parameter_updates();
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
    std::atomic<int> expected_task_count_{0};
    
    mutable std::mutex task_notification_mutex_;
    mutable std::condition_variable task_notification_cv_;
    
    bool connect_to_endpoint(const RemoteEndpoint& endpoint) {
        try {
            auto tcp_comm = static_cast<TcpPipelineCommunicator<T>*>(this->coordinator_comm_.get());
            bool connected = tcp_comm->connect_to_peer(endpoint.stage_id, endpoint.host, endpoint.port);
            
            if (connected) {
                printf("Connected to stage %s at %s:%d\n", 
                       endpoint.stage_id.c_str(), endpoint.host.c_str(), endpoint.port);
            }
            
            return connected;
            
        } catch (const std::exception& e) {
            printf("Failed to connect to %s:%d - %s\n", 
                   endpoint.host.c_str(), endpoint.port, e.what());
            return false;
        }
    }
    
    bool deploy_stage_config(const RemoteEndpoint& endpoint, const StageConfig& config) {
        try {
            // Send stage configuration as a text message
            std::string config_json = config.to_json().dump();
            auto config_msg = Message<T>::create_text_message(
                CommandType::CONFIG_RECEIVED, config_json, "coordinator", endpoint.stage_id);
            
            printf("Sending CONFIG_RECEIVED (type %d) to %s\n", 
                   static_cast<int>(CommandType::CONFIG_RECEIVED), endpoint.stage_id.c_str());
            
            this->send_message_to_stage(endpoint.stage_id, config_msg);
            this->coordinator_comm_->flush_output_messages();
            
            printf("Sent configuration to stage %s\n", endpoint.stage_id.c_str());
            return true;
            
        } catch (const std::exception& e) {
            printf("Failed to deploy config to stage %s: %s\n", 
                   endpoint.stage_id.c_str(), e.what());
            return false;
        }
    }
    
    bool wait_for_stage_readiness() {
        int ready_count = 0;
        auto start_time = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::seconds(60); // Longer timeout for initial setup
        
        printf("Waiting for %d stages to report ready...\n", this->num_stages_);
        
        while (ready_count < this->num_stages_) {
            if (std::chrono::steady_clock::now() - start_time > timeout) {
                printf("Timeout waiting for stage readiness (%d/%d ready)\n", 
                       ready_count, this->num_stages_);
                return false;
            }
            
            // Check for any incoming messages (using simple polling)
            if (this->coordinator_comm_->has_input_message()) {
                printf("Coordinator has input messages available\n");
                
                // Process all available messages
                while (this->coordinator_comm_->has_input_message()) {
                    try {
                        auto message = this->coordinator_comm_->dequeue_input_message();
                        printf("Received message type %d from %s\n", 
                               static_cast<int>(message.command_type), message.sender_id.c_str());
                        
                        if (message.command_type == CommandType::READY_SIGNAL) {
                            ready_count++;
                            printf("Stage %s reported ready (%d/%d)\n", 
                                   message.sender_id.c_str(), ready_count, this->num_stages_);
                        } else {
                            printf("Received non-ready message type %d from %s during readiness wait\n",
                                   static_cast<int>(message.command_type), message.sender_id.c_str());
                        }
                    } catch (const std::runtime_error& e) {
                        printf("Error dequeuing message: %s\n", e.what());
                        break; // No more messages
                    }
                }
            }
            
            // Small delay to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        return true;
    }
    
    void wait_for_parameter_updates() {
        int confirmations = 0;
        auto start_time = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::seconds(30);
        
        while (confirmations < this->num_stages_) {
            if (std::chrono::steady_clock::now() - start_time > timeout) {
                printf("Timeout waiting for parameter update confirmations (%d/%d received)\n",
                       confirmations, this->num_stages_);
                break;
            }
            
            this->coordinator_comm_->receive_messages();
            
            while (this->coordinator_comm_->has_parameter_update_message()) {
                try {
                    auto message = this->coordinator_comm_->dequeue_parameter_update_message();
                    if (message.command_type == CommandType::PARAMETERS_UPDATED) {
                        confirmations++;
                        printf("Parameter update confirmed from stage %s (%d/%d)\n",
                               message.sender_id.c_str(), confirmations, this->num_stages_);
                    }
                } catch (const std::runtime_error& e) {
                    break;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

} // namespace tpipeline
