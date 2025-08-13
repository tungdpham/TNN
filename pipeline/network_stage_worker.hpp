#pragma once

#include "pipeline_stage.hpp"
#include "tcp_communicator.hpp"
#include "network_serialization.hpp"
#include "../nn/sequential.hpp"
#define ASIO_STANDALONE
#include "../third_party/asio/asio/include/asio.hpp"
#include <thread>
#include <atomic>
#include <memory>

namespace tpipeline {

/**
 * @brief Network-based pipeline stage worker
 * 
 * Standalone worker process that listens for stage configurations
 * from a coordinator and processes distributed pipeline tasks.
 */
template <typename T = float>
class NetworkStageWorker {
public:
    explicit NetworkStageWorker(int listen_port)
        : listen_port_(listen_port),
          io_context_(),
          work_guard_(asio::make_work_guard(io_context_)),
          is_running_(false),
          is_configured_(false) {
        
        // Create TCP communicator
        communicator_ = std::make_unique<TcpPipelineCommunicator<T>>(
            io_context_, "localhost", listen_port_);
        
        // Set up message notification callback
        communicator_->set_message_notification_callback([this]() {
            printf("Tcp communicator: Message notification callback triggered\n");
            std::lock_guard<std::mutex> lock(message_mutex_);
            process_message(communicator_->dequeue_input_message());
        });
    }
    
    ~NetworkStageWorker() {
        stop();
    }
    
    void start() {
        if (is_running_) {
            printf("Worker already running\n");
            return;
        }
        
        is_running_ = true;
        
        // Start IO context in background thread
        io_thread_ = std::thread([this]() {
            io_context_.run();
        });
        
        printf("Network stage worker listening on port %d\n", listen_port_);
    }
    
    void stop() {
        is_running_ = false;
        
        message_cv_.notify_all();
        
        // if (message_thread_.joinable()) {
        //     message_thread_.join();
        // }
        
        work_guard_.reset();
        io_context_.stop();
        
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
        
        printf("Network stage worker stopped\n");
    }
    
    void wait_for_shutdown() {
        if (message_thread_.joinable()) {
            message_thread_.join();
        }
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }
    
    bool is_configured() const { return is_configured_; }
    std::string get_stage_id() const { return stage_id_; }

private:
    int listen_port_;
    asio::io_context io_context_;
    asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
    std::thread io_thread_;
    std::thread message_thread_;
    
    std::unique_ptr<TcpPipelineCommunicator<T>> communicator_;
    std::unique_ptr<PipelineStage<T>> stage_;
    
    std::atomic<bool> is_running_;
    std::atomic<bool> is_configured_;
    std::string stage_id_;
    
    std::mutex message_mutex_;
    std::condition_variable message_cv_;
    
    // void message_loop() {
    //     printf("Starting message processing loop\n");
        
    //     while (is_running_) {
    //         std::unique_lock<std::mutex> lock(message_mutex_);
    //         message_cv_.wait(lock, [this]() {
    //             printf("Stage waiting for messages...\n");
    //             return !is_running_ || communicator_->has_input_message();
    //         });
            
    //         if (!is_running_){
    //             printf("Exiting message loop due to shutdown\n");
    //             break; 
    //         }
            
    //         // Process all available messages
    //         while (communicator_->has_input_message()) {
    //             try {
    //                 printf("Worker processing input message\n");
    //                 auto message = communicator_->dequeue_input_message();
    //                 process_message(message);
    //             } catch (const std::exception& e) {
    //                 printf("Error processing message: %s\n", e.what());
    //             }
    //         }
    //     }
        
    //     printf("Message processing loop ended\n");
    // }
    
    void process_message(const Message<T>& message) {
        printf("Worker processing message type %d\n", static_cast<int>(message.command_type));
        
        switch (message.command_type) {
        case CommandType::CONFIG_RECEIVED:
            printf("Handling configuration message\n");
            handle_configuration(message);
            break;
            
        case CommandType::HANDSHAKE_REQUEST:
            printf("Handling handshake request\n");
            handle_handshake(message);
            break;
            
        default:
            // If we have a configured stage, delegate to it
            if (is_configured_ && stage_) {
                printf("Forwarding message type %d to configured stage\n", 
                       static_cast<int>(message.command_type));
            
                // Forward the message to the stage's communicator
                // stage_->get_communicator()->enqueue_input_message(message);
                stage_->process_message(message);
            } else {
                printf("Received message type %d but stage not configured\n", 
                       static_cast<int>(message.command_type));
            }
            break;
        }
    }
    
    void handle_configuration(const Message<T>& message) {
        if (!message.text_data.has_value()) {
            printf("Configuration message missing text data\n");
            return;
        }
        
        try {
            // Parse stage configuration
            nlohmann::json config_json = nlohmann::json::parse(message.text_data.value());
            //print the configuration JSON
            printf("Received configuration JSON: %s\n", config_json.dump(2).c_str());

            StageConfig config = StageConfig::from_json(config_json);
            
            stage_id_ = config.stage_id;
            
            printf("Received configuration for stage %s\n", stage_id_.c_str());
            
            // Create the model from configuration
            auto model = std::make_unique<tnn::Sequential<T>>(
                tnn::Sequential<T>::load_from_config(config.model_config));
            
            model->enable_profiling(true);
            
            printf("Created model with %zu layers\n", model->size());
            
            // Connect to other stages and coordinator
            setup_stage_connections(config);
            
            // Create the pipeline stage with a custom communicator wrapper
            auto stage_comm_wrapper = create_stage_communicator_wrapper();
            
            stage_ = std::make_unique<PipelineStage<T>>(
                std::move(model), std::move(stage_comm_wrapper), stage_id_);
            
            is_configured_ = true;
            
            // Send ready signal to coordinator
            auto ready_msg = Message<T>::create_signal_message(
                CommandType::READY_SIGNAL, true, stage_id_, "coordinator");
            communicator_->enqueue_output_message(ready_msg);
            communicator_->flush_output_messages();
            
            printf("Stage %s configured and ready\n", stage_id_.c_str());
            
        } catch (const std::exception& e) {
            printf("Failed to configure stage: %s\n", e.what());
            
            // Send error message back to coordinator
            auto error_msg = Message<T>::error_message(
                std::string("Configuration failed: ") + e.what(), 
                stage_id_.empty() ? "unknown" : stage_id_, "coordinator");
            communicator_->enqueue_output_message(error_msg);
            communicator_->flush_output_messages();
        }
    }
    
    void handle_handshake(const Message<T>& message) {
        // Respond to handshake
        auto response = Message<T>::create_control_message(
            CommandType::HANDSHAKE_RESPONSE, 
            stage_id_.empty() ? "worker" : stage_id_, 
            message.sender_id);
        
        communicator_->enqueue_output_message(response);
        communicator_->flush_output_messages();
        
        printf("Responded to handshake from %s\n", message.sender_id.c_str());
    }
    
    void setup_stage_connections(const StageConfig& config) {
        // Connect to coordinator
        if (!config.coordinator_endpoint.empty()) {
            auto [host, port] = parse_endpoint(config.coordinator_endpoint);
            if (!communicator_->connect_to_peer("coordinator", host, port)) {
                throw std::runtime_error("Failed to connect to coordinator");
            }
            printf("Connected to coordinator at %s\n", config.coordinator_endpoint.c_str());
        }
        
        // Connect to next stage if specified
        if (!config.next_stage_endpoint.empty()) {
            auto [host, port] = parse_endpoint(config.next_stage_endpoint);
            if (!communicator_->connect_to_peer("next_stage", host, port)) {
                printf("Warning: Failed to connect to next stage at %s\n", 
                       config.next_stage_endpoint.c_str());
            } else {
                printf("Connected to next stage at %s\n", config.next_stage_endpoint.c_str());
            }
        }
        
        // Connect to previous stage if specified
        if (!config.prev_stage_endpoint.empty()) {
            auto [host, port] = parse_endpoint(config.prev_stage_endpoint);
            if (!communicator_->connect_to_peer("prev_stage", host, port)) {
                printf("Warning: Failed to connect to previous stage at %s\n", 
                       config.prev_stage_endpoint.c_str());
            } else {
                printf("Connected to previous stage at %s\n", config.prev_stage_endpoint.c_str());
            }
        }
    }
    
    std::unique_ptr<PipelineCommunicator<T>, std::function<void(PipelineCommunicator<T>*)>>
    create_stage_communicator_wrapper() {
        // Create a wrapper that uses the same underlying TCP communicator
        // but provides the interface expected by PipelineStage
        return std::unique_ptr<PipelineCommunicator<T>, std::function<void(PipelineCommunicator<T>*)>>(
            communicator_.get(),
            [](PipelineCommunicator<T>*) {
                // Don't delete - we manage the lifetime
            });
    }
    
    std::pair<std::string, int> parse_endpoint(const std::string& endpoint) {
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
template <typename T = float>
class StandaloneNetworkWorker {
public:
    static int run_worker(int listen_port) {
        try {
            NetworkStageWorker<T> worker(listen_port);
            
            // Set up signal handling for graceful shutdown
            std::signal(SIGINT, [](int) {
                printf("\nReceived interrupt signal, shutting down...\n");
                std::exit(0);
            });
            
            worker.start();
            
            printf("Worker started on port %d. Press Ctrl+C to stop.\n", listen_port);
            
            // Wait indefinitely
            worker.wait_for_shutdown();
            
            return 0;
            
        } catch (const std::exception& e) {
            printf("Worker failed: %s\n", e.what());
            return 1;
        }
    }
};

} // namespace tpipeline
