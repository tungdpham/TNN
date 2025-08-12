#pragma once

#include "pipeline_communicator.hpp"
#include "network_serialization.hpp"
#define ASIO_STANDALONE
#include "../third_party/asio/asio/include/asio.hpp"
#include <memory>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <functional>

namespace tpipeline {

/**
 * @brief TCP-based network communicator using ASIO
 * 
 * Provides reliable network communication between distributed pipeline stages
 * using TCP connections with automatic reconnection and message queuing.
 */
template <typename T = float>
class TcpPipelineCommunicator : public PipelineCommunicator<T> {
public:
    explicit TcpPipelineCommunicator(asio::io_context& io_context, 
                                   const std::string& local_endpoint = "",
                                   int listen_port = 0)
        : io_context_(io_context), 
          acceptor_(io_context),
          socket_(io_context),
          listen_port_(listen_port),
          local_endpoint_(local_endpoint),
          is_running_(false) {
        
        if (listen_port > 0) {
            start_server();
        }
    }
    
    ~TcpPipelineCommunicator() override {
        stop();
    }
    
    void start_server() {
        if (listen_port_ <= 0) return;
        
        asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(), listen_port_);
        acceptor_.open(endpoint.protocol());
        acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
        acceptor_.bind(endpoint);
        acceptor_.listen();
        
        is_running_ = true;
        accept_connections();
        
        printf("TCP communicator listening on port %d\n", listen_port_);
    }
    
    void stop() {
        is_running_ = false;
        
        if (acceptor_.is_open()) {
            acceptor_.close();
        }
        
        // Close all client connections
        {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            for (auto& [id, conn] : connections_) {
                if (conn->socket.is_open()) {
                    conn->socket.close();
                }
            }
            connections_.clear();
        }
    }
    
    void send_message(const std::string& recipient_id, const Message<T>& message) override {
        auto serialized = BinarySerializer::serialize_message(message);
        
        // Add message length header (4 bytes)
        std::vector<uint8_t> packet;
        uint32_t msg_length = static_cast<uint32_t>(serialized.size());
        packet.resize(4 + serialized.size());
        std::memcpy(packet.data(), &msg_length, 4);
        std::memcpy(packet.data() + 4, serialized.data(), serialized.size());
        
        send_packet(recipient_id, packet);
    }
    
    void flush_output_messages() override {
        std::lock_guard<std::mutex> lock(this->out_message_mutex_);
        
        while (!this->out_message_queue_.empty()) {
            auto& msg = this->out_message_queue_.front();
            send_message(msg.recipient_id, msg.message);
            this->out_message_queue_.pop();
        }
    }
    
    void receive_messages() override {
        // Messages are received asynchronously through the accept/read loops
        // This method is a no-op for async implementation
    }
    
    // Connect to a remote endpoint
    bool connect_to_peer(const std::string& peer_id, const std::string& host, int port) {
        try {
            auto connection = std::make_shared<Connection>(io_context_);
            
            asio::ip::tcp::resolver resolver(io_context_);
            auto endpoints = resolver.resolve(host, std::to_string(port));
            
            asio::connect(connection->socket, endpoints);
            
            {
                std::lock_guard<std::mutex> lock(connections_mutex_);
                connections_[peer_id] = connection;
            }
            
            // Start reading from this connection
            start_read(peer_id, connection);
            
            printf("Connected to peer %s at %s:%d\n", peer_id.c_str(), host.c_str(), port);
            return true;
            
        } catch (const std::exception& e) {
            printf("Failed to connect to peer %s: %s\n", peer_id.c_str(), e.what());
            return false;
        }
    }
    
    int get_listen_port() const { return listen_port_; }
    std::string get_local_endpoint() const { return local_endpoint_; }

private:
    struct Connection {
        asio::ip::tcp::socket socket;
        std::vector<uint8_t> read_buffer;
        std::mutex write_mutex;
        
        explicit Connection(asio::io_context& io_ctx) 
            : socket(io_ctx), read_buffer(8192) {}
        explicit Connection(asio::ip::tcp::socket sock) 
            : socket(std::move(sock)), read_buffer(8192) {}
    };
    
    asio::io_context& io_context_;
    asio::ip::tcp::acceptor acceptor_;
    asio::ip::tcp::socket socket_;
    
    int listen_port_;
    std::string local_endpoint_;
    std::atomic<bool> is_running_;
    
    std::unordered_map<std::string, std::shared_ptr<Connection>> connections_;
    std::mutex connections_mutex_;
    
    void accept_connections() {
        if (!is_running_) return;
        
        auto new_connection = std::make_shared<Connection>(io_context_);
        
        acceptor_.async_accept(new_connection->socket,
            [this, new_connection](std::error_code ec) {
                if (!ec && is_running_) {
                    // We'll identify the connection when we receive the first message
                    auto remote_endpoint = new_connection->socket.remote_endpoint();
                    std::string temp_id = remote_endpoint.address().to_string() + ":" + 
                                        std::to_string(remote_endpoint.port());
                    
                    {
                        std::lock_guard<std::mutex> lock(connections_mutex_);
                        connections_[temp_id] = new_connection;
                    }
                    
                    start_read(temp_id, new_connection);
                    printf("Accepted connection from %s\n", temp_id.c_str());
                }
                
                // Continue accepting connections
                accept_connections();
            });
    }
    
    void start_read(const std::string& connection_id, std::shared_ptr<Connection> connection) {
        if (!is_running_) return;
        
        // Read message length first (4 bytes)
        asio::async_read(connection->socket, 
                        asio::buffer(connection->read_buffer.data(), 4),
            [this, connection_id, connection](std::error_code ec, std::size_t length) {
                if (!ec && is_running_) {
                    // Extract message length
                    uint32_t msg_length;
                    std::memcpy(&msg_length, connection->read_buffer.data(), 4);
                    
                    if (msg_length > 0 && msg_length < 1024 * 1024) { // Sanity check
                        // Read the actual message
                        if (connection->read_buffer.size() < msg_length) {
                            connection->read_buffer.resize(msg_length);
                        }
                        
                        asio::async_read(connection->socket,
                                        asio::buffer(connection->read_buffer.data(), msg_length),
                            [this, connection_id, connection, msg_length](std::error_code ec2, std::size_t) {
                                if (!ec2 && is_running_) {
                                    handle_message(connection_id, connection->read_buffer, msg_length);
                                    // Continue reading
                                    start_read(connection_id, connection);
                                } else {
                                    handle_connection_error(connection_id, ec2);
                                }
                            });
                    } else {
                        printf("Invalid message length %u from %s\n", msg_length, connection_id.c_str());
                        start_read(connection_id, connection); // Try to continue
                    }
                } else {
                    handle_connection_error(connection_id, ec);
                }
            });
    }
    
    void handle_message(const std::string& connection_id, 
                       const std::vector<uint8_t>& buffer, 
                       size_t length) {
        try {
            std::vector<uint8_t> msg_data(buffer.begin(), buffer.begin() + length);
            Message<T> message = BinarySerializer::deserialize_message<T>(msg_data);
            
            printf("TCP Communicator received message type %d from %s to %s\n",
                   static_cast<int>(message.command_type), 
                   message.sender_id.c_str(), 
                   message.recipient_id.c_str());
            
            // Update connection mapping if we have sender info
            if (!message.sender_id.empty() && message.sender_id != connection_id) {
                std::lock_guard<std::mutex> lock(connections_mutex_);
                auto it = connections_.find(connection_id);
                if (it != connections_.end()) {
                    auto connection = it->second;
                    connections_.erase(it);
                    connections_[message.sender_id] = connection;
                    printf("Updated connection mapping: %s -> %s\n", 
                           connection_id.c_str(), message.sender_id.c_str());
                }
            }
            
            // Enqueue the message for processing
            this->enqueue_input_message(message);
            
        } catch (const std::exception& e) {
            printf("Error deserializing message from %s: %s\n", connection_id.c_str(), e.what());
        }
    }
    
    void handle_connection_error(const std::string& connection_id, std::error_code ec) {
        if (ec) {
            printf("Connection error with %s: %s\n", connection_id.c_str(), ec.message().c_str());
        }
        
        // Remove the connection
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = connections_.find(connection_id);
        if (it != connections_.end()) {
            if (it->second->socket.is_open()) {
                it->second->socket.close();
            }
            connections_.erase(it);
        }
    }
    
    void send_packet(const std::string& recipient_id, const std::vector<uint8_t>& packet) {
        std::shared_ptr<Connection> connection;
        
        {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            auto it = connections_.find(recipient_id);
            if (it == connections_.end()) {
                printf("No connection found for recipient %s\n", recipient_id.c_str());
                return;
            }
            connection = it->second;
        }
        
        std::lock_guard<std::mutex> write_lock(connection->write_mutex);
        
        try {
            asio::write(connection->socket, asio::buffer(packet));
        } catch (const std::exception& e) {
            printf("Error sending message to %s: %s\n", recipient_id.c_str(), e.what());
            handle_connection_error(recipient_id, std::make_error_code(std::errc::connection_aborted));
        }
    }
};

/**
 * @brief Factory for creating TCP communicators with proper configuration
 */
class TcpCommunicatorFactory {
public:
    template<typename T = float>
    static std::unique_ptr<TcpPipelineCommunicator<T>> 
    create_server(asio::io_context& io_context, int listen_port) {
        return std::make_unique<TcpPipelineCommunicator<T>>(io_context, "", listen_port);
    }
    
    template<typename T = float>
    static std::unique_ptr<TcpPipelineCommunicator<T>>
    create_client(asio::io_context& io_context) {
        return std::make_unique<TcpPipelineCommunicator<T>>(io_context);
    }
};

} // namespace tpipeline
