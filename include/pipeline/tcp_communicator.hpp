/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "network_serialization.hpp"
#include "pipeline_communicator.hpp"
#include <asio.hpp>
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_map>

namespace tpipeline {

template <typename T = float>
class TcpPipelineCommunicator : public PipelineCommunicator<T> {
public:
  explicit TcpPipelineCommunicator(asio::io_context &io_context,
                                   const std::string &local_endpoint = "",
                                   int listen_port = 0)
      : io_context_(io_context), acceptor_(io_context), socket_(io_context),
        listen_port_(listen_port), local_endpoint_(local_endpoint),
        is_running_(false) {

    if (listen_port > 0) {
      std::cout << "Initializing TCP communicator on port " << listen_port
                << std::endl;
      start_server();
    }
  }

  ~TcpPipelineCommunicator() override { stop(); }

  void start_server() {
    if (listen_port_ <= 0) {
      throw std::invalid_argument("Listen port must be greater than 0");
    }

    asio::ip::tcp::endpoint endpoint(
        asio::ip::tcp::v4(), static_cast<asio::ip::port_type>(listen_port_));
    acceptor_.open(endpoint.protocol());
    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
    acceptor_.bind(endpoint);
    acceptor_.listen();

    is_running_ = true;
    accept_connections();

    std::cout << "TCP communicator listening on port " << listen_port_
              << std::endl;
  }

  void stop() {
    is_running_ = false;

    if (acceptor_.is_open()) {
      std::error_code ec;
      acceptor_.close(ec);
    }

    {
      std::lock_guard<std::mutex> lock(connections_mutex_);
      for (auto &[id, conn] : connections_) {
        if (conn->socket.is_open()) {
          std::error_code ec;
          conn->socket.close(ec);
        }
      }
      connections_.clear();
    }
  }

  void send_message(const Message<T> &message) override {
    auto serialized = BinarySerializer::serialize_message(message);

    std::vector<uint8_t> packet;
    uint32_t msg_length = static_cast<uint32_t>(serialized.size());
    packet.resize(4 + serialized.size());
    std::memcpy(packet.data(), &msg_length, 4);
    std::memcpy(packet.data() + 4, serialized.data(), serialized.size());

    send_packet(message.recipient_id, packet);
  }

  void flush_output_messages() override {
    std::lock_guard<std::mutex> lock(this->out_message_mutex_);

    if (this->out_message_queue_.empty()) {
      std::cout << "Warning: No output messages to flush\n" << std::endl;
      return;
    }

    while (!this->out_message_queue_.empty()) {
      auto &msg = this->out_message_queue_.front();
      send_message(msg);
      this->out_message_queue_.pop();
    }
  }

  bool connect_to_peer(const std::string &peer_id, const std::string &host,
                       int port) {
    try {
      auto connection = std::make_shared<Connection>(io_context_);

      asio::ip::tcp::resolver resolver(io_context_);
      auto endpoints = resolver.resolve(host, std::to_string(port));

      asio::connect(connection->socket, endpoints);

      {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        connections_[peer_id] = connection;
      }

      start_read(peer_id, connection);

      std::cout << "Connected to peer " << peer_id << " at " << host << ":"
                << port << std::endl;
      return true;

    } catch (const std::exception &e) {
      std::cout << "Failed to connect to peer " << peer_id << ": " << e.what()
                << std::endl;
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

    explicit Connection(asio::io_context &io_ctx)
        : socket(io_ctx), read_buffer(8192) {}
    explicit Connection(asio::ip::tcp::socket sock)
        : socket(std::move(sock)), read_buffer(8192) {}
  };

  asio::io_context &io_context_;
  asio::ip::tcp::acceptor acceptor_;
  asio::ip::tcp::socket socket_;

  int listen_port_;
  std::string local_endpoint_;
  std::atomic<bool> is_running_;

  std::unordered_map<std::string, std::shared_ptr<Connection>> connections_;
  std::mutex connections_mutex_;

  void accept_connections() {
    if (!is_running_)
      return;

    auto new_connection = std::make_shared<Connection>(io_context_);

    acceptor_.async_accept(
        new_connection->socket, [this, new_connection](std::error_code ec) {
          if (!ec && is_running_) {

            auto remote_endpoint = new_connection->socket.remote_endpoint();
            std::string temp_id = remote_endpoint.address().to_string() + ":" +
                                  std::to_string(remote_endpoint.port());

            {
              std::lock_guard<std::mutex> lock(connections_mutex_);
              connections_[temp_id] = new_connection;
            }

            start_read(temp_id, new_connection);
            std::cout << "Accepted connection from " << temp_id << std::endl;
          }

          accept_connections();
        });
  }

  void start_read(const std::string &connection_id,
                  std::shared_ptr<Connection> connection) {
    if (!is_running_)
      return;

    asio::async_read(
        connection->socket, asio::buffer(connection->read_buffer.data(), 4),
        [this, connection_id, connection](std::error_code ec,
                                          [[maybe_unused]] std::size_t length) {
          if (!ec && is_running_) {

            uint32_t msg_length;
            std::memcpy(&msg_length, connection->read_buffer.data(), 4);

            if (msg_length > 0 && msg_length < 2048 * 2048) {

              if (connection->read_buffer.size() < msg_length) {
                connection->read_buffer.resize(msg_length);
              }

              asio::async_read(
                  connection->socket,
                  asio::buffer(connection->read_buffer.data(), msg_length),
                  [this, connection_id, connection,
                   msg_length](std::error_code ec2, std::size_t) {
                    if (!ec2 && is_running_) {
                      handle_message(connection_id, connection->read_buffer,
                                     msg_length);

                      start_read(connection_id, connection);
                    } else {
                      handle_connection_error(connection_id, ec2);
                    }
                  });
            } else {
              std::cout << "Invalid message length " << msg_length << " from "
                        << connection_id << std::endl;
              start_read(connection_id, connection);
            }
          } else {
            handle_connection_error(connection_id, ec);
          }
        });
  }

  void handle_message(const std::string &connection_id,
                      const std::vector<uint8_t> &buffer, size_t length) {
    try {
      std::vector<uint8_t> msg_data(buffer.begin(), buffer.begin() + length);
      Message<T> message = BinarySerializer::deserialize_message<T>(msg_data);

      this->enqueue_input_message(message);

    } catch (const std::exception &e) {
      std::cout << "Error deserializing message from " << connection_id << ": "
                << e.what() << std::endl;
    }
  }

  void handle_connection_error(const std::string &connection_id,
                               std::error_code ec) {
    if (ec) {
      if (ec == asio::error::eof) {
        std::cout << "Connection closed by peer: " << connection_id
                  << std::endl;
      } else if (ec == asio::error::connection_reset) {
        std::cout << "Connection reset by peer: " << connection_id << std::endl;
      } else if (ec == asio::error::operation_aborted) {
        std::cout << "Connection operation aborted: " << connection_id
                  << std::endl;
      } else if (ec == asio::error::connection_refused) {
        std::cout << "Connection refused by peer: " << connection_id
                  << std::endl;
      } else {
        std::cout << "Connection error with " << connection_id << ": "
                  << ec.message() << std::endl;
      }
    }

    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto it = connections_.find(connection_id);
    if (it != connections_.end()) {
      if (it->second->socket.is_open()) {
        it->second->socket.close();
      }
      connections_.erase(it);
    }
  }

  void send_packet(const std::string &recipient_id,
                   const std::vector<uint8_t> &packet) {
    std::shared_ptr<Connection> connection;

    {
      std::lock_guard<std::mutex> lock(connections_mutex_);
      auto it = connections_.find(recipient_id);
      if (it == connections_.end()) {
        std::cout << "No connection found for recipient " << recipient_id
                  << std::endl;
        return;
      }
      connection = it->second;
    }

    std::lock_guard<std::mutex> write_lock(connection->write_mutex);

    try {
      asio::write(connection->socket, asio::buffer(packet));
    } catch (const std::exception &e) {
      std::cout << "Error sending message to " << recipient_id << ": "
                << e.what() << std::endl;
      handle_connection_error(
          recipient_id, std::make_error_code(std::errc::connection_aborted));
    }
  }
};

class TcpCommunicatorFactory {
public:
  template <typename T = float>
  static std::unique_ptr<TcpPipelineCommunicator<T>>
  create_server(asio::io_context &io_context, int listen_port) {
    return std::make_unique<TcpPipelineCommunicator<T>>(io_context, "",
                                                        listen_port);
  }

  template <typename T = float>
  static std::unique_ptr<TcpPipelineCommunicator<T>>
  create_client(asio::io_context &io_context) {
    return std::make_unique<TcpPipelineCommunicator<T>>(io_context);
  }
};

} // namespace tpipeline
