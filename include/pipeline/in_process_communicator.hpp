/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include "communicator.hpp"

namespace tnn {
class InProcessCommunicator : public Communicator {
public:
  InProcessCommunicator() : shutdown_flag_(false) {
    delivery_thread_ = std::thread(&InProcessCommunicator::delivery_loop, this);
  }

  ~InProcessCommunicator() {
    shutdown_flag_ = true;
    outgoing_cv_.notify_one();
    if (delivery_thread_.joinable()) {
      delivery_thread_.join();
    }
  }

  void send_message(const Message &message) override {
    if (message.header.recipient_id.empty()) {
      throw std::runtime_error("Message recipient_id is empty");
    }
    this->enqueue_output_message(message);

    outgoing_cv_.notify_one();
  }

  void delivery_loop() {
    while (!shutdown_flag_) {
      std::unique_lock<std::mutex> lock(this->out_message_mutex_);
      outgoing_cv_.wait(lock,
                        [this]() { return !this->out_message_queue_.empty() || shutdown_flag_; });

      if (shutdown_flag_) {
        break;
      }

      while (!this->out_message_queue_.empty()) {
        lock.lock();
        auto msg = this->out_message_queue_.front();
        this->out_message_queue_.pop();
        lock.unlock();
        try {
          Endpoint recipient_endpoint = this->get_recipient(msg.header.recipient_id);
          Communicator *recipient_comm =
              recipient_endpoint.get_parameter<Communicator *>("communicator");
          if (recipient_comm == nullptr) {
            throw std::runtime_error("Recipient communicator is null for " +
                                     msg.header.recipient_id);
          }
          recipient_comm->enqueue_input_message(msg);
        } catch (const std::exception &e) {
          std::cerr << "Failed to deliver message to " << msg.header.recipient_id << ": "
                    << e.what() << std::endl;
        }
      }
    }
  }

  void flush_output_messages() override {
    while (this->has_output_message()) {
      Message message = this->out_message_queue_.front();
      this->out_message_queue_.pop();

      send_message(message);
    }
  }

  bool connect_to_endpoint(const std::string &peer_id, const Endpoint &endpoint) override {
    // No-op for in-process communicator
    return true;
  }

  bool disconnect_from_endpoint(const std::string &peer_id, const Endpoint &endpoint) override {
    // No-op for in-process communicator
    return true;
  }

private:
  std::condition_variable outgoing_cv_;
  std::thread delivery_thread_;
  mutable std::mutex communicators_mutex_;
  std::atomic<bool> shutdown_flag_;
};

} // namespace tnn