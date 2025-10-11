/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */

#pragma once

#include "communicator.hpp"

namespace tpipeline {
class InProcessPipelineCommunicator : public PipelineCommunicator {
public:
  InProcessPipelineCommunicator() : shutdown_flag_(false) { start_delivery_thread(); }

  ~InProcessPipelineCommunicator() {
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

  void start_delivery_thread() {
    delivery_thread_ = std::thread([this]() {
      while (!shutdown_flag_) {
        std::unique_lock<std::mutex> lock(this->out_message_mutex_);
        outgoing_cv_.wait(lock,
                          [this]() { return !this->out_message_queue_.empty() || shutdown_flag_; });

        if (shutdown_flag_) {
          break;
        }

        if (this->out_message_queue_.empty()) {
          continue;
        }

        Message outgoing = std::move(this->out_message_queue_.front());
        this->out_message_queue_.pop();
        lock.unlock();

        try {
          std::lock_guard<std::mutex> comm_lock(communicators_mutex_);
          auto it = communicators_.find(outgoing.header.recipient_id);
          if (it != communicators_.end() && it->second) {
            it->second->enqueue_input_message(outgoing);
          }
        } catch (const std::exception &e) {
          std::cerr << "Error delivering message: " << e.what() << std::endl;
        }
      }
    });
  }

  void flush_output_messages() override {
    while (this->has_output_message()) {
      Message message = this->out_message_queue_.front();
      this->out_message_queue_.pop();

      send_message(message);
    }
  }

  void register_communicator(const std::string &recipient_id,
                             std::shared_ptr<PipelineCommunicator> communicator) {
    std::lock_guard<std::mutex> lock(communicators_mutex_);
    communicators_[recipient_id] = communicator;
  }

private:
  std::condition_variable outgoing_cv_;
  std::thread delivery_thread_;
  std::unordered_map<std::string, std::shared_ptr<PipelineCommunicator>> communicators_;
  mutable std::mutex communicators_mutex_;
  std::atomic<bool> shutdown_flag_;
};

} // namespace tpipeline