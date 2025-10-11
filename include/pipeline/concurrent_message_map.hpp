/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "message.hpp"
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/concurrent_unordered_map.h>

namespace tpipeline {

class ConcurrentMessageMap {
private:
  std::atomic<size_t> total_message_count_{0};
  tbb::concurrent_unordered_map<CommandType, tbb::concurrent_queue<Message>> queues_;

public:
  ConcurrentMessageMap() = default;

  ConcurrentMessageMap(const ConcurrentMessageMap &other) = delete;
  ConcurrentMessageMap &operator=(const ConcurrentMessageMap &other) = delete;

  ConcurrentMessageMap(ConcurrentMessageMap &&other) noexcept : queues_(std::move(other.queues_)) {}

  ConcurrentMessageMap &operator=(ConcurrentMessageMap &&other) noexcept {
    if (this != &other) {
      clear();
      queues_ = std::move(other.queues_);
    }
    return *this;
  }

  ~ConcurrentMessageMap() { clear(); }

  void push(CommandType type, const Message &message) {
    queues_[type].push(message);
    total_message_count_.fetch_add(1, std::memory_order_relaxed);
  }

  bool pop(CommandType type, Message &message) {
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      if (it->second.try_pop(message)) {
        total_message_count_.fetch_sub(1, std::memory_order_relaxed);
        return true;
      }
    }
    return false;
  }

  size_t size(CommandType type) const {
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      return it->second.unsafe_size();
    }
    return 0;
  }

  bool empty(CommandType type) const { return size(type) == 0; }

  bool empty() const { return total_message_count_.load(std::memory_order_relaxed) == 0; }

  std::vector<Message> pop_all(CommandType type) {
    std::vector<Message> messages;
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      Message message;
      while (it->second.try_pop(message)) {
        messages.push_back(std::move(message));
      }
    }
    return messages;
  }

  void clear() {
    for (auto &pair : queues_) {
      Message dummy;
      while (pair.second.try_pop(dummy)) {
      }
    }
    queues_.clear();
  }
};

} // namespace tpipeline
