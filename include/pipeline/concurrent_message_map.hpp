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

template <typename T = float> class ConcurrentMessageMap {
private:
  tbb::concurrent_unordered_map<CommandType, tbb::concurrent_queue<Message<T>>> queues_;

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

  void push(CommandType type, const Message<T> &message) { queues_[type].push(message); }

  bool pop(CommandType type, Message<T> &message) {
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      return it->second.try_pop(message);
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

  bool has_any_message() const {
    for (const auto &pair : queues_) {
      if (pair.second.unsafe_size() > 0) {
        return true;
      }
    }
    return false;
  }

  std::vector<Message<T>> pop_all(CommandType type) {
    std::vector<Message<T>> messages;
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      Message<T> message;
      while (it->second.try_pop(message)) {
        messages.push_back(std::move(message));
      }
    }
    return messages;
  }

  void clear() {
    for (auto &pair : queues_) {
      Message<T> dummy;
      while (pair.second.try_pop(dummy)) {
      }
    }
    queues_.clear();
  }
};

} // namespace tpipeline
