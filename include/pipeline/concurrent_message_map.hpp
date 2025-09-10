/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "message.hpp"
#include <mutex>
#include <queue>

#ifdef USE_TBB
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/concurrent_unordered_map.h>
#endif

namespace tpipeline {

template <typename T = float> class ConcurrentMessageMap {
private:
#ifdef USE_TBB
  tbb::concurrent_unordered_map<CommandType, tbb::concurrent_queue<Message<T>>>
      queues_;
#else
  struct QueueWithMutex {
    std::queue<Message<T>> queue;
    mutable std::mutex mutex;

    QueueWithMutex() = default;
    QueueWithMutex(const QueueWithMutex &other) {
      std::lock_guard<std::mutex> lock(other.mutex);
      queue = other.queue;
    }
    QueueWithMutex &operator=(const QueueWithMutex &other) {
      if (this != &other) {
        std::lock(mutex, other.mutex);
        std::lock_guard<std::mutex> lock1(mutex, std::adopt_lock);
        std::lock_guard<std::mutex> lock2(other.mutex, std::adopt_lock);
        queue = other.queue;
      }
      return *this;
    }
  };

  std::unordered_map<CommandType, QueueWithMutex> queues_;
  mutable std::mutex map_mutex_;
#endif

public:
  void push(CommandType type, const Message<T> &message) {
#ifdef USE_TBB
    queues_[type].push(message);
#else
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    auto &queue_entry = queues_[type];
    std::lock_guard<std::mutex> queue_lock(queue_entry.mutex);
    queue_entry.queue.push(message);
#endif
  }

  bool pop(CommandType type, Message<T> &message) {
#ifdef USE_TBB
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      return it->second.try_pop(message);
    }
    return false;
#else
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      std::lock_guard<std::mutex> queue_lock(it->second.mutex);
      if (!it->second.queue.empty()) {
        message = it->second.queue.front();
        it->second.queue.pop();
        return true;
      }
    }
    return false;
#endif
  }

  size_t size(CommandType type) const {
#ifdef USE_TBB
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      return it->second.unsafe_size();
    }
    return 0;
#else
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      std::lock_guard<std::mutex> queue_lock(it->second.mutex);
      return it->second.queue.size();
    }
    return 0;
#endif
  }

  bool empty(CommandType type) const { return size(type) == 0; }

  bool has_any_message() const {
#ifdef USE_TBB
    for (const auto &pair : queues_) {
      if (pair.second.unsafe_size() > 0) {
        return true;
      }
    }
    return false;
#else
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    for (const auto &pair : queues_) {
      std::lock_guard<std::mutex> queue_lock(pair.second.mutex);
      if (!pair.second.queue.empty()) {
        return true;
      }
    }
    return false;
#endif
  }

  std::vector<Message<T>> pop_all(CommandType type) {
    std::vector<Message<T>> messages;
#ifdef USE_TBB
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      Message<T> message;
      while (it->second.try_pop(message)) {
        messages.push_back(std::move(message));
      }
    }
#else
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    auto it = queues_.find(type);
    if (it != queues_.end()) {
      std::lock_guard<std::mutex> queue_lock(it->second.mutex);
      while (!it->second.queue.empty()) {
        messages.push_back(it->second.queue.front());
        it->second.queue.pop();
      }
    }
#endif
    return messages;
  }

  void clear() {
#ifdef USE_TBB
    queues_.clear();
#else
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    queues_.clear();
#endif
  }
};

} // namespace tpipeline
