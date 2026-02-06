#pragma once

#include <mutex>
#include <shared_mutex>
#include <unordered_map>

template <typename Key, typename Value>
class ConcurrentMap {
private:
  std::unordered_map<Key, Value> map_;
  mutable std::shared_mutex mutex_;

public:
  void emplace(const Key& key, Value&& value) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    map_.emplace(key, std::move(value));
  }

  bool try_get(const Key& key, Value& value) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      value = it->second;
      return true;
    }
    return false;
  }

  void erase(const Key& key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    map_.erase(key);
  }

  bool contains(const Key& key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return map_.find(key) != map_.end();
  }

  Value& operator[](const Key& key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    return map_[key];
  }

  const Value& operator[](const Key& key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return map_[key];
  }

  size_t size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return map_.size();
  }
};
