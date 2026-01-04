#pragma once

#include "event.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <vector>

namespace tnn {
class Profiler {
public:
  Profiler() = default;
  ~Profiler() = default;

  static Profiler &instance() {
    static Profiler instance;
    return instance;
  }

  Profiler(const Profiler &other) {
    std::lock_guard<std::mutex> other_lock(other.event_mutex_);
    std::lock_guard<std::mutex> this_lock(event_mutex_);
    events_ = other.events_;
    profiler_start_time_ = other.profiler_start_time_;
  }

  Profiler &operator=(const Profiler &other) {
    if (this != &other) {
      std::lock_guard<std::mutex> other_lock(other.event_mutex_);
      std::lock_guard<std::mutex> this_lock(event_mutex_);
      events_ = other.events_;
      profiler_start_time_ = other.profiler_start_time_;
    }
    return *this;
  }

  Profiler(Profiler &&other) noexcept {
    std::lock_guard<std::mutex> other_lock(other.event_mutex_);
    std::lock_guard<std::mutex> this_lock(event_mutex_);
    events_ = std::move(other.events_);
    profiler_start_time_ = other.profiler_start_time_;
  }

  Profiler &operator=(Profiler &&other) noexcept {
    if (this != &other) {
      std::lock_guard<std::mutex> other_lock(other.event_mutex_);
      std::lock_guard<std::mutex> this_lock(event_mutex_);
      events_ = std::move(other.events_);
      profiler_start_time_ = other.profiler_start_time_;
    }
    return *this;
  }

  void add_event(const Event &event) {
    std::lock_guard<std::mutex> lock(event_mutex_);
    events_.push_back(event);
  }

  std::vector<Event> get_events() const {
    std::lock_guard<std::mutex> lock(event_mutex_);
    return events_;
  }

  void sort(std::function<bool(const Event &a, const Event &b)> comp = [](const Event &a,
                                                                          const Event &b) {
    return a.start_time == b.start_time ? a.end_time > b.end_time : a.start_time < b.start_time;
  }) {
    std::lock_guard<std::mutex> lock(event_mutex_);
    std::sort(events_.begin(), events_.end(), comp);
  }

  void init_start_time(Clock::time_point time) {
    std::lock_guard<std::mutex> lock(start_mutex_);
    profiler_start_time_ = time;
  }

  Clock::time_point start_time() const {
    std::lock_guard<std::mutex> lock(start_mutex_);
    return profiler_start_time_;
  }

private:
  mutable std::mutex event_mutex_;
  std::vector<Event> events_;
  mutable std::mutex start_mutex_;
  Clock::time_point profiler_start_time_;
};
} // namespace tnn