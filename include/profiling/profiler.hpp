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

  void merge(const Profiler &other) {
    std::lock_guard<std::mutex> other_lock(other.event_mutex_);
    std::lock_guard<std::mutex> this_lock(event_mutex_);
    auto duration = other.profiler_start_time_ - profiler_start_time_;
    const auto &events = other.events_;
    for (const auto &event : events) {
      Event adjusted_event = event;
      adjusted_event.start_time -= duration;
      adjusted_event.end_time -= duration;
      events_.push_back(adjusted_event);
    }
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

  void reset() {
    std::lock_guard<std::mutex> lock(event_mutex_);
    events_.clear();
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

class GlobalProfiler {
  static Profiler global_profiler_;

public:
  static Profiler &get_profiler() { return global_profiler_; }

  static void reset_profiler() { global_profiler_ = Profiler(); }

  static void merge_profiler(const Profiler &profiler) { global_profiler_.merge(profiler); }

  static void init_start_time(Clock::time_point time) { global_profiler_.init_start_time(time); }

  static Clock::time_point start_time() { return global_profiler_.start_time(); }

  static void add_event(const Event &event) { global_profiler_.add_event(event); }

  static void sort_events(std::function<bool(const Event &a, const Event &b)> comp =
                              [](const Event &a, const Event &b) {
                                return a.start_time == b.start_time ? a.end_time > b.end_time
                                                                    : a.start_time < b.start_time;
                              }) {
    global_profiler_.sort(comp);
  }

  static std::vector<Event> get_events() { return global_profiler_.get_events(); }
};

} // namespace tnn