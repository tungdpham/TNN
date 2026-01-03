#pragma once

#include "event.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
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

  void add_event(const Event &event) { events_.push_back(event); }

  const std::vector<Event> &get_events() const { return events_; }

  void sort(std::function<bool(const Event &a, const Event &b)> comp =
                [](const Event &a, const Event &b) { return a.start_time < b.start_time; }) {
    std::sort(events_.begin(), events_.end(), comp);
  }

  void init_start_time(Clock::time_point time) { profiler_start_time_ = time; }

  Clock::time_point start_time() const { return profiler_start_time_; }

private:
  std::vector<Event> events_;
  Clock::time_point profiler_start_time_;
};
} // namespace tnn