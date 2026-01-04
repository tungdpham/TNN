#pragma once

#include "event.hpp"
#include "profiler.hpp"
#include <chrono>

namespace tnn {
class ProfilerAggregator {
public:
  ProfilerAggregator() = default;
  ~ProfilerAggregator() = default;

  static ProfilerAggregator &instance() {
    static ProfilerAggregator instance;
    return instance;
  }

  void add_profiler(const Profiler &profiler, std::string source_name = "") {
    auto duration = profiler.start_time() - global_start_time_;
    const auto &events = profiler.get_events();
    for (const auto &event : events) {
      Event adjusted_event = event;
      adjusted_event.start_time -= duration;
      adjusted_event.end_time -= duration;
      aggregated_events_.push_back(adjusted_event);
    }
  }

  void sort(std::function<bool(const Event &a, const Event &b)> comp = [](const Event &a,
                                                                          const Event &b) {
    return a.start_time == b.start_time ? a.end_time > b.end_time : a.start_time < b.start_time;
  }) {
    std::sort(aggregated_events_.begin(), aggregated_events_.end(), comp);
  }

  void set_global_start_time(const Clock::time_point &time) { global_start_time_ = time; }

  Clock::time_point get_global_start_time() const { return global_start_time_; }

  const std::vector<Event> &get_aggregated_events() const { return aggregated_events_; }

private:
  Clock::time_point global_start_time_;
  std::vector<Event> aggregated_events_;
};
} // namespace tnn