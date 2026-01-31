#pragma once

#include <sys/types.h>

#include <chrono>
#include <string>

namespace tnn {

enum class EventType : uint8_t { COMPUTE, COMMUNICATION, OTHER };

inline std::string event_type_to_string(EventType type) {
  switch (type) {
    case EventType::COMPUTE:
      return "COMPUTE";
    case EventType::COMMUNICATION:
      return "COMMUNICATION";
    case EventType::OTHER:
      return "OTHER";
    default:
      return "UNKNOWN";
  }
}

namespace Time = std::chrono;

typedef Time::system_clock Clock;

struct Event {
  EventType type;
  Clock::time_point start_time;
  Clock::time_point end_time;
  std::string name;
  std::string source;
};
}  // namespace tnn