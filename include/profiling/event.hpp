#pragma once

#include <sys/types.h>

#include <chrono>
#include <cstdint>
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

// Serialization (const version for Writer)
template <typename Archiver>
void archive(Archiver &archiver, const Event &event) {
  archiver(static_cast<int64_t>(event.start_time.time_since_epoch().count()));
  archiver(static_cast<int64_t>(event.end_time.time_since_epoch().count()));
  archiver(static_cast<uint8_t>(event.type));
  archiver(event.name);
  archiver(event.source);
}

// Deserialization (non-const version for Reader)
template <typename Archiver>
void archive(Archiver &archiver, Event &event) {
  int64_t start_time_count = event.start_time.time_since_epoch().count();
  int64_t end_time_count = event.end_time.time_since_epoch().count();
  uint8_t type_value = static_cast<uint8_t>(event.type);
  archiver(start_time_count);
  archiver(end_time_count);
  archiver(type_value);
  archiver(event.name);
  archiver(event.source);
  event.start_time = Clock::time_point(Clock::duration(start_time_count));
  event.end_time = Clock::time_point(Clock::duration(end_time_count));
  event.type = static_cast<EventType>(type_value);
}

}  // namespace tnn