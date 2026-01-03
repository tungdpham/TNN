#pragma once

#include "device/device_manager.hpp"
#include "endian.hpp"
#include "message.hpp"
#include "packet.hpp"
#include "profiling/profiler.hpp"
#include "tbuffer.hpp"
#include "tensor/tensor.hpp"
#include <cstdint>
#include <sys/types.h>

template <typename VariantType, typename T, uint64_t index = 0> constexpr uint64_t variant_index() {
  static_assert(std::variant_size_v<VariantType> > index, "Type not found in variant");
  if constexpr (index == std::variant_size_v<VariantType>) {
    return index;
  } else if constexpr (std::is_same_v<std::variant_alternative_t<index, VariantType>, T>) {
    return index;
  } else {
    return variant_index<VariantType, T, index + 1>();
  }
}

/**
 * Very Important note: size_t is platform dependent. On 64-bit systems, it is usually
 * 8 bytes, while on 32-bit systems, it is 4 bytes. For serialization, we need a fixed size
 * type to ensure compatibility across different platforms. Here, we use uint64_t for sizes
 * and counts, which is 8 bytes on all platforms.
 */
namespace tnn {
class BinarySerializer {
public:
  template <typename T = float> static void serialize(const Tensor<T> &tensor, TBuffer &buffer) {
    std::vector<size_t> shape = tensor.shape();
    uint64_t shape_size = static_cast<uint64_t>(shape.size());
    buffer.append(shape_size);
    for (size_t dim : shape) {
      buffer.append(static_cast<uint64_t>(dim));
    }
    buffer.append(tensor.data(), tensor.size(), true);
  }

  static void serialize(const Event &event, TBuffer &buffer) {
    buffer.append(static_cast<int64_t>(event.start_time.time_since_epoch().count()));
    buffer.append(static_cast<int64_t>(event.end_time.time_since_epoch().count()));
    buffer.append(static_cast<uint8_t>(event.type));
    buffer.append(event.name);
    buffer.append(event.source);
  }

  static void serialize(const Profiler &profiler, TBuffer &buffer) {
    auto events = profiler.get_events();
    buffer.append(static_cast<int64_t>(profiler.start_time().time_since_epoch().count()));
    int64_t event_count = static_cast<int64_t>(events.size());
    buffer.append(event_count);
    for (const auto &event : events) {
      serialize(event, buffer);
    }
  }

  static void serialize(const PacketHeader &header, TBuffer &buffer) {
    buffer.append(header.PROTOCOL_VERSION);
    buffer.append(header.endianess);
    buffer.append(header.length);
    buffer.append(header.msg_length);
    buffer.append(header.msg_serial_id);
    buffer.append(header.packet_offset);
    buffer.append(header.total_packets);
    buffer.append<uint8_t>(static_cast<uint8_t>(header.compression_type));
  }

  static void serialize(const MessageHeader &header, TBuffer &buffer) {
    buffer.append(header.recipient_id);
    buffer.append(header.sender_id);
    buffer.append(header.command_type);
  }

  static void serialize(const MessageData &data, TBuffer &buffer) {
    buffer.append(data.payload_type);
    if (std::holds_alternative<std::monostate>(data.payload)) {
      // No additional data to write
    } else if (std::holds_alternative<PooledJob<float>>(data.payload)) {
      const auto &job = std::get<PooledJob<float>>(data.payload);
      buffer.append(static_cast<uint64_t>(job->micro_batch_id));
      serialize<float>(job->data, buffer);
    } else if (std::holds_alternative<std::string>(data.payload)) {
      const auto &str = std::get<std::string>(data.payload);
      buffer.append(str);
    } else if (std::holds_alternative<bool>(data.payload)) {
      const auto &flag = std::get<bool>(data.payload);
      buffer.append(static_cast<uint8_t>(flag ? 1 : 0));
    } else if (std::holds_alternative<Profiler>(data.payload)) {
      const auto &profiler = std::get<Profiler>(data.payload);
      serialize(profiler, buffer);
    } else {
      throw std::runtime_error("Unsupported payload type in MessageData");
    }
  }

  static void serialize(const Message &message, TBuffer &buffer) {
    serialize(message.header(), buffer);
    serialize(message.data(), buffer);
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, PacketHeader &header) {
    buffer.read(offset, header.PROTOCOL_VERSION);
    buffer.read(offset, header.endianess);
    buffer.read(offset, header.length);
    buffer.read(offset, header.msg_length);
    buffer.read(offset, header.msg_serial_id);
    buffer.read(offset, header.packet_offset);
    buffer.read(offset, header.total_packets);
    buffer.read<uint8_t>(offset, reinterpret_cast<uint8_t &>(header.compression_type));
    if (header.endianess != get_system_endianness()) {
      bswap(header.length);
      bswap(header.msg_length);
      bswap(header.msg_serial_id);
      bswap(header.packet_offset);
      bswap(header.total_packets);
    }
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, MessageHeader &header) {
    buffer.read(offset, header.recipient_id);
    buffer.read(offset, header.sender_id);
    uint16_t cmd_type;
    buffer.read<uint16_t>(offset, cmd_type);
    header.command_type = static_cast<CommandType>(cmd_type);
  }

  template <typename T = float>
  static void deserialize(const TBuffer &buffer, size_t &offset, Tensor<T> &tensor) {
    uint64_t shape_size;
    buffer.read(offset, shape_size);
    std::vector<uint64_t> shape(shape_size);
    for (uint64_t i = 0; i < shape_size; ++i) {
      buffer.read<uint64_t>(offset, shape[i]);
    }
    tensor.ensure(shape, &getCPU());
    if (tensor.size() > 0) {
      buffer.read(offset, tensor.data(), tensor.size(), true);
      offset += tensor.size() * sizeof(T);
    }
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, PooledJob<float> &job) {
    buffer.read<uint64_t>(offset, reinterpret_cast<uint64_t &>(job->micro_batch_id));
    deserialize(buffer, offset, job->data);
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, std::string &str) {
    buffer.read(offset, str, true);
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, bool &flag) {
    uint8_t value;
    buffer.read(offset, value);
    flag = (value != 0);
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, Event &event) {
    int64_t start_time_count;
    int64_t end_time_count;
    uint8_t type_value;
    buffer.read(offset, start_time_count);
    buffer.read(offset, end_time_count);
    buffer.read(offset, type_value);
    event.start_time = Clock::time_point(Clock::duration(start_time_count));
    event.end_time = Clock::time_point(Clock::duration(end_time_count));
    event.type = static_cast<EventType>(type_value);
    buffer.read(offset, event.name, true);
    buffer.read(offset, event.source, true);
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, Profiler &profiler) {
    int64_t start_time_count;
    buffer.read(offset, start_time_count);
    profiler.init_start_time(Clock::time_point(Clock::duration(start_time_count)));
    uint64_t event_count;
    buffer.read(offset, event_count);
    for (uint64_t i = 0; i < event_count; ++i) {
      Event event;
      deserialize(buffer, offset, event);
      profiler.add_event(event);
    }
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, MessageData &data) {
    // Determine payload type based on payload_type
    uint64_t payload_type;
    buffer.read(offset, payload_type);
    data.payload_type = payload_type;
    switch (data.payload_type) {
    case variant_index<PayloadType, std::monostate>(): // std::monostate
      data.payload = std::monostate{};
      break;
    case variant_index<PayloadType, PooledJob<float>>(): { // Job<float>
      PooledJob<float> job =
          JobPool<float>::instance().get_job((buffer.size() - offset) / sizeof(float));
      deserialize(buffer, offset, job);
      data.payload = std::move(job);
    } break;
    case variant_index<PayloadType, std::string>(): { // std::string
      std::string str;
      deserialize(buffer, offset, str);
      data.payload = std::move(str);
    } break;
    case variant_index<PayloadType, bool>(): { // bool
      bool flag;
      deserialize(buffer, offset, flag);
      data.payload = flag;
    } break;
    case variant_index<PayloadType, Profiler>(): { // Profiler
      Profiler profiler;
      deserialize(buffer, offset, profiler);
      data.payload = std::move(profiler);
    } break;
    default:
      throw std::runtime_error("Unsupported payload type in MessageData deserialization");
    }
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, Message &message) {
    deserialize(buffer, offset, message.header());
    deserialize(buffer, offset, message.data());
  }

}; // namespace BinarySerializer

} // namespace tnn