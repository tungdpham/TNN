#pragma once

#include "device/device_manager.hpp"
#include "device/dptr.hpp"
#include "device/pool_allocator.hpp"
#include "endian.hpp"
#include "message.hpp"
#include "packet.hpp"
#include "profiling/profiler.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"
#include <concepts>
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
 * Very Important Note: size_t is platform dependent. On 64-bit systems, it is usually
 * 8 bytes, while on 32-bit systems, it is 4 bytes. For serialization, we need a fixed size
 * type to ensure compatibility across different platforms. Here, we use uint64_t for sizes
 * and counts, which is 8 bytes on all platforms.
 */
namespace tnn {

template <typename T>
concept Buffer = requires(T t, const T ct, size_t &offset, uint8_t *ptr, size_t len, dptr &dev_ptr,
                          const dptr &cdev_ptr, std::string &str, uint64_t &val) {
  t.write(offset, val);
  t.write(offset, ptr, len);
  t.write(offset, cdev_ptr, len);
  t.write(offset, static_cast<const std::string &>(str));

  ct.read(offset, val);
  ct.read(offset, ptr, len);
  ct.read(offset, dev_ptr, len);
  ct.read(offset, str);

  { ct.size() } -> std::convertible_to<size_t>;
};

class BinarySerializer {
private:
  static bool deserialize_to_gpu_;

public:
  static void set_deserialize_to_gpu(bool flag) { deserialize_to_gpu_ = flag; }

  template <Buffer BufferType>
  static void serialize(BufferType &buffer, size_t &offset, const Tensor &tensor) {
    DType_t dtype = tensor->data_type();
    buffer.template write<uint32_t>(offset, static_cast<uint32_t>(dtype));
    std::vector<size_t> shape = tensor->shape();
    uint64_t shape_size = static_cast<uint64_t>(shape.size());
    buffer.write(offset, shape_size);
    for (size_t dim : shape) {
      buffer.write(offset, static_cast<uint64_t>(dim));
    }
    const auto &dptr = tensor->data_ptr();
    size_t byte_size = tensor->size() * get_dtype_size(dtype);
    buffer.write(offset, dptr, byte_size);
  }

  template <Buffer BufferType>
  static void serialize(BufferType &buffer, size_t &offset, const Event &event) {
    buffer.write(offset, static_cast<int64_t>(event.start_time.time_since_epoch().count()));
    buffer.write(offset, static_cast<int64_t>(event.end_time.time_since_epoch().count()));
    buffer.write(offset, static_cast<uint8_t>(event.type));
    buffer.write(offset, event.name);
    buffer.write(offset, event.source);
  }

  template <Buffer BufferType>
  static void serialize(BufferType &buffer, size_t &offset, const Profiler &profiler) {
    auto events = profiler.get_events();
    buffer.write(offset, static_cast<int64_t>(profiler.start_time().time_since_epoch().count()));
    int64_t event_count = static_cast<int64_t>(events.size());
    buffer.write(offset, event_count);
    for (const auto &event : events) {
      serialize(buffer, offset, event);
    }
  }

  template <Buffer BufferType>
  static void serialize(BufferType &buffer, size_t &offset, const PacketHeader &header) {
    buffer.write(offset, header.PROTOCOL_VERSION);
    buffer.write(offset, header.endianess);
    buffer.write(offset, header.type);
    buffer.write(offset, header.length);
    buffer.write(offset, header.msg_length);
    buffer.write(offset, header.msg_serial_id);
    buffer.write(offset, header.packet_offset);
    buffer.write(offset, header.total_packets);
    buffer.write(offset, static_cast<uint8_t>(header.compression_type));
  }

  template <Buffer BufferType>
  static void serialize(BufferType &buffer, size_t &offset, const MessageHeader &header) {
    buffer.write(offset, static_cast<uint16_t>(header.command_type));
  }

  template <Buffer BufferType>
  static void serialize(BufferType &buffer, size_t &offset, const MessageData &data) {
    buffer.write(offset, data.payload_type);
    if (std::holds_alternative<std::monostate>(data.payload)) {
      // No additional data to write
    } else if (std::holds_alternative<Job>(data.payload)) {
      const auto &job = std::get<Job>(data.payload);
      buffer.write(offset, static_cast<uint64_t>(job.mb_id));
      serialize(buffer, offset, job.data);
    } else if (std::holds_alternative<std::string>(data.payload)) {
      const auto &str = std::get<std::string>(data.payload);
      buffer.write(offset, str);
    } else if (std::holds_alternative<bool>(data.payload)) {
      const auto &flag = std::get<bool>(data.payload);
      buffer.write(offset, static_cast<uint8_t>(flag ? 1 : 0));
    } else if (std::holds_alternative<Profiler>(data.payload)) {
      const auto &profiler = std::get<Profiler>(data.payload);
      serialize(buffer, offset, profiler);
    } else if (std::holds_alternative<StageConfig>(data.payload)) {
      const auto &stage_config = std::get<StageConfig>(data.payload);
      std::string json_dump = stage_config.to_json().dump();
      buffer.write(offset, json_dump);
    } else {
      throw std::runtime_error("Unsupported payload type in MessageData");
    }
  }

  template <Buffer BufferType>
  static void serialize(BufferType &buffer, size_t &offset, const Message &message) {
    serialize(buffer, offset, message.header());
    serialize(buffer, offset, message.data());
  }

  template <Buffer BufferType>
  static void deserialize(const BufferType &buffer, size_t &offset, PacketHeader &header) {
    buffer.read(offset, header.PROTOCOL_VERSION);
    buffer.read(offset, header.endianess);
    buffer.template read<PacketType>(offset, header.type);
    buffer.read(offset, header.length);
    buffer.read(offset, header.msg_length);
    buffer.read(offset, header.msg_serial_id);
    buffer.read(offset, header.packet_offset);
    buffer.read(offset, header.total_packets);
    buffer.template read<uint8_t>(offset, reinterpret_cast<uint8_t &>(header.compression_type));
    if (header.endianess != get_system_endianness()) {
      bswap(header.type);
      bswap(header.length);
      bswap(header.msg_length);
      bswap(header.msg_serial_id);
      bswap(header.packet_offset);
      bswap(header.total_packets);
    }
  }

  template <Buffer BufferType>
  static void deserialize(const BufferType &buffer, size_t &offset, MessageHeader &header) {
    uint16_t cmd_type;
    buffer.template read<uint16_t>(offset, cmd_type);
    header.command_type = static_cast<CommandType>(cmd_type);
  }

  template <Buffer BufferType>
  static void deserialize(const BufferType &buffer, size_t &offset, Tensor &tensor) {
    size_t dtype_size;
    DType_t dtype;
    uint32_t dtype_value;
    buffer.template read<uint32_t>(offset, dtype_value);
    dtype = static_cast<DType_t>(dtype_value);
    dtype_size = get_dtype_size(dtype);
    uint64_t shape_size;
    buffer.read(offset, shape_size);
    std::vector<uint64_t> shape(shape_size);
    for (uint64_t i = 0; i < shape_size; ++i) {
      buffer.template read<uint64_t>(offset, shape[i]);
    }
    auto &device = deserialize_to_gpu_ ? getGPU() : getCPU();
    tensor = Tensor::create_pooled(PoolAllocator::instance(device), dtype,
                                   std::vector<size_t>(shape.begin(), shape.end()));
    if (tensor->size() > 0) {
      auto &dptr = tensor->data_ptr();
      size_t byte_size = tensor->size() * dtype_size;
      buffer.read(offset, dptr, byte_size);
    }
  }

  template <Buffer BufferType>
  static void deserialize(const BufferType &buffer, size_t &offset, Job &job) {
    buffer.template read<uint64_t>(offset, reinterpret_cast<uint64_t &>(job.mb_id));
    deserialize(buffer, offset, job.data);
  }

  template <Buffer BufferType>
  static void deserialize(const BufferType &buffer, size_t &offset, std::string &str) {
    buffer.read(offset, str);
  }

  template <Buffer BufferType>
  static void deserialize(const BufferType &buffer, size_t &offset, bool &flag) {
    uint8_t value;
    buffer.read(offset, value);
    flag = (value != 0);
  }

  template <Buffer BufferType>
  static void deserialize(const BufferType &buffer, size_t &offset, Event &event) {
    int64_t start_time_count;
    int64_t end_time_count;
    uint8_t type_value;
    buffer.read(offset, start_time_count);
    buffer.read(offset, end_time_count);
    buffer.read(offset, type_value);
    event.start_time = Clock::time_point(Clock::duration(start_time_count));
    event.end_time = Clock::time_point(Clock::duration(end_time_count));
    event.type = static_cast<EventType>(type_value);
    buffer.read(offset, event.name);
    buffer.read(offset, event.source);
  }

  template <Buffer BufferType>
  static void deserialize(const BufferType &buffer, size_t &offset, Profiler &profiler) {
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

  template <Buffer BufferType>
  static void deserialize(const BufferType &buffer, size_t &offset, MessageData &data) {
    // Determine payload type based on payload_type
    uint64_t payload_type;
    buffer.read(offset, payload_type);
    data.payload_type = payload_type;
    switch (data.payload_type) {
    case variant_index<PayloadType, std::monostate>(): // std::monostate
      data.payload = std::monostate{};
      break;
    case variant_index<PayloadType, Job>(): { // Job
      Job job;
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
    case variant_index<PayloadType, StageConfig>(): { // StageConfig
      uint64_t json_size;
      buffer.read(offset, json_size);
      std::string json_str;
      json_str.resize(json_size);
      buffer.read(offset, json_str.data(), json_size);
      StageConfig config = StageConfig::from_json(nlohmann::json::parse(json_str));
      data.payload = std::move(config);
    } break;
    default:
      throw std::runtime_error("Unsupported payload type in MessageData deserialization");
    }
  }

  template <Buffer BufferType>
  static void deserialize(const BufferType &buffer, size_t &offset, Message &message) {
    deserialize(buffer, offset, message.header());
    deserialize(buffer, offset, message.data());
  }

}; // namespace BinarySerializer

} // namespace tnn