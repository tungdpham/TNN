/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "device/dptr.hpp"
#include "device/iallocator.hpp"
#include "endian.hpp"
#include "message.hpp"
#include "packet.hpp"
#include "profiling/profiler.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"
#include "vbuffer.hpp"

namespace tnn {

template <typename VariantType, typename T, uint64_t index = 0>
constexpr uint64_t variant_index() {
  static_assert(std::variant_size_v<VariantType> > index, "Type not found in variant");
  if constexpr (index == std::variant_size_v<VariantType>) {
    return index;
  } else if constexpr (std::is_same_v<std::variant_alternative_t<index, VariantType>, T>) {
    return index;
  } else {
    return variant_index<VariantType, T, index + 1>();
  }
}

template <typename T>
concept VBufferType = requires(T t, VBuffer &vb, VBuffer &cvb) {
  { t } -> std::convertible_to<VBuffer &>;
};

class VSerializer {
private:
  IAllocator &allocator_;

public:
  VSerializer(IAllocator &allocator)
      : allocator_(allocator) {}

  ~VSerializer() = default;

  void serialize(VBuffer &buffer, size_t &offset, Tensor &&tensor) {
    DType_t dtype = tensor->data_type();
    buffer.write(offset, static_cast<uint32_t>(dtype));
    std::vector<size_t> shape = tensor->shape();
    uint64_t shape_size = static_cast<uint64_t>(shape.size());
    buffer.write(offset, shape_size);
    for (size_t dim : shape) {
      buffer.write(offset, static_cast<uint64_t>(dim));
    }
    size_t byte_size = tensor->size() * get_dtype_size(dtype);
    dptr data_ptr = tensor->data_ptr();
    buffer.append(data_ptr.span(0, byte_size));
  }

  void serialize(VBuffer &buffer, size_t &offset, Event &&event) {
    buffer.write(offset, static_cast<int64_t>(event.start_time.time_since_epoch().count()));
    buffer.write(offset, static_cast<int64_t>(event.end_time.time_since_epoch().count()));
    buffer.write(offset, static_cast<uint8_t>(event.type));
    buffer.write(offset, event.name);
    buffer.write(offset, event.source);
  }

  void serialize(VBuffer &buffer, size_t &offset, Profiler &&profiler) {
    auto events = profiler.get_events();
    buffer.write(offset, static_cast<int64_t>(profiler.start_time().time_since_epoch().count()));
    int64_t event_count = static_cast<int64_t>(events.size());
    buffer.write(offset, event_count);
    for (auto &event : events) {
      serialize(buffer, offset, std::move(event));
    }
  }

  void serialize(VBuffer &buffer, size_t &offset, const PacketHeader &header) {
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

  void serialize(VBuffer &buffer, size_t &offset, const MessageHeader &header) {
    buffer.write(offset, static_cast<uint16_t>(header.command_type));
  }

  void serialize(VBuffer &buffer, size_t &offset, MessageData &&data) {
    buffer.write(offset, data.payload.index());  // Write payload type index
    if (std::holds_alternative<std::monostate>(data.payload)) {
    } else if (std::holds_alternative<Job>(data.payload)) {
      auto &job = std::get<Job>(data.payload);
      buffer.write(offset, static_cast<uint64_t>(job.mb_id));
      serialize(buffer, offset, std::move(job.data));
    } else if (std::holds_alternative<std::string>(data.payload)) {
      auto &str = std::get<std::string>(data.payload);
      buffer.write(offset, std::move(str));
    } else if (std::holds_alternative<bool>(data.payload)) {
      const auto &flag = std::get<bool>(data.payload);
      buffer.write(offset, static_cast<uint8_t>(flag ? 1 : 0));
    } else if (std::holds_alternative<Profiler>(data.payload)) {
      auto &profiler = std::get<Profiler>(data.payload);
      serialize(buffer, offset, std::move(profiler));
    } else if (std::holds_alternative<StageConfig>(data.payload)) {
      auto &stage_config = std::get<StageConfig>(data.payload);
      std::string json_dump = stage_config.to_json().dump();
      buffer.write(offset, std::move(json_dump));
    } else {
      throw std::runtime_error("Unsupported payload type in MessageData");
    }
  }

  void serialize(VBuffer &buffer, size_t &offset, Message &&message) {
    serialize(buffer, offset, message.header());
    serialize(buffer, offset, std::move(message.data()));
  }

  void deserialize(VBuffer &buffer, size_t &offset, PacketHeader &header) {
    buffer.read(offset, header.PROTOCOL_VERSION);
    buffer.read(offset, header.endianess);
    buffer.read(offset, header.type);
    buffer.read(offset, header.length);
    buffer.read(offset, header.msg_length);
    buffer.read(offset, header.msg_serial_id);
    buffer.read(offset, header.packet_offset);
    buffer.read(offset, header.total_packets);
    buffer.read(offset, reinterpret_cast<uint8_t &>(header.compression_type));

    if (header.endianess != get_system_endianness()) {
      bswap(header.type);
      bswap(header.length);
      bswap(header.msg_length);
      bswap(header.msg_serial_id);
      bswap(header.packet_offset);
      bswap(header.total_packets);
    }
  }

  void deserialize(VBuffer &buffer, size_t &offset, MessageHeader &header) {
    uint16_t cmd_type;
    buffer.read(offset, cmd_type);
    header.command_type = static_cast<CommandType>(cmd_type);
  }

  void deserialize(VBuffer &buffer, size_t &offset, Tensor &tensor) {
    size_t dtype_size;
    DType_t dtype;
    uint32_t dtype_value;
    buffer.read(offset, dtype_value);
    dtype = static_cast<DType_t>(dtype_value);
    dtype_size = get_dtype_size(dtype);
    uint64_t shape_size;
    buffer.read(offset, shape_size);
    std::vector<uint64_t> shape(shape_size);
    for (uint64_t i = 0; i < shape_size; ++i) {
      buffer.read(offset, shape[i]);
    }
    size_t byte_size =
        dtype_size * std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    dptr tensor_data = buffer.get(offset).span(0, byte_size);
    tensor = make_tensor(allocator_, dtype, std::move(tensor_data), shape);
  }

  void deserialize(VBuffer &buffer, size_t &offset, Job &job) {
    buffer.read(offset, reinterpret_cast<uint64_t &>(job.mb_id));
    deserialize(buffer, offset, job.data);
  }

  void deserialize(VBuffer &buffer, size_t &offset, std::string &str) { buffer.read(offset, str); }

  void deserialize(VBuffer &buffer, size_t &offset, bool &flag) {
    uint8_t value;
    buffer.read(offset, value);
    flag = (value != 0);
  }

  void deserialize(VBuffer &buffer, size_t &offset, Event &event) {
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

  void deserialize(VBuffer &buffer, size_t &offset, Profiler &profiler) {
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

  void deserialize(VBuffer &buffer, size_t &offset, MessageData &data) {
    uint64_t payload_type;
    buffer.read(offset, payload_type);
    switch (payload_type) {
      case variant_index<PayloadType, std::monostate>():
        data.payload = std::monostate{};
        break;
      case variant_index<PayloadType, Job>(): {
        Job job;
        deserialize(buffer, offset, job);
        data.payload = std::move(job);
      } break;
      case variant_index<PayloadType, std::string>(): {
        std::string str;
        deserialize(buffer, offset, str);
        data.payload = std::move(str);
      } break;
      case variant_index<PayloadType, bool>(): {
        bool flag;
        deserialize(buffer, offset, flag);
        data.payload = flag;
      } break;
      case variant_index<PayloadType, Profiler>(): {
        Profiler profiler;
        deserialize(buffer, offset, profiler);
        data.payload = std::move(profiler);
      } break;
      case variant_index<PayloadType, StageConfig>(): {
        uint64_t json_size;
        buffer.read(offset, json_size);
        std::string json_str;
        json_str.resize(json_size);
        buffer.read(offset, reinterpret_cast<uint8_t *>(json_str.data()), json_size);
        StageConfig config = StageConfig::from_json(nlohmann::json::parse(json_str));
        data.payload = std::move(config);
      } break;
      default:
        throw std::runtime_error("Unsupported payload type in MessageData deserialization");
    }
  }

  void deserialize(VBuffer &buffer, size_t &offset, Message &message) {
    deserialize(buffer, offset, message.header());
    deserialize(buffer, offset, message.data());
  }
};

}  // namespace tnn
