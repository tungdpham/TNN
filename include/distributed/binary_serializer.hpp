#pragma once

#include <sys/types.h>

#include <cstdint>
#include <variant>

#include "device/iallocator.hpp"
#include "distributed/io.hpp"
#include "distributed/stage_config.hpp"
#include "message.hpp"
#include "packet.hpp"
#include "profiling/profiler.hpp"
#include "tensor/tensor.hpp"
#include "type/type.hpp"

/**
 * Very Important Note: size_t is platform dependent. On 64-bit systems, it is usually
 * 8 bytes, while on 32-bit systems, it is 4 bytes. For serialization, we need a fixed size
 * type to ensure compatibility across different platforms. Here, we use uint64_t for sizes
 * and counts, which is 8 bytes on all platforms.
 */
namespace tnn {

class BinarySerializer {
private:
  IAllocator &allocator_;

public:
  BinarySerializer(IAllocator &allocator)
      : allocator_(allocator) {}

  void serialize(Writer &writer, const Job &job) { writer(job); }

  void serialize(Writer &writer, const std::monostate &) {
    // No data to serialize for std::monostate
  }

  void serialize(Writer &writer, const bool &flag) {
    uint8_t value = flag ? 1 : 0;
    writer(value);
  }

  void serialize(Writer &writer, const std::string &str) { writer(str); }

  void serialize(Writer &writer, const Profiler &profiler) { writer(profiler); }

  void serialize(Writer &writer, const StageConfig &config) { writer(config); }

  void serialize(Writer &writer, const PacketHeader &header) { writer(header); }

  void serialize(Writer &writer, const MessageHeader &header) { writer(header); }

  void serialize(Writer &writer, const MessageData &data) {
    writer(data.payload.index());
    data.payload.visit([&](const auto &value) { serialize(writer, value); });
  }

  void serialize(Writer &writer, const Message &message) {
    serialize(writer, message.header());
    serialize(writer, message.data());
  }

  void deserialize(Reader &reader, PacketHeader &header) {
    reader(header.PROTOCOL_VERSION, header.endianess);
    reader.set_endianess(header.endianess);
    reader(header.type, header.packet_length, header.msg_length, header.msg_serial_id,
           header.packet_offset, header.total_packets, header.compression_type);
  }

  void deserialize(Reader &reader, MessageHeader &header) { reader(header); }

  void deserialize(Reader &reader, Tensor &tensor) {
    size_t dtype_size;
    DType_t dtype;
    reader(dtype);
    dtype_size = get_dtype_size(dtype);
    uint64_t shape_size;
    reader(shape_size);
    Vec<uint64_t> shape(shape_size);
    for (uint64_t i = 0; i < shape_size; ++i) {
      reader(shape[i]);
    }
    tensor = make_tensor(allocator_, dtype, Vec<size_t>(shape.begin(), shape.end()));
    if (tensor->size() > 0) {
      auto dptr = tensor->data_ptr();
      reader(make_blob(dptr.get<unsigned char>(), tensor->size() * dtype_size, dptr.device()));
    }
  }

  void deserialize(Reader &reader, Job &job) {
    uint64_t mb_id;
    reader(mb_id);
    job.mb_id = static_cast<size_t>(mb_id);
    deserialize(reader, job.data);
  }

  void deserialize(Reader &reader, std::string &str) { reader(str); }

  void deserialize(Reader &reader, bool &flag) {
    uint8_t value;
    reader(value);
    flag = (value != 0);
  }

  void deserialize(Reader &reader, Profiler &profiler) { reader(profiler); }

  void deserialize(Reader &reader, StageConfig &config) { reader(config); }

  void deserialize(Reader &reader, MessageData &data) {
    // Determine payload type based on payload_type
    uint32_t payload_type;
    reader(payload_type);
    switch (payload_type) {
      case PayloadType::index_of<std::monostate>():  // std::monostate
        data.payload = std::monostate{};
        break;
      case PayloadType::index_of<Job>(): {  // Job
        Job job;
        deserialize(reader, job);
        data.payload = std::move(job);
      } break;
      case PayloadType::index_of<std::string>(): {  // std::string
        std::string str;
        deserialize(reader, str);
        data.payload = std::move(str);
      } break;
      case PayloadType::index_of<bool>(): {  // bool
        bool flag;
        deserialize(reader, flag);
        data.payload = flag;
      } break;
      case PayloadType::index_of<Profiler>(): {  // Profiler
        Profiler profiler;
        deserialize(reader, profiler);
        data.payload = std::move(profiler);
      } break;
      case PayloadType::index_of<StageConfig>(): {  // StageConfig
        StageConfig config;
        deserialize(reader, config);
        data.payload = std::move(config);
      } break;
      default:
        throw std::runtime_error("Unsupported payload type in MessageData deserialization");
    }
  }

  void deserialize(Reader &reader, Message &message) {
    deserialize(reader, message.header());
    deserialize(reader, message.data());
  }

};  // namespace BinarySerializer

}  // namespace tnn