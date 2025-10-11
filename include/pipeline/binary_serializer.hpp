#pragma once

#include "message.hpp"
#include "tbuffer.hpp"

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
namespace tpipeline {
class BinarySerializer {
public:
  template <typename T = float> static void serialize(const Tensor<T> &tensor, TBuffer &buffer) {
    std::vector<size_t> shape = tensor.shape();
    uint64_t shape_size = static_cast<uint64_t>(shape.size());
    buffer.write_value(shape_size);
    for (size_t dim : shape) {
      buffer.write_value(static_cast<uint64_t>(dim));
    }
    buffer.write_array(tensor.data(), tensor.size()); // write_array expects count, not bytes
  }

  static void serialize(const FixedHeader &header, TBuffer &buffer) {
    buffer.write_value(header.PROTOCOL_VERSION);
    buffer.write_value(header.endianess);
    buffer.write_value(header.length);
  }

  static void serialize(const MessageHeader &header, TBuffer &buffer) {
    buffer.write_value(header.recipient_id);
    buffer.write_value(header.sender_id);
    buffer.write_value(header.command_type);
  }

  static void serialize(const MessageData &data, TBuffer &buffer) {
    buffer.write_value(data.payload_type);
    if (std::holds_alternative<std::monostate>(data.payload)) {
      // No additional data to write

    } else if (std::holds_alternative<Task<float>>(data.payload)) {
      const auto &task = std::get<Task<float>>(data.payload);
      buffer.write_value(static_cast<uint64_t>(task.micro_batch_id));
      serialize<float>(task.data, buffer);

    } else if (std::holds_alternative<std::string>(data.payload)) {
      const auto &str = std::get<std::string>(data.payload);
      uint64_t str_length = static_cast<uint64_t>(str.size());
      buffer.write_value(str_length);
      const char *chars = str.data();
      buffer.write_array(reinterpret_cast<const uint8_t *>(chars), str_length);

    } else if (std::holds_alternative<bool>(data.payload)) {
      const auto &flag = std::get<bool>(data.payload);
      buffer.write_value(static_cast<uint8_t>(flag ? 1 : 0));

    } else {
      throw std::runtime_error("Unsupported payload type in MessageData");
    }
  }

  static void serialize(const Message &message, TBuffer &buffer) {
    serialize(message.header, buffer);
    serialize(message.data, buffer);
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, FixedHeader &header) {
    header.PROTOCOL_VERSION = buffer.read_value<uint8_t>(offset);
    header.endianess = buffer.read_value<uint8_t>(offset);
    header.length = buffer.read_value<uint64_t>(offset);
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, MessageHeader &header) {
    header.recipient_id = buffer.read_string(offset);
    header.sender_id = buffer.read_string(offset);
    header.command_type = static_cast<CommandType>(buffer.read_value<uint16_t>(offset));
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, Tensor<float> &tensor) {
    uint64_t shape_size = buffer.read_value<uint64_t>(offset);
    std::vector<size_t> shape(shape_size);
    for (uint64_t i = 0; i < shape_size; ++i) {
      shape[i] = static_cast<size_t>(buffer.read_value<uint64_t>(offset));
    }
    tensor = Tensor<float>(shape);
    if (tensor.size() > 0) {
      std::memcpy(tensor.data(), buffer.get() + offset, tensor.size() * sizeof(float));
      offset += tensor.size() * sizeof(float);
    }
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, Task<float> &task) {
    task.micro_batch_id = static_cast<size_t>(buffer.read_value<uint64_t>(offset));
    Tensor<float> tensor;
    deserialize(buffer, offset, tensor);
    task.data = std::move(tensor);
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, std::string &str) {
    uint64_t str_length = buffer.read_value<uint64_t>(offset);
    if (str_length > 0) {
      str.resize(str_length);
      buffer.read_array(offset, reinterpret_cast<uint8_t *>(str.data()), str_length);
    } else {
      str.clear();
    }
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, bool &flag) {
    uint8_t value = buffer.read_value<uint8_t>(offset);
    flag = (value != 0);
  }

  static void deserialize(const TBuffer &buffer, size_t &offset,
                          std::vector<Tensor<float>> &tensors) {
    uint64_t tensor_count = buffer.read_value<uint64_t>(offset);
    tensors.clear();
    tensors.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
      Tensor<float> tensor;
      deserialize(buffer, offset, tensor);
      tensors.push_back(std::move(tensor));
    }
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, MessageData &data) {
    // Determine payload type based on payload_type
    uint64_t payload_type = buffer.read_value<uint64_t>(offset);
    data.payload_type = payload_type;
    switch (payload_type) {
    case variant_index<PayloadType, std::monostate>(): // std::monostate
      data.payload = std::monostate{};
      break;
    case variant_index<PayloadType, Task<float>>(): { // Task<float>
      Task<float> task;
      deserialize(buffer, offset, task);
      data.payload = std::move(task);
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
    default:
      throw std::runtime_error("Unsupported payload type in MessageData deserialization");
    }
  }

  static void deserialize(const TBuffer &buffer, size_t &offset, Message &message) {
    deserialize(buffer, offset, message.header);
    deserialize(buffer, offset, message.data);
  }

}; // namespace BinarySerializer

} // namespace tpipeline