#pragma once

#include "message.hpp"
#include "stage_config.hpp"
#include "task.hpp"
#include "tensor/tensor.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tpipeline {
class BinarySerializer {
public:
  template <typename T>
  static std::vector<uint8_t> serialize_tensor(const Tensor<T> &tensor) {
    std::vector<uint8_t> buffer;

    auto shape = tensor.shape();
    uint32_t shape_size = static_cast<uint32_t>(shape.size());
    write_value(buffer, shape_size);

    for (size_t dim : shape) {
      uint32_t dim_val = static_cast<uint32_t>(dim);
      write_value(buffer, dim_val);
    }

    const T *data = tensor.data();
    size_t data_size = tensor.size();
    uint32_t data_count = static_cast<uint32_t>(data_size);
    write_value(buffer, data_count);

    size_t data_bytes = data_size * sizeof(T);
    size_t old_size = buffer.size();
    buffer.resize(old_size + data_bytes);
    std::memcpy(buffer.data() + old_size, data, data_bytes);

    return buffer;
  }

  template <typename T>
  static Tensor<T> deserialize_tensor(const std::vector<uint8_t> &buffer,
                                      size_t &offset) {

    uint32_t shape_size = read_value<uint32_t>(buffer, offset);
    std::vector<size_t> shape(shape_size);

    for (uint32_t i = 0; i < shape_size; ++i) {
      uint32_t dim = read_value<uint32_t>(buffer, offset);
      shape[i] = dim;
    }

    uint32_t data_count = read_value<uint32_t>(buffer, offset);

    Tensor<T> tensor(shape);
    size_t data_bytes = data_count * sizeof(T);

    if (offset + data_bytes > buffer.size()) {
      throw std::runtime_error("Invalid tensor data in buffer");
    }

    std::memcpy(tensor.data(), buffer.data() + offset, data_bytes);
    offset += data_bytes;

    return tensor;
  }

  template <typename T>
  static std::vector<uint8_t> serialize_task(const Task<T> &task) {
    std::vector<uint8_t> buffer;

    write_value(buffer, static_cast<uint32_t>(task.type));

    write_value(buffer, static_cast<uint32_t>(task.micro_batch_id));

    auto tensor_data = serialize_tensor(task.data);
    write_value(buffer, static_cast<uint32_t>(tensor_data.size()));
    buffer.insert(buffer.end(), tensor_data.begin(), tensor_data.end());

    return buffer;
  }

  template <typename T>
  static Task<T> deserialize_task(const std::vector<uint8_t> &buffer,
                                  size_t &offset) {

    uint32_t type_val = read_value<uint32_t>(buffer, offset);
    TaskType type = static_cast<TaskType>(type_val);

    uint32_t micro_batch_id = read_value<uint32_t>(buffer, offset);

    uint32_t tensor_size = read_value<uint32_t>(buffer, offset);
    if (offset + tensor_size > buffer.size()) {
      throw std::runtime_error("Invalid task data in buffer");
    }

    std::vector<uint8_t> tensor_buffer(buffer.begin() + offset,
                                       buffer.begin() + offset + tensor_size);
    offset += tensor_size;

    size_t tensor_offset = 0;
    Tensor<T> tensor = deserialize_tensor<T>(tensor_buffer, tensor_offset);

    return Task<T>(type, tensor, static_cast<int>(micro_batch_id));
  }

  template <typename T>
  static std::vector<uint8_t> serialize_message(const Message<T> &message) {
    std::vector<uint8_t> buffer;

    write_value(buffer, static_cast<uint32_t>(message.command_type));

    write_value(buffer, static_cast<uint32_t>(message.sequence_number));

    write_string(buffer, message.sender_id);

    write_string(buffer, message.recipient_id);

    uint8_t flags = 0;
    if (message.has_task())
      flags |= 0x01;
    if (message.has_text())
      flags |= 0x02;
    if (message.has_signal())
      flags |= 0x04;
    write_value(buffer, flags);

    if (message.has_task()) {
      auto task_data = serialize_task(message.get_task());
      write_value(buffer, static_cast<uint32_t>(task_data.size()));
      buffer.insert(buffer.end(), task_data.begin(), task_data.end());
    }

    if (message.has_text()) {
      write_string(buffer, message.get_text());
    }

    if (message.has_signal()) {
      write_value(buffer, static_cast<uint8_t>(message.get_signal() ? 1 : 0));
    }

    return buffer;
  }

  template <typename T>
  static Message<T> deserialize_message(const std::vector<uint8_t> &buffer) {
    size_t offset = 0;

    uint32_t cmd_type = read_value<uint32_t>(buffer, offset);
    CommandType command_type = static_cast<CommandType>(cmd_type);

    Message<T> message(command_type);

    message.sequence_number =
        static_cast<int>(read_value<uint32_t>(buffer, offset));

    message.sender_id = read_string(buffer, offset);

    message.recipient_id = read_string(buffer, offset);

    uint8_t flags = read_value<uint8_t>(buffer, offset);

    if (flags & 0x01) {
      uint32_t task_size = read_value<uint32_t>(buffer, offset);
      if (offset + task_size > buffer.size()) {
        throw std::runtime_error("Invalid task data in message buffer");
      }

      std::vector<uint8_t> task_buffer(buffer.begin() + offset,
                                       buffer.begin() + offset + task_size);
      offset += task_size;

      size_t task_offset = 0;
      Task<T> task = deserialize_task<T>(task_buffer, task_offset);
      message.payload = task;
    }

    if (flags & 0x02) {
      std::string text = read_string(buffer, offset);
      message.payload = text;
    }

    if (flags & 0x04) {
      uint8_t signal_val = read_value<uint8_t>(buffer, offset);
      message.payload = (signal_val != 0);
    }

    return message;
  }

private:
  template <typename T>
  static void write_value(std::vector<uint8_t> &buffer, const T &value) {
    size_t old_size = buffer.size();
    buffer.resize(old_size + sizeof(T));
    std::memcpy(buffer.data() + old_size, &value, sizeof(T));
  }

  template <typename T>
  static T read_value(const std::vector<uint8_t> &buffer, size_t &offset) {
    if (offset + sizeof(T) > buffer.size()) {
      throw std::runtime_error("Buffer underrun while reading value");
    }

    T value;
    std::memcpy(&value, buffer.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
  }

  static void write_string(std::vector<uint8_t> &buffer,
                           const std::string &str) {
    uint32_t length = static_cast<uint32_t>(str.length());
    write_value(buffer, length);

    size_t old_size = buffer.size();
    buffer.resize(old_size + str.length());
    std::memcpy(buffer.data() + old_size, str.data(), str.length());
  }

  static std::string read_string(const std::vector<uint8_t> &buffer,
                                 size_t &offset) {
    uint32_t length = read_value<uint32_t>(buffer, offset);

    if (offset + length > buffer.size()) {
      throw std::runtime_error("Buffer underrun while reading string");
    }

    std::string str(reinterpret_cast<const char *>(buffer.data() + offset),
                    length);
    offset += length;
    return str;
  }
};
} // namespace tpipeline