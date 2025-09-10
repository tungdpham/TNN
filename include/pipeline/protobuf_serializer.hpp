#pragma once

#include "message.hpp"
#include "pipeline.pb.h"
#include "stage_config.hpp"
#include "task.hpp"
#include "tensor/tensor.hpp"
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tpipeline {

class GoogleProtobufSerializer {
public:
  template <typename T>
  static std::vector<uint8_t> serialize_tensor(const Tensor<T> &tensor) {
    tpipeline::proto::Tensor proto_tensor;

    auto shape = tensor.shape();
    for (size_t dim : shape) {
      proto_tensor.add_shape(static_cast<uint32_t>(dim));
    }

    proto_tensor.set_dtype(get_dtype_string<T>());

    const T *data = tensor.data();
    size_t data_size = tensor.size() * sizeof(T);
    proto_tensor.set_data(data, data_size);

    std::string serialized;
    if (!proto_tensor.SerializeToString(&serialized)) {
      throw std::runtime_error("Failed to serialize tensor to protobuf");
    }

    return std::vector<uint8_t>(serialized.begin(), serialized.end());
  }

  template <typename T>
  static Tensor<T> deserialize_tensor(const std::vector<uint8_t> &buffer) {
    tpipeline::proto::Tensor proto_tensor;
    std::string serialized(buffer.begin(), buffer.end());

    if (!proto_tensor.ParseFromString(serialized)) {
      throw std::runtime_error("Failed to parse tensor from protobuf");
    }

    if (proto_tensor.dtype() != get_dtype_string<T>()) {
      throw std::runtime_error("Tensor data type mismatch: expected " +
                               get_dtype_string<T>() + ", got " +
                               proto_tensor.dtype());
    }

    std::vector<size_t> shape;
    for (int i = 0; i < proto_tensor.shape_size(); ++i) {
      shape.push_back(proto_tensor.shape(i));
    }

    Tensor<T> tensor(shape);
    const std::string &data_bytes = proto_tensor.data();

    if (data_bytes.size() != tensor.size() * sizeof(T)) {
      throw std::runtime_error("Tensor data size mismatch");
    }

    std::memcpy(tensor.data(), data_bytes.data(), data_bytes.size());
    return tensor;
  }

  template <typename T>
  static std::vector<uint8_t> serialize_task(const Task<T> &task) {
    tpipeline::proto::Task proto_task;

    proto_task.set_type(static_cast<tpipeline::proto::TaskType>(task.type));
    proto_task.set_micro_batch_id(static_cast<uint32_t>(task.micro_batch_id));

    auto tensor_bytes = serialize_tensor(task.data);
    tpipeline::proto::Tensor proto_tensor;
    std::string tensor_str(tensor_bytes.begin(), tensor_bytes.end());
    if (!proto_tensor.ParseFromString(tensor_str)) {
      throw std::runtime_error("Failed to parse tensor for task serialization");
    }
    *proto_task.mutable_data() = proto_tensor;

    std::string serialized;
    if (!proto_task.SerializeToString(&serialized)) {
      throw std::runtime_error("Failed to serialize task to protobuf");
    }

    return std::vector<uint8_t>(serialized.begin(), serialized.end());
  }

  template <typename T>
  static Task<T> deserialize_task(const std::vector<uint8_t> &buffer) {
    tpipeline::proto::Task proto_task;
    std::string serialized(buffer.begin(), buffer.end());

    if (!proto_task.ParseFromString(serialized)) {
      throw std::runtime_error("Failed to parse task from protobuf");
    }

    TaskType type = static_cast<TaskType>(proto_task.type());

    std::string tensor_str = proto_task.data().SerializeAsString();
    std::vector<uint8_t> tensor_bytes(tensor_str.begin(), tensor_str.end());
    Tensor<T> tensor = deserialize_tensor<T>(tensor_bytes);

    return Task<T>(type, tensor, proto_task.micro_batch_id());
  }

  template <typename T>
  static std::vector<uint8_t> serialize_message(const Message<T> &message) {
    tpipeline::proto::Message proto_message;

    proto_message.set_command_type(
        static_cast<tpipeline::proto::CommandType>(message.command_type));
    proto_message.set_sequence_number(
        static_cast<uint32_t>(message.sequence_number));
    proto_message.set_sender_id(message.sender_id);
    proto_message.set_recipient_id(message.recipient_id);

    auto duration = message.timestamp.time_since_epoch();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    proto_message.set_timestamp(static_cast<uint64_t>(nanos.count()));

    if (message.has_task()) {
      auto task_bytes = serialize_task(message.get_task());
      tpipeline::proto::Task proto_task;
      std::string task_str(task_bytes.begin(), task_bytes.end());
      if (!proto_task.ParseFromString(task_str)) {
        throw std::runtime_error(
            "Failed to parse task for message serialization");
      }
      *proto_message.mutable_task() = proto_task;
    } else if (message.has_text()) {
      proto_message.set_text(message.get_text());
    } else if (message.has_signal()) {
      proto_message.set_signal(message.get_signal());
    }

    std::string serialized;
    if (!proto_message.SerializeToString(&serialized)) {
      throw std::runtime_error("Failed to serialize message to protobuf");
    }

    return std::vector<uint8_t>(serialized.begin(), serialized.end());
  }

  template <typename T>
  static Message<T> deserialize_message(const std::vector<uint8_t> &buffer) {
    tpipeline::proto::Message proto_message;
    std::string serialized(buffer.begin(), buffer.end());

    if (!proto_message.ParseFromString(serialized)) {
      throw std::runtime_error("Failed to parse message from protobuf");
    }

    CommandType command_type =
        static_cast<CommandType>(proto_message.command_type());
    Message<T> message(command_type);

    message.sequence_number = proto_message.sequence_number();
    message.sender_id = proto_message.sender_id();
    message.recipient_id = proto_message.recipient_id();

    auto nanos = std::chrono::nanoseconds(proto_message.timestamp());
    message.timestamp = std::chrono::steady_clock::time_point(nanos);

    switch (proto_message.payload_case()) {
    case tpipeline::proto::Message::kTask: {
      std::string task_str = proto_message.task().SerializeAsString();
      std::vector<uint8_t> task_bytes(task_str.begin(), task_str.end());
      Task<T> task = deserialize_task<T>(task_bytes);
      message.payload = task;
      break;
    }
    case tpipeline::proto::Message::kText:
      message.payload = proto_message.text();
      break;
    case tpipeline::proto::Message::kSignal:
      message.payload = proto_message.signal();
      break;
    default:

      break;
    }

    return message;
  }

  static std::vector<uint8_t>
  serialize_stage_config(const StageConfig &config) {
    tpipeline::proto::StageConfig proto_config;

    proto_config.set_stage_id(config.stage_id);
    proto_config.set_stage_index(config.stage_index);
    proto_config.set_model_config_json(config.model_config.dump());
    proto_config.set_next_stage_endpoint(config.next_stage_endpoint);
    proto_config.set_prev_stage_endpoint(config.prev_stage_endpoint);
    proto_config.set_coordinator_endpoint(config.coordinator_endpoint);

    std::string serialized;
    if (!proto_config.SerializeToString(&serialized)) {
      throw std::runtime_error("Failed to serialize stage config to protobuf");
    }

    return std::vector<uint8_t>(serialized.begin(), serialized.end());
  }

  static StageConfig
  deserialize_stage_config(const std::vector<uint8_t> &buffer) {
    tpipeline::proto::StageConfig proto_config;
    std::string serialized(buffer.begin(), buffer.end());

    if (!proto_config.ParseFromString(serialized)) {
      throw std::runtime_error("Failed to parse stage config from protobuf");
    }

    StageConfig config;
    config.stage_id = proto_config.stage_id();
    config.stage_index = proto_config.stage_index();
    config.model_config =
        nlohmann::json::parse(proto_config.model_config_json());
    config.next_stage_endpoint = proto_config.next_stage_endpoint();
    config.prev_stage_endpoint = proto_config.prev_stage_endpoint();
    config.coordinator_endpoint = proto_config.coordinator_endpoint();

    return config;
  }

private:
  template <typename T> static std::string get_dtype_string() {
    if constexpr (std::is_same_v<T, float>)
      return "float";
    else if constexpr (std::is_same_v<T, double>)
      return "double";
    else if constexpr (std::is_same_v<T, int32_t>)
      return "int32";
    else if constexpr (std::is_same_v<T, int64_t>)
      return "int64";
    else if constexpr (std::is_same_v<T, uint32_t>)
      return "uint32";
    else if constexpr (std::is_same_v<T, uint64_t>)
      return "uint64";
    else
      return "unknown";
  }
};
} // namespace tpipeline
