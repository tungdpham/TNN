/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "command_type.hpp"
#include "endian.hpp"
#include "load_tracker.hpp"
#include "task.hpp"
#include "tbuffer.hpp"
#include "tensor/tensor.hpp"
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace tpipeline {
using PayloadType = std::variant<std::monostate, Task<float>, std::string, bool, LoadTracker>;

struct FixedHeader {
  uint8_t PROTOCOL_VERSION = 1;
  Endianness endianess; // 1 for little-endian, 0 for big-endian
  uint64_t length = 0;  // Length of the rest of the message (excluding fixed header part)

  FixedHeader() : endianess(get_system_endianness()) {}

  FixedHeader(uint64_t len) : length(len) { endianess = get_system_endianness(); }

  static constexpr uint64_t size() {
    return sizeof(uint8_t) +    // PROTOCOL_VERSION
           sizeof(Endianness) + // endianess
           sizeof(uint64_t);    // length
  }
};

struct MessageHeader {
  std::string recipient_id; // ID of the recipient stage
  std::string sender_id;    // ID of the sender stage
  CommandType command_type; // Type of command

  MessageHeader() : command_type(CommandType::_START) {}

  MessageHeader(std::string recipient, CommandType cmd_type)
      : recipient_id(std::move(recipient)), sender_id(""), command_type(cmd_type) {}

  MessageHeader(const MessageHeader &other)
      : recipient_id(other.recipient_id), sender_id(other.sender_id),
        command_type(other.command_type) {}

  const uint64_t size() const {
    return sizeof(uint64_t) + recipient_id.size() + // recipient_id length + data
           sizeof(uint64_t) + sender_id.size() +    // sender_id length + data
           sizeof(uint16_t);                        // command_type (serialized as uint16_t)
  }
};

struct MessageData {
  uint64_t payload_type; // to indicate which type is held in the variant
  PayloadType payload;

  MessageData(const PayloadType &pay) : payload(std::move(pay)) {
    payload_type = static_cast<uint64_t>(payload.index());
  }

  MessageData(const MessageData &other)
      : payload_type(other.payload_type), payload(other.payload) {}

  const uint64_t size() const {
    // Rough estimate of size; actual serialization may differ
    uint64_t size = 0;

    // size of payload_type
    size += sizeof(payload_type);

    // size of payload
    if (std::holds_alternative<std::monostate>(payload)) {
      // No additional size for monostate

    } else if (std::holds_alternative<Task<float>>(payload)) {
      const auto &task = std::get<Task<float>>(payload);
      size += sizeof(uint64_t); // micro_batch_id
      size += sizeof(uint64_t); // shape size (uint64_t in serialization)
      size +=
          task.data.shape().size() * sizeof(uint64_t); // each dimension (uint64_t in serialization)
      // No size prefix for tensor data itself
      size += task.data.size() * sizeof(float); // data

    } else if (std::holds_alternative<std::string>(payload)) {
      const auto &str = std::get<std::string>(payload);
      size += sizeof(uint64_t); // string length (uint64_t in serialization)
      size += str.size();       // string data

    } else if (std::holds_alternative<bool>(payload)) {
      size += sizeof(uint8_t); // bool serialized as uint8_t
    } else {
      throw new std::runtime_error("Unknown payload type in MessageData");
    }
    return size;
  }
};

struct Message {
  MessageHeader header;
  MessageData data;

  Message() : header("", CommandType::_START), data(std::monostate{}) {}

  Message(std::string recipient_id, CommandType cmd_type, PayloadType payload)
      : header(std::move(recipient_id), cmd_type), data(std::move(payload)) {}

  Message(std::string recipient_id, CommandType cmd_type)
      : header(std::move(recipient_id), cmd_type), data(std::monostate{}) {}

  Message(const Message &other) : header(other.header), data(other.data) {}

  ~Message() = default;

  template <typename PayloadType> bool has_type() const {
    return std::holds_alternative<PayloadType>(data.payload);
  }

  template <typename PayloadType> const PayloadType &get() const {
    return std::get<PayloadType>(data.payload);
  }

  template <typename PayloadType> void set(const PayloadType &new_payload) {
    data.payload = new_payload;
    data.payload_type = static_cast<uint64_t>(data.payload.index());
  }

  const uint64_t size() const { return header.size() + data.size(); }
};

} // namespace tpipeline