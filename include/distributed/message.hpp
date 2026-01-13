/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "command_type.hpp"
#include "distributed/job_pool.hpp"
#include "job.hpp"
#include "profiling/profiler.hpp"
#include "tensor/tensor.hpp"
#include <arpa/inet.h>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace tnn {
using PayloadType = std::variant<std::monostate, PooledJob<float>, std::string, bool, Profiler>;

struct MessageHeader {
  std::string recipient_id; // ID of the recipient stage
  std::string sender_id;    // ID of the sender stage
  CommandType command_type; // Type of command

  MessageHeader() : command_type(CommandType::_START) {}

  MessageHeader(std::string sender_id, std::string recipient, CommandType cmd_type)
      : recipient_id(std::move(recipient)), sender_id(std::move(sender_id)),
        command_type(cmd_type) {}

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

  MessageData(PayloadType &&pay) : payload(std::move(pay)) {
    payload_type = static_cast<uint64_t>(payload.index());
  }

  MessageData(const MessageData &other) = delete;

  MessageData(MessageData &&other) noexcept
      : payload_type(other.payload_type), payload(std::move(other.payload)) {}

  ~MessageData() = default;

  MessageData &operator=(MessageData &&other) noexcept {
    if (this != &other) {
      payload_type = other.payload_type;
      payload = std::move(other.payload);
    }
    return *this;
  }

  const uint64_t size() const {
    // Rough estimate of size; actual serialization may differ
    uint64_t size = 0;

    // size of payload_type
    size += sizeof(payload_type);

    // size of payload
    if (std::holds_alternative<std::monostate>(payload)) {
      // No additional size for monostate

    } else if (std::holds_alternative<PooledJob<float>>(payload)) {
      const auto &job = std::get<PooledJob<float>>(payload);
      size += sizeof(uint64_t); // micro_batch_id
      size += sizeof(uint64_t); // shape size (uint64_t in serialization)
      size +=
          job->data.shape().size() * sizeof(uint64_t); // each dimension (uint64_t in serialization)
      // No size prefix for tensor data itself
      size += job->data.size() * sizeof(float); // data

    } else if (std::holds_alternative<std::string>(payload)) {
      const auto &str = std::get<std::string>(payload);
      size += sizeof(uint64_t); // string length (uint64_t in serialization)
      size += str.size();       // string data

    } else if (std::holds_alternative<bool>(payload)) {
      size += sizeof(uint8_t); // bool serialized as uint8_t

    } else if (std::holds_alternative<Profiler>(payload)) {
      const auto &profiler = std::get<Profiler>(payload);
      size += sizeof(int64_t); // profiler_start_time_ (serialized as int64_t)
      auto events = profiler.get_events();
      size += sizeof(uint64_t); // number of events
      for (const auto &event : events) {
        size += sizeof(int64_t);   // start_time_ (serialized as int64_t)
        size += sizeof(int64_t);   // end_time_ (serialized as int64_t)
        size += sizeof(uint64_t);  // name length (uint64_t in serialization)
        size += event.name.size(); // name data
      }
    } else {
      throw new std::runtime_error("Unknown payload type in MessageData");
    }
    return size;
  }
};

struct Message {
private:
  MessageHeader header_;
  MessageData data_;

public:
  Message() : header_("", "", CommandType::_START), data_(std::monostate{}) {}

  Message(std::string sender_id, std::string recipient_id, CommandType cmd_type,
          PayloadType &&payload)
      : header_(std::move(sender_id), std::move(recipient_id), cmd_type),
        data_(std::move(payload)) {}

  Message(std::string sender_id, std::string recipient_id, CommandType cmd_type)
      : header_(std::move(sender_id), std::move(recipient_id), cmd_type), data_(std::monostate{}) {}

  Message(MessageHeader &&header, MessageData &&data)
      : header_(std::move(header)), data_(std::move(data)) {}

  Message(const Message &other) = delete;

  Message(Message &&other) noexcept
      : header_(std::move(other.header_)), data_(std::move(other.data_)) {}

  ~Message() = default;

  Message &operator=(const Message &other) = delete;

  Message &operator=(Message &&other) noexcept {
    if (this != &other) {
      header_ = std::move(other.header_);
      data_ = std::move(other.data_);
    }
    return *this;
  }

  MessageHeader &header() { return header_; }
  const MessageHeader &header() const { return header_; }

  MessageData &data() { return data_; }
  const MessageData &data() const { return data_; }

  template <typename PayloadType> bool has_type() const {
    return std::holds_alternative<PayloadType>(data_.payload);
  }

  template <typename PayloadType> PayloadType &get() {
    return std::get<PayloadType>(data_.payload);
  }

  template <typename PayloadType> const PayloadType &get() const {
    return std::get<PayloadType>(data_.payload);
  }

  template <typename PayloadType> void set(const PayloadType &new_payload) {
    data_.payload = new_payload;
    data_.payload_type = static_cast<uint64_t>(data_.payload.index());
  }

  const uint64_t size() const { return header_.size() + data_.size(); }
};

} // namespace tnn