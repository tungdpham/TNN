/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <arpa/inet.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <variant>

#include "command_type.hpp"
#include "distributed/stage_config.hpp"
#include "job.hpp"
#include "profiling/profiler.hpp"
#include "type/type.hpp"

namespace tnn {
using PayloadType = std::variant<std::monostate, Job, std::string, bool, Profiler, StageConfig>;

struct SizeVisitor {
  uint64_t operator()(const std::monostate &) const { return 0; }
  uint64_t operator()(const Job &job) const {
    uint64_t size = 0;
    size += sizeof(uint64_t);                             // mb_id
    size += sizeof(uint32_t);                             // dtype (serialized as uint32_t)
    size += sizeof(uint64_t);                             // shape size (uint64_t in serialization)
    size += job.data->shape().size() * sizeof(uint64_t);  // each dimension (uint64_t)
    size += job.data->size() * get_dtype_size(job.data->data_type());  // tensor data
    return size;
  }
  uint64_t operator()(const std::string &str) const {
    return sizeof(uint64_t) + str.size();  // string length + data
  }
  uint64_t operator()(const bool &) const {
    return sizeof(uint8_t);  // bool serialized as uint8_t
  }
  uint64_t operator()(const Profiler &profiler) const {
    uint64_t size = 0;
    size += sizeof(int64_t);  // profiler_start_time_
    auto events = profiler.get_events();
    size += sizeof(int64_t);  // number of events
    for (const auto &event : events) {
      size += sizeof(int64_t);      // start_time_
      size += sizeof(int64_t);      // end_time_
      size += sizeof(uint8_t);      // event.type
      size += sizeof(uint64_t);     // name length
      size += event.name.size();    // name data
      size += sizeof(uint64_t);     // source length
      size += event.source.size();  // source data
    }
    return size;
  }
  uint64_t operator()(const StageConfig &stage_config) const {
    auto json_str = stage_config.to_json().dump();
    return sizeof(uint64_t) + json_str.size();  // JSON string size
  }
};

struct MessageHeader {
  CommandType command_type;  // Type of command

  MessageHeader()
      : command_type(CommandType::_START) {}

  MessageHeader(CommandType cmd_type)
      : command_type(cmd_type) {}

  MessageHeader(const MessageHeader &other)
      : command_type(other.command_type) {}

  const uint64_t size() const {
    return sizeof(uint16_t);  // command_type (serialized as uint16_t)
  }
};

struct MessageData {
  PayloadType payload;

  MessageData(PayloadType &&pay = std::monostate{})
      : payload(std::move(pay)) {}

  MessageData(const MessageData &other) = delete;

  MessageData(MessageData &&other) noexcept
      : payload(std::move(other.payload)) {}

  ~MessageData() = default;

  MessageData &operator=(const MessageData &other) = delete;

  MessageData &operator=(MessageData &&other) noexcept {
    if (this != &other) {
      payload = std::move(other.payload);
    }
    return *this;
  }

  const uint64_t size() const {
    return sizeof(uint64_t) +                   // payload type index
           std::visit(SizeVisitor{}, payload);  // payload size
  }
};

struct Message {
private:
  MessageHeader header_;
  MessageData data_;

public:
  Message(CommandType cmd_type = CommandType::_START, std::monostate payload = std::monostate{})
      : header_(cmd_type),
        data_(std::move(payload)) {}

  Message(CommandType cmd_type, PayloadType &&payload)
      : header_(cmd_type),
        data_(std::move(payload)) {}

  Message(const MessageHeader &header, MessageData &&data)
      : header_(header),
        data_(std::move(data)) {}

  Message(const Message &other) = delete;

  Message(Message &&other) noexcept
      : header_(std::move(other.header_)),
        data_(std::move(other.data_)) {}

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

  template <typename PayloadType>
  bool has_type() const {
    return std::holds_alternative<PayloadType>(data_.payload);
  }

  template <typename PayloadType>
  PayloadType &get() {
    return std::get<PayloadType>(data_.payload);
  }

  template <typename PayloadType>
  const PayloadType &get() const {
    return std::get<PayloadType>(data_.payload);
  }

  template <typename PayloadType>
  void set(const PayloadType &new_payload) {
    data_.payload = new_payload;
  }

  const uint64_t size() const { return header_.size() + data_.size(); }
};

}  // namespace tnn