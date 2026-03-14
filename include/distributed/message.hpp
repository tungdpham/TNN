/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include <arpa/inet.h>

#include <cstring>
#include <string>

#include "command_type.hpp"
#include "common/variant.hpp"
#include "distributed/stage_config.hpp"
#include "job.hpp"
#include "profiling/profiler.hpp"

namespace tnn {
using PayloadType = Variant<std::monostate, Job, std::string, bool, Profiler, StageConfig>;

struct MessageHeader {
  CommandType command_type;  // Type of command

  MessageHeader()
      : command_type(CommandType::_START) {}

  MessageHeader(CommandType cmd_type)
      : command_type(cmd_type) {}

  MessageHeader(const MessageHeader &other)
      : command_type(other.command_type) {}
};

template <typename Archiver>
void archive(Archiver &archiver, const MessageHeader &header) {
  archiver(header.command_type);
}

template <typename Archiver>
void archive(Archiver &archiver, MessageHeader &header) {
  archiver(header.command_type);
}

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
};

template <typename Archiver>
void archive(Archiver &archiver, const MessageData &message_data) {
  archiver(message_data.payload);
}

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
    return data_.payload.holds<PayloadType>();
  }

  template <typename PayloadType>
  PayloadType &get() {
    return data_.payload.get<PayloadType>();
  }

  template <typename PayloadType>
  const PayloadType &get() const {
    return data_.payload.get<PayloadType>();
  }

  template <typename PayloadType>
  void set(const PayloadType &new_payload) {
    data_.payload = new_payload;
  }
};

template <typename Archiver>
void archive(Archiver &archiver, const Message &message) {
  archiver(message.header(), message.data());
}

}  // namespace tnn