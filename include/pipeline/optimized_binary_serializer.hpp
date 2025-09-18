/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "message.hpp"
#include "serialization_layout.hpp"

namespace tpipeline {
class BinarySerializer {
  template <typename T> static uint8_t *serialize(const Message<T> &message) {
    T *buffer;
    uint32_t offset = 0;

    if (message.has_signal()) {
    }
    return nullptr; // Placeholder
  }

  template <typename T> static Message<T> deserialize(const uint8_t *data, size_t size) {
    return Message<T>(); // Placeholder
  }

  static void write_value(uint8_t *buffer, uint32_t &offset, const uint32_t &value) {
    uint32_t value;
    if (get_system_endianness() == Endianness::LITTLE) {
      uint32_t value = htole32(value);
    } else {
      uint32_t value = htobe32(value);
    }
    std::memcpy(buffer + offset, &value, sizeof(uint32_t));
    offset += sizeof(uint32_t);
  }

  static uint32_t read_value(const uint8_t *buffer, uint32_t &offset, uint32_t &value) {
    uint32_t value;
    if (get_system_endianness() == Endianness::LITTLE) {
      std::memcpy(&value, buffer + offset, sizeof(uint32_t));
      value = le32toh(value);
    } else {
      std::memcpy(&value, buffer + offset, sizeof(uint32_t));
      value = be32toh(value);
    }
    return offset + sizeof(uint32_t);
  }

  static uint32_t write_value(uint8_t *buffer, uint32_t &offset, const float &value) {
    // due to lack of float standardization across platforms, we store float as-is
    std::memcpy(buffer + offset, &value, sizeof(float));
    return offset + sizeof(float);
  }

  static void read_value(const uint8_t *buffer, uint32_t &offset, float &value) {
    // due to lack of float standardization across platforms, we read float as-is
    std::memcpy(&value, buffer + offset, sizeof(float));
    offset += sizeof(float);
  }

  static void write_value(uint8_t *buffer, uint32_t &offset, const bool &value) {
    uint8_t val = value ? 1 : 0;
    std::memcpy(buffer + offset, &val, sizeof(uint8_t));
    offset += sizeof(uint8_t);
  }

  static void read_value(const uint8_t *buffer, uint32_t &offset, bool &value) {
    uint8_t val;
    std::memcpy(&val, buffer + offset, sizeof(uint8_t));
    value = (val != 0);
    offset += sizeof(uint8_t);
  }

  static void write_string(uint8_t *buffer, uint32_t &offset, const std::string &str) {
    uint32_t length = static_cast<uint32_t>(str.length());
    write_value(buffer, offset, length);
    std::memcpy(buffer + offset, str.data(), str.length());
    offset += str.length();
  }

  static void read_string(const uint8_t *buffer, uint32_t &offset, std::string &str) {
    uint32_t length;
    offset = read_value(buffer, offset, length);
    str.assign(reinterpret_cast<const char *>(buffer + offset), length);
    offset += length;
  }

  template <typename T>
  static void write_value(uint8_t *buffer, uint32_t &offset, const T *value, size_t count) {
    size_t byte_size = count * sizeof(T);
    std::memcpy(buffer + offset, value, byte_size);
    offset += byte_size;
  }
};
} // namespace tpipeline