#pragma once

#include "endian.hpp"
#include <cstdint>

namespace tnn {

enum class CompressionType : uint8_t { NONE = 0, ZSTD = 1, QUANTIZATION = 2 };

// Fixed header at the start of each packet.
struct PacketHeader {
  // Packet information
  uint8_t PROTOCOL_VERSION = 1;
  Endianness endianess; // 1 for little-endian, 0 for big-endian
  uint64_t length = 0;  // Length of the rest of the packet (excluding fixed header part)

  // For fragmentation (not implemented yet)
  uint64_t msg_length = 0;    // Total length for the entire message.
  uint64_t msg_serial_id = 0; // Unique ID for the entire message.
  uint32_t packet_offset = 0; // packet index for fragmentation
  uint32_t total_packets = 1; // total packets for fragmentation

  CompressionType compression_type = CompressionType::NONE;

  PacketHeader() : endianess(get_system_endianness()) {}

  PacketHeader(uint64_t len, uint64_t msg_len, uint32_t pkt_offset, uint32_t total_pkts,
               CompressionType comp_type = CompressionType::NONE)
      : length(len), msg_length(msg_len), packet_offset(pkt_offset), total_packets(total_pkts),
        compression_type(comp_type) {
    endianess = get_system_endianness();
  }

  static constexpr uint64_t size() {
    return sizeof(uint8_t) +        // PROTOCOL_VERSION
           sizeof(Endianness) +     // endianess
           sizeof(uint64_t) +       // length
           sizeof(uint64_t) +       // msg_length
           sizeof(uint64_t) +       // msg_serial_id
           sizeof(uint32_t) +       // packet_offset
           sizeof(uint32_t) +       // total_packets
           sizeof(CompressionType); // compression_type
  }
};
} // namespace tnn