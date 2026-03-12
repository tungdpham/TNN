#pragma once

#include <cstdint>

#include "device/dptr.hpp"
#include "endian.hpp"

namespace tnn {

enum class CompressionType : uint8_t { NONE = 0, ZSTD = 1, QUANTIZATION = 2 };

enum class PacketType : uint8_t { DATA_FRAGMENT = 0, MSG_PREPARE = 1, MSG_READY_TO_WRITE = 2 };

// Fixed header at the start of each packet.
struct PacketHeader {
  // Packet information
  uint8_t PROTOCOL_VERSION = 1;
  PacketType type = PacketType::DATA_FRAGMENT;  // Type of packet
  Endianness endianess;                         // 1 for little-endian, 0 for big-endian
  uint64_t packet_length = 0;  // Length of the rest of the packet (excluding fixed header part)

  // For fragmentation
  uint64_t msg_length = 0;     // Total length for the entire message.
  uint64_t msg_serial_id = 0;  // Unique ID for the entire message.
  uint32_t packet_offset = 0;  // packet index for fragmentation
  uint32_t total_packets = 1;  // total packets for fragmentation

  CompressionType compression_type = CompressionType::NONE;

  PacketHeader()
      : endianess(get_system_endianness()) {}

  PacketHeader(PacketType t, uint64_t packet_len, uint64_t msg_len, uint32_t pkt_offset,
               uint32_t total_pkts, CompressionType comp_type = CompressionType::NONE)
      : type(t),
        packet_length(packet_len),
        msg_length(msg_len),
        packet_offset(pkt_offset),
        total_packets(total_pkts),
        compression_type(comp_type) {
    endianess = get_system_endianness();
  }

  static constexpr uint64_t size() {
    return sizeof(uint8_t) +         // PROTOCOL_VERSION
           sizeof(PacketType) +      // type
           sizeof(Endianness) +      // endianess
           sizeof(uint64_t) +        // packet_length
           sizeof(uint64_t) +        // msg_length
           sizeof(uint64_t) +        // msg_serial_id
           sizeof(uint32_t) +        // packet_offset
           sizeof(uint32_t) +        // total_packets
           sizeof(CompressionType);  // compression_type
  }

  template <typename Archiver>
  void archive(Archiver &archiver) {
    archiver & PROTOCOL_VERSION & type & endianess & packet_length & msg_length & msg_serial_id &
        packet_offset & total_packets & compression_type;
  }
};

struct Packet {
  PacketHeader header;
  dptr data;
};

}  // namespace tnn